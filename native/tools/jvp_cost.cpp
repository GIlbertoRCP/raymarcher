// Phase 4, step 1 measurement: what the compiler's metric JVP costs on the GPU.
//
// For each catalog metric and precision, compiles with NVRTC (sm of the device)
//   eval_kernel   g and d g at a point (metric_eval)
//   jvp_kernel<P> g, d g and P tangent directions (metric_jvp, compiler/src/tangent.ts)
// and reports NVRTC compile time, registers, local (spill) bytes, and the time
// per call over 2^20 points (median of 15 launches). No fast math, no register
// cap: these are the plain costs, not the tuned tracer configuration.
//
//   raymarcher_jvp_cost [out.json]

#include <cuda.h>
#include <nvml.h>
#include <nvrtc.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef RM_NATIVE_DIR
#error "RM_NATIVE_DIR must point at native/"
#endif

#define CU(call)                                                               \
  do {                                                                         \
    CUresult r_ = (call);                                                      \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char* s_ = nullptr;                                                \
      cuGetErrorString(r_, &s_);                                               \
      std::fprintf(stderr, "%s failed: %s\n", #call, s_ ? s_ : "?");           \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static const char* KERNEL = R"(
#include "metric.cuh"
using M = rm::RM_METRIC::Metric;
typedef RM_REAL Real;
constexpr int NP = M::PARAM_COUNT > 0 ? M::PARAM_COUNT : 1;

extern "C" __global__ void eval_kernel(const Real* xs, const Real* params, Real* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Real x[4] = {xs[4 * i], xs[4 * i + 1], xs[4 * i + 2], xs[4 * i + 3]};
  Real g[16], dg[64];
  M::eval(x, params, g, dg);
  Real s = 0;
  for (int k = 0; k < 16; ++k) s += g[k];
  for (int k = 0; k < 64; ++k) s += dg[k];
  out[i] = s;
}

extern "C" __global__ void jvp_kernel(const Real* xs, const Real* params, const Real* dirs, Real* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Real x[4] = {xs[4 * i], xs[4 * i + 1], xs[4 * i + 2], xs[4 * i + 3]};
  Real dx[4 * RM_P], dp[NP * RM_P];
  for (int k = 0; k < 4 * RM_P; ++k) dx[k] = dirs[k] * x[k & 3];
  for (int k = 0; k < NP * RM_P; ++k) dp[k] = dirs[4 * RM_P + k];
  Real g[16], dg[64], tg[16 * RM_P], tdg[64 * RM_P];
  M::jvp<Real, RM_P>(x, params, dx, dp, g, dg, tg, tdg);
  Real s = 0;
  for (int k = 0; k < 16; ++k) s += g[k];
  for (int k = 0; k < 64; ++k) s += dg[k];
  for (int k = 0; k < 16 * RM_P; ++k) s += tg[k];
  for (int k = 0; k < 64 * RM_P; ++k) s += tdg[k];
  out[i] = s;
}
)";

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static std::string identifier(std::string id) {
  for (char& c : id)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) c = '_';
  return id;
}

struct Compiled {
  CUmodule module = nullptr;
  double compile_ms = 0;
  size_t ptx_bytes = 0;
};

static Compiled compile(const std::string& header, const std::string& id, bool fp64, int P, const std::string& arch) {
  const char* headers[] = {header.c_str()};
  const char* names[] = {"metric.cuh"};
  const std::string o_arch = "--gpu-architecture=" + arch;
  const std::string o_real = std::string("-DRM_REAL=") + (fp64 ? "double" : "float");
  const std::string o_metric = "-DRM_METRIC=" + identifier(id);
  const std::string o_p = "-DRM_P=" + std::to_string(P);
  const char* opts[] = {"--std=c++17", o_arch.c_str(), o_real.c_str(), o_metric.c_str(), o_p.c_str()};
  const auto t0 = std::chrono::steady_clock::now();
  nvrtcProgram prog;
  nvrtcCreateProgram(&prog, KERNEL, "jvp_cost.cu", 1, headers, names);
  const nvrtcResult res = nvrtcCompileProgram(prog, 5, opts);
  if (res != NVRTC_SUCCESS) {
    size_t n = 0;
    nvrtcGetProgramLogSize(prog, &n);
    std::string log(n, '\0');
    nvrtcGetProgramLog(prog, &log[0]);
    std::fprintf(stderr, "NVRTC failed for %s:\n%s\n", id.c_str(), log.c_str());
    std::exit(1);
  }
  Compiled c;
  nvrtcGetPTXSize(prog, &c.ptx_bytes);
  size_t size = 0;
  nvrtcGetCUBINSize(prog, &size);
  std::vector<char> cubin(size);
  nvrtcGetCUBIN(prog, cubin.data());
  nvrtcDestroyProgram(&prog);
  c.compile_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  CU(cuModuleLoadData(&c.module, cubin.data()));
  return c;
}

struct KernelCost {
  int registers = 0, local_bytes = 0;
  double ns_per_call = 0;
};

static KernelCost measure(CUfunction f, std::vector<void*> args, int n) {
  KernelCost k;
  CU(cuFuncGetAttribute(&k.registers, CU_FUNC_ATTRIBUTE_NUM_REGS, f));
  CU(cuFuncGetAttribute(&k.local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f));
  CUevent a, b;
  CU(cuEventCreate(&a, 0));
  CU(cuEventCreate(&b, 0));
  const int block = 256, grid = (n + block - 1) / block;
  std::vector<double> ms;
  for (int rep = 0; rep < 18; ++rep) {
    CU(cuEventRecord(a, nullptr));
    CU(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, nullptr, args.data(), nullptr));
    CU(cuEventRecord(b, nullptr));
    CU(cuEventSynchronize(b));
    float t = 0;
    CU(cuEventElapsedTime(&t, a, b));
    if (rep >= 3) ms.push_back(t);  // three warm-up launches
  }
  std::sort(ms.begin(), ms.end());
  k.ns_per_call = ms[ms.size() / 2] * 1e6 / n;
  cuEventDestroy(a);
  cuEventDestroy(b);
  return k;
}

int main(int argc, char** argv) {
  const std::string out_path = argc > 1 ? argv[1] : "";
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  char name[256] = {};
  cuDeviceGetName(name, sizeof name, dev);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  const std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
  char driver[96] = "unknown";
  if (nvmlInit() == NVML_SUCCESS) {
    nvmlSystemGetDriverVersion(driver, sizeof driver);
    nvmlShutdown();
  }
  int nvrtc_major = 0, nvrtc_minor = 0;
  nvrtcVersion(&nvrtc_major, &nvrtc_minor);

  const int n = 1 << 20;
  // Points at radius 5..40 (a fixed pseudo-random set) and directions in [-1, 1].
  std::vector<double> xs(4 * n), dirs(64);
  unsigned s = 12345;
  auto rnd = [&] { return (s = s * 1664525u + 1013904223u) / 4294967296.0; };
  for (int i = 0; i < n; ++i) {
    const double r = 5 + 35 * rnd(), th = 3.14159265 * rnd(), ph = 6.2831853 * rnd();
    xs[4 * i] = 10 * rnd();
    xs[4 * i + 1] = r * std::sin(th) * std::cos(ph);
    xs[4 * i + 2] = r * std::sin(th) * std::sin(ph);
    xs[4 * i + 3] = r * std::cos(th);
  }
  for (double& d : dirs) d = 2 * rnd() - 1;

  const std::vector<std::string> ids = {"schwarzschild", "kerr", "reissner-nordstrom", "minkowski"};
  const std::vector<std::vector<double>> params = {{1}, {1, 0.94}, {1, 0.6}, {1}};
  std::ostringstream json;
  json << "{\n  \"check\": \"phase4-step1-jvp-cost\",\n";
  json << "  \"gpu\": \"" << name << "\", \"arch\": \"" << arch << "\", \"driver\": \"" << driver << "\", \"nvrtc\": \"" << nvrtc_major
       << "." << nvrtc_minor << "\",\n";
  json << "  \"points\": " << n << ", \"block\": 256, \"options\": \"--std=c++17, no fast math, no register cap\",\n";
  json << "  \"note\": \"ns_per_call is kernel time / points, median of 15 launches after 3 warm-ups; the GPU may carry background load\",\n";
  json << "  \"results\": [\n";
  bool first = true;
  for (size_t m = 0; m < ids.size(); ++m) {
    const std::string header = read_file(std::string(RM_NATIVE_DIR) + "/generated/" + ids[m] + ".cuh");
    for (bool fp64 : {false, true}) {
      const size_t es = fp64 ? 8 : 4;
      auto upload = [&](const std::vector<double>& v) {
        CUdeviceptr p;
        CU(cuMemAlloc(&p, v.size() * es + 8));
        if (fp64) CU(cuMemcpyHtoD(p, v.data(), v.size() * 8));
        else {
          std::vector<float> f(v.size());
          for (size_t k = 0; k < v.size(); ++k) f[k] = static_cast<float>(v[k]);
          CU(cuMemcpyHtoD(p, f.data(), f.size() * 4));
        }
        return p;
      };
      CUdeviceptr d_x = upload(xs), d_p = upload(params[m]), d_dirs = upload(dirs), d_out;
      CU(cuMemAlloc(&d_out, n * es));
      int nn = n;
      KernelCost eval;
      double eval_compile = 0;
      for (int P : {1, 2, 3, 4}) {
        Compiled c = compile(header, ids[m], fp64, P, arch);
        CUfunction fe, fj;
        CU(cuModuleGetFunction(&fe, c.module, "eval_kernel"));
        CU(cuModuleGetFunction(&fj, c.module, "jvp_kernel"));
        if (P == 1) {
          eval = measure(fe, {&d_x, &d_p, &d_out, &nn}, n);
          eval_compile = c.compile_ms;
        }
        const KernelCost jvp = measure(fj, {&d_x, &d_p, &d_dirs, &d_out, &nn}, n);
        char line[640];
        std::snprintf(line, sizeof line,
                      "    {\"metric\": \"%s\", \"precision\": \"%s\", \"P\": %d, \"compile_ms\": %.0f, \"ptx_bytes\": %zu, "
                      "\"eval\": {\"registers\": %d, \"local_bytes\": %d, \"ns_per_call\": %.3f}, "
                      "\"jvp\": {\"registers\": %d, \"local_bytes\": %d, \"ns_per_call\": %.3f}, \"jvp_over_eval\": %.2f}",
                      ids[m].c_str(), fp64 ? "fp64" : "fp32", P, c.compile_ms, c.ptx_bytes, eval.registers, eval.local_bytes,
                      eval.ns_per_call, jvp.registers, jvp.local_bytes, jvp.ns_per_call, jvp.ns_per_call / eval.ns_per_call);
        json << (first ? "" : ",\n") << line;
        first = false;
        std::printf("%-18s %s P=%d  compile %5.0f ms  eval %3d regs %4d B local %7.3f ns | jvp %3d regs %5d B local %8.3f ns  (x%.2f)\n",
                    ids[m].c_str(), fp64 ? "fp64" : "fp32", P, c.compile_ms, eval.registers, eval.local_bytes, eval.ns_per_call,
                    jvp.registers, jvp.local_bytes, jvp.ns_per_call, jvp.ns_per_call / eval.ns_per_call);
        (void)eval_compile;
        cuModuleUnload(c.module);
      }
      cuMemFree(d_x);
      cuMemFree(d_p);
      cuMemFree(d_dirs);
      cuMemFree(d_out);
    }
  }
  json << "\n  ]\n}\n";
  if (!out_path.empty()) {
    std::ofstream(out_path) << json.str();
    std::printf("wrote %s\n", out_path.c_str());
  }
  cuDevicePrimaryCtxRelease(dev);
  return 0;
}
