// The fp64 validation suite (docs/phase-2-plan.md, steps 3 and 5), run on the
// GPU through the NVRTC batch kernels: raymarcher --validate [--out <dir>].
#pragma once

#include <string>
#include <vector>

#include "nvrtc_tracer.h"

struct SystemStrings {
  std::string gpu, arch, driver, cuda_driver_api, nvrtc;
};

/** Runs every check (generated_dir holds the JS fixtures), writes <out_dir>/<check>.json and <out_dir>/fp64-suite.json; returns 0 when all pass. */
int run_validation(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys,
                   const std::string& out_dir, const std::string& generated_dir);

/** Phase 4 step 4: total-flux gradient of the forward model vs supersampled FD (raymarcher --image-gradients [--out f]). */
int run_image_gradients(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys,
                        const std::string& out_path);

/** Phase 4 step 5, gate 1: image-loss gradient (with the edge term) vs FD of supersampled images (raymarcher --gate1 [--out f]). */
int run_gate1(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys, const std::string& out_path);
/** Gate-1 diagnosis: dense dual Jacobian vs per-sample FD at several h, smooth vs branch-changing samples (raymarcher --gate1-diag). */
int run_gate1_diag(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog);

/** Phase 4: plain vs symbolic-tangent vs dual-number batch kernels on Kerr (raymarcher --bench-tangents [--out f]). */
int run_tangent_benchmark(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys,
                          const std::string& out_path);
