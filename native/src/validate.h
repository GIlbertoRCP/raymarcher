// The fp64 validation suite (docs/phase-2-plan.md, steps 3 and 5), run on the
// GPU through the NVRTC batch kernels: raymarcher --validate [--out <dir>].
#pragma once

#include <string>
#include <vector>

#include "nvrtc_tracer.h"

struct SystemStrings {
  std::string gpu, arch, driver, cuda_driver_api, nvrtc;
};

/** Runs every check, writes <out_dir>/<check>.json and <out_dir>/fp64-suite.json; returns 0 when all pass. */
int run_validation(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys,
                   const std::string& out_dir);
