// Generated from spacetime "Reissner–Nordström" (metrics/reissner-nordstrom.st) by compiler/src/backend-cuda.ts.
// Do not edit; run `npm run export-cuda` instead.
#pragma once

#ifndef RM_HD
#if defined(__CUDACC__)
#define RM_HD __host__ __device__
#else
#define RM_HD
#endif
#endif

#ifndef __CUDACC_RTC__
#include <cmath>
#endif

namespace rm {
namespace reissner_nordstrom {

#ifndef __CUDACC_RTC__
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
#endif

constexpr const char* NAME = "Reissner–Nordström";
constexpr int PARAM_COUNT = 2;
constexpr bool HAS_HORIZON = true;
constexpr bool HAS_DISK = true;

template <typename Real>
RM_HD inline void metric_eval(const Real x[4], const Real* params, Real g[16], Real dg[64]) {
  (void)x;
  (void)params;
  const Real t10 = x[1] * x[1];
  const Real t11 = x[2] * x[2];
  const Real t13 = x[3] * x[3];
  const Real t14 = t10 + t11 + t13;
  const Real t15 = sqrt(t14);
  const Real t17 = params[0] * Real(2.0) * t15;
  const Real t8 = params[0] * params[1];
  const Real t18 = t8 * t8;
  const Real t20 = t18 * Real(-1.0);
  const Real t21 = t17 + t20;
  const Real t23 = Real(1.0) / (t15 * t15);
  const Real t24 = t21 * t23;
  const Real t29 = Real(-1.0) + t24;
  const Real t25 = Real(1.0) / (t15);
  const Real t31 = x[1] * Real(-1.0) * t21 * t23 * t25;
  const Real t32 = x[2] * Real(-1.0) * t21 * t23 * t25;
  const Real t33 = x[3] * Real(-1.0) * t21 * t23 * t25;
  const Real t26 = x[1] * t25;
  const Real t34 = t26 * t26;
  const Real t35 = t21 * t23 * t34;
  const Real t36 = Real(1.0) + t35;
  const Real t38 = x[1] * x[2] * t21 * t23 * t25 * t25;
  const Real t39 = x[1] * x[3] * t21 * t23 * t25 * t25;
  const Real t27 = x[2] * t25;
  const Real t40 = t27 * t27;
  const Real t41 = t21 * t23 * t40;
  const Real t42 = Real(1.0) + t41;
  const Real t44 = x[2] * x[3] * t21 * t23 * t25 * t25;
  const Real t28 = x[3] * t25;
  const Real t45 = t28 * t28;
  const Real t46 = t21 * t23 * t45;
  const Real t47 = Real(1.0) + t46;
  const Real t99 = x[1] * params[0] * Real(2.0) * t23 * t25;
  const Real t96 = Real(1.0) / (t15 * t15 * t15);
  const Real t101 = x[1] * t21 * t25 * Real(-2.0) * t96;
  const Real t102 = t99 + t101;
  const Real t103 = Real(-1.0) * t21 * t23 * t25;
  const Real t104 = x[1] * x[1] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t105 = x[1] * x[1] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t107 = x[1] * x[1] * t21 * t23 * t23 * t25;
  const Real t108 = t103 + t104 + t105 + t107;
  const Real t109 = x[1] * x[2] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t110 = x[1] * x[2] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t111 = x[1] * x[2] * t21 * t23 * t23 * t25;
  const Real t112 = t109 + t110 + t111;
  const Real t113 = x[1] * x[3] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t114 = x[1] * x[3] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t115 = x[1] * x[3] * t21 * t23 * t23 * t25;
  const Real t116 = t113 + t114 + t115;
  const Real t117 = x[1] * params[0] * Real(2.0) * t23 * t25 * t34;
  const Real t118 = x[1] * t21 * t25 * t34 * Real(-2.0) * t96;
  const Real t119 = x[1] * x[1] * Real(-1.0) * t23 * t25;
  const Real t120 = t25 + t119;
  const Real t122 = x[1] * Real(2.0) * t21 * t23 * t25 * t120;
  const Real t123 = t117 + t118 + t122;
  const Real t124 = x[2] * t21 * t23 * t25 * t25;
  const Real t125 = x[1] * x[1] * x[2] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t126 = x[1] * x[1] * x[2] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t127 = x[1] * x[1] * x[2] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t128 = t124 + t125 + t126 + t127 + t127;
  const Real t129 = x[3] * t21 * t23 * t25 * t25;
  const Real t130 = x[1] * x[1] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t131 = x[1] * x[1] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t132 = x[1] * x[1] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t133 = t129 + t130 + t131 + t132 + t132;
  const Real t134 = x[1] * params[0] * Real(2.0) * t23 * t25 * t40;
  const Real t135 = x[1] * t21 * t25 * t40 * Real(-2.0) * t96;
  const Real t138 = x[1] * x[2] * x[2] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t139 = t134 + t135 + t138;
  const Real t140 = x[1] * x[2] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t141 = x[1] * x[2] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t142 = x[1] * x[2] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t143 = t140 + t141 + t142 + t142;
  const Real t144 = x[1] * params[0] * Real(2.0) * t23 * t25 * t45;
  const Real t145 = x[1] * t21 * t25 * t45 * Real(-2.0) * t96;
  const Real t148 = x[1] * x[3] * x[3] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t149 = t144 + t145 + t148;
  const Real t152 = x[2] * params[0] * Real(2.0) * t23 * t25;
  const Real t154 = x[2] * t21 * t25 * Real(-2.0) * t96;
  const Real t155 = t152 + t154;
  const Real t157 = x[2] * x[2] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t158 = x[2] * x[2] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t159 = x[2] * x[2] * t21 * t23 * t23 * t25;
  const Real t160 = t103 + t157 + t158 + t159;
  const Real t161 = x[2] * x[3] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t162 = x[2] * x[3] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t163 = x[2] * x[3] * t21 * t23 * t23 * t25;
  const Real t164 = t161 + t162 + t163;
  const Real t165 = x[2] * params[0] * Real(2.0) * t23 * t25 * t34;
  const Real t166 = x[2] * t21 * t25 * t34 * Real(-2.0) * t96;
  const Real t168 = x[1] * x[1] * x[2] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t169 = t165 + t166 + t168;
  const Real t170 = x[1] * t21 * t23 * t25 * t25;
  const Real t171 = x[1] * x[2] * x[2] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t172 = x[1] * x[2] * x[2] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t173 = x[1] * x[2] * x[2] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t174 = t170 + t171 + t172 + t173 + t173;
  const Real t175 = x[2] * params[0] * Real(2.0) * t23 * t25 * t40;
  const Real t176 = x[2] * t21 * t25 * t40 * Real(-2.0) * t96;
  const Real t177 = x[2] * x[2] * Real(-1.0) * t23 * t25;
  const Real t178 = t25 + t177;
  const Real t180 = x[2] * Real(2.0) * t21 * t23 * t25 * t178;
  const Real t181 = t175 + t176 + t180;
  const Real t182 = x[2] * x[2] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t183 = x[2] * x[2] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t184 = x[2] * x[2] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t185 = t129 + t182 + t183 + t184 + t184;
  const Real t186 = x[2] * params[0] * Real(2.0) * t23 * t25 * t45;
  const Real t187 = x[2] * t21 * t25 * t45 * Real(-2.0) * t96;
  const Real t190 = x[2] * x[3] * x[3] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t191 = t186 + t187 + t190;
  const Real t194 = x[3] * params[0] * Real(2.0) * t23 * t25;
  const Real t196 = x[3] * t21 * t25 * Real(-2.0) * t96;
  const Real t197 = t194 + t196;
  const Real t199 = x[3] * x[3] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t200 = x[3] * x[3] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t201 = x[3] * x[3] * t21 * t23 * t23 * t25;
  const Real t202 = t103 + t199 + t200 + t201;
  const Real t203 = x[3] * params[0] * Real(2.0) * t23 * t25 * t34;
  const Real t204 = x[3] * t21 * t25 * t34 * Real(-2.0) * t96;
  const Real t206 = x[1] * x[1] * x[3] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t207 = t203 + t204 + t206;
  const Real t208 = x[1] * x[3] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t209 = x[1] * x[3] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t210 = x[1] * x[3] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t211 = t170 + t208 + t209 + t210 + t210;
  const Real t212 = x[3] * params[0] * Real(2.0) * t23 * t25 * t40;
  const Real t213 = x[3] * t21 * t25 * t40 * Real(-2.0) * t96;
  const Real t215 = x[2] * x[2] * x[3] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t216 = t212 + t213 + t215;
  const Real t217 = x[2] * x[3] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t218 = x[2] * x[3] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t219 = x[2] * x[3] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t220 = t124 + t217 + t218 + t219 + t219;
  const Real t221 = x[3] * params[0] * Real(2.0) * t23 * t25 * t45;
  const Real t222 = x[3] * t21 * t25 * t45 * Real(-2.0) * t96;
  const Real t223 = x[3] * x[3] * Real(-1.0) * t23 * t25;
  const Real t224 = t25 + t223;
  const Real t226 = x[3] * Real(2.0) * t21 * t23 * t25 * t224;
  const Real t227 = t221 + t222 + t226;
  g[0] = t29;
  g[1] = t31;
  g[2] = t32;
  g[3] = t33;
  g[4] = t31;
  g[5] = t36;
  g[6] = t38;
  g[7] = t39;
  g[8] = t32;
  g[9] = t38;
  g[10] = t42;
  g[11] = t44;
  g[12] = t33;
  g[13] = t39;
  g[14] = t44;
  g[15] = t47;
  dg[0] = Real(0.0);
  dg[1] = Real(0.0);
  dg[2] = Real(0.0);
  dg[3] = Real(0.0);
  dg[4] = Real(0.0);
  dg[5] = Real(0.0);
  dg[6] = Real(0.0);
  dg[7] = Real(0.0);
  dg[8] = Real(0.0);
  dg[9] = Real(0.0);
  dg[10] = Real(0.0);
  dg[11] = Real(0.0);
  dg[12] = Real(0.0);
  dg[13] = Real(0.0);
  dg[14] = Real(0.0);
  dg[15] = Real(0.0);
  dg[16] = t102;
  dg[17] = t108;
  dg[18] = t112;
  dg[19] = t116;
  dg[20] = t108;
  dg[21] = t123;
  dg[22] = t128;
  dg[23] = t133;
  dg[24] = t112;
  dg[25] = t128;
  dg[26] = t139;
  dg[27] = t143;
  dg[28] = t116;
  dg[29] = t133;
  dg[30] = t143;
  dg[31] = t149;
  dg[32] = t155;
  dg[33] = t112;
  dg[34] = t160;
  dg[35] = t164;
  dg[36] = t112;
  dg[37] = t169;
  dg[38] = t174;
  dg[39] = t143;
  dg[40] = t160;
  dg[41] = t174;
  dg[42] = t181;
  dg[43] = t185;
  dg[44] = t164;
  dg[45] = t143;
  dg[46] = t185;
  dg[47] = t191;
  dg[48] = t197;
  dg[49] = t116;
  dg[50] = t164;
  dg[51] = t202;
  dg[52] = t116;
  dg[53] = t207;
  dg[54] = t143;
  dg[55] = t211;
  dg[56] = t164;
  dg[57] = t143;
  dg[58] = t216;
  dg[59] = t220;
  dg[60] = t202;
  dg[61] = t211;
  dg[62] = t220;
  dg[63] = t227;
}

template <typename Real>
RM_HD inline Real geodesic_rhs(const Real y[8], const Real* params, Real dy[8]) {
  const Real* x = y;
  (void)x;
  (void)params;
  const Real t10 = x[1] * x[1];
  const Real t11 = x[2] * x[2];
  const Real t13 = x[3] * x[3];
  const Real t14 = t10 + t11 + t13;
  const Real t15 = sqrt(t14);
  const Real t17 = params[0] * Real(2.0) * t15;
  const Real t8 = params[0] * params[1];
  const Real t18 = t8 * t8;
  const Real t20 = t18 * Real(-1.0);
  const Real t21 = t17 + t20;
  const Real t23 = Real(1.0) / (t15 * t15);
  const Real t24 = t21 * t23;
  const Real t29 = Real(-1.0) + t24;
  const Real t25 = Real(1.0) / (t15);
  const Real t31 = x[1] * Real(-1.0) * t21 * t23 * t25;
  const Real t32 = x[2] * Real(-1.0) * t21 * t23 * t25;
  const Real t33 = x[3] * Real(-1.0) * t21 * t23 * t25;
  const Real t26 = x[1] * t25;
  const Real t34 = t26 * t26;
  const Real t35 = t21 * t23 * t34;
  const Real t36 = Real(1.0) + t35;
  const Real t38 = x[1] * x[2] * t21 * t23 * t25 * t25;
  const Real t39 = x[1] * x[3] * t21 * t23 * t25 * t25;
  const Real t27 = x[2] * t25;
  const Real t40 = t27 * t27;
  const Real t41 = t21 * t23 * t40;
  const Real t42 = Real(1.0) + t41;
  const Real t44 = x[2] * x[3] * t21 * t23 * t25 * t25;
  const Real t28 = x[3] * t25;
  const Real t45 = t28 * t28;
  const Real t46 = t21 * t23 * t45;
  const Real t47 = Real(1.0) + t46;
  const Real t99 = x[1] * params[0] * Real(2.0) * t23 * t25;
  const Real t96 = Real(1.0) / (t15 * t15 * t15);
  const Real t101 = x[1] * t21 * t25 * Real(-2.0) * t96;
  const Real t102 = t99 + t101;
  const Real t103 = Real(-1.0) * t21 * t23 * t25;
  const Real t104 = x[1] * x[1] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t105 = x[1] * x[1] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t107 = x[1] * x[1] * t21 * t23 * t23 * t25;
  const Real t108 = t103 + t104 + t105 + t107;
  const Real t109 = x[1] * x[2] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t110 = x[1] * x[2] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t111 = x[1] * x[2] * t21 * t23 * t23 * t25;
  const Real t112 = t109 + t110 + t111;
  const Real t113 = x[1] * x[3] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t114 = x[1] * x[3] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t115 = x[1] * x[3] * t21 * t23 * t23 * t25;
  const Real t116 = t113 + t114 + t115;
  const Real t117 = x[1] * params[0] * Real(2.0) * t23 * t25 * t34;
  const Real t118 = x[1] * t21 * t25 * t34 * Real(-2.0) * t96;
  const Real t119 = x[1] * x[1] * Real(-1.0) * t23 * t25;
  const Real t120 = t25 + t119;
  const Real t122 = x[1] * Real(2.0) * t21 * t23 * t25 * t120;
  const Real t123 = t117 + t118 + t122;
  const Real t124 = x[2] * t21 * t23 * t25 * t25;
  const Real t125 = x[1] * x[1] * x[2] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t126 = x[1] * x[1] * x[2] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t127 = x[1] * x[1] * x[2] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t128 = t124 + t125 + t126 + t127 + t127;
  const Real t129 = x[3] * t21 * t23 * t25 * t25;
  const Real t130 = x[1] * x[1] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t131 = x[1] * x[1] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t132 = x[1] * x[1] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t133 = t129 + t130 + t131 + t132 + t132;
  const Real t134 = x[1] * params[0] * Real(2.0) * t23 * t25 * t40;
  const Real t135 = x[1] * t21 * t25 * t40 * Real(-2.0) * t96;
  const Real t138 = x[1] * x[2] * x[2] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t139 = t134 + t135 + t138;
  const Real t140 = x[1] * x[2] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t141 = x[1] * x[2] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t142 = x[1] * x[2] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t143 = t140 + t141 + t142 + t142;
  const Real t144 = x[1] * params[0] * Real(2.0) * t23 * t25 * t45;
  const Real t145 = x[1] * t21 * t25 * t45 * Real(-2.0) * t96;
  const Real t148 = x[1] * x[3] * x[3] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t149 = t144 + t145 + t148;
  const Real t152 = x[2] * params[0] * Real(2.0) * t23 * t25;
  const Real t154 = x[2] * t21 * t25 * Real(-2.0) * t96;
  const Real t155 = t152 + t154;
  const Real t157 = x[2] * x[2] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t158 = x[2] * x[2] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t159 = x[2] * x[2] * t21 * t23 * t23 * t25;
  const Real t160 = t103 + t157 + t158 + t159;
  const Real t161 = x[2] * x[3] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t162 = x[2] * x[3] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t163 = x[2] * x[3] * t21 * t23 * t23 * t25;
  const Real t164 = t161 + t162 + t163;
  const Real t165 = x[2] * params[0] * Real(2.0) * t23 * t25 * t34;
  const Real t166 = x[2] * t21 * t25 * t34 * Real(-2.0) * t96;
  const Real t168 = x[1] * x[1] * x[2] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t169 = t165 + t166 + t168;
  const Real t170 = x[1] * t21 * t23 * t25 * t25;
  const Real t171 = x[1] * x[2] * x[2] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t172 = x[1] * x[2] * x[2] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t173 = x[1] * x[2] * x[2] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t174 = t170 + t171 + t172 + t173 + t173;
  const Real t175 = x[2] * params[0] * Real(2.0) * t23 * t25 * t40;
  const Real t176 = x[2] * t21 * t25 * t40 * Real(-2.0) * t96;
  const Real t177 = x[2] * x[2] * Real(-1.0) * t23 * t25;
  const Real t178 = t25 + t177;
  const Real t180 = x[2] * Real(2.0) * t21 * t23 * t25 * t178;
  const Real t181 = t175 + t176 + t180;
  const Real t182 = x[2] * x[2] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t183 = x[2] * x[2] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t184 = x[2] * x[2] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t185 = t129 + t182 + t183 + t184 + t184;
  const Real t186 = x[2] * params[0] * Real(2.0) * t23 * t25 * t45;
  const Real t187 = x[2] * t21 * t25 * t45 * Real(-2.0) * t96;
  const Real t190 = x[2] * x[3] * x[3] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t191 = t186 + t187 + t190;
  const Real t194 = x[3] * params[0] * Real(2.0) * t23 * t25;
  const Real t196 = x[3] * t21 * t25 * Real(-2.0) * t96;
  const Real t197 = t194 + t196;
  const Real t199 = x[3] * x[3] * params[0] * t23 * t25 * t25 * Real(-2.0);
  const Real t200 = x[3] * x[3] * Real(2.0) * t21 * t25 * t25 * t96;
  const Real t201 = x[3] * x[3] * t21 * t23 * t23 * t25;
  const Real t202 = t103 + t199 + t200 + t201;
  const Real t203 = x[3] * params[0] * Real(2.0) * t23 * t25 * t34;
  const Real t204 = x[3] * t21 * t25 * t34 * Real(-2.0) * t96;
  const Real t206 = x[1] * x[1] * x[3] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t207 = t203 + t204 + t206;
  const Real t208 = x[1] * x[3] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t209 = x[1] * x[3] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t210 = x[1] * x[3] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t211 = t170 + t208 + t209 + t210 + t210;
  const Real t212 = x[3] * params[0] * Real(2.0) * t23 * t25 * t40;
  const Real t213 = x[3] * t21 * t25 * t40 * Real(-2.0) * t96;
  const Real t215 = x[2] * x[2] * x[3] * t21 * t23 * t23 * t25 * t25 * Real(-2.0);
  const Real t216 = t212 + t213 + t215;
  const Real t217 = x[2] * x[3] * x[3] * params[0] * Real(2.0) * t23 * t25 * t25 * t25;
  const Real t218 = x[2] * x[3] * x[3] * t21 * t25 * t25 * t25 * Real(-2.0) * t96;
  const Real t219 = x[2] * x[3] * x[3] * Real(-1.0) * t21 * t23 * t23 * t25 * t25;
  const Real t220 = t124 + t217 + t218 + t219 + t219;
  const Real t221 = x[3] * params[0] * Real(2.0) * t23 * t25 * t45;
  const Real t222 = x[3] * t21 * t25 * t45 * Real(-2.0) * t96;
  const Real t223 = x[3] * x[3] * Real(-1.0) * t23 * t25;
  const Real t224 = t25 + t223;
  const Real t226 = x[3] * Real(2.0) * t21 * t23 * t25 * t224;
  const Real t227 = t221 + t222 + t226;
  const Real a00 = t29, a01 = t31, a02 = t32, a03 = t33;
  const Real a10 = t31, a11 = t36, a12 = t38, a13 = t39;
  const Real a20 = t32, a21 = t38, a22 = t42, a23 = t44;
  const Real a30 = t33, a31 = t39, a32 = t44, a33 = t47;
  const Real b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const Real b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const Real b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const Real b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const Real s = Real(1) / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  const Real i00 = (a11 * b11 - a12 * b10 + a13 * b09) * s;
  const Real i01 = (a02 * b10 - a01 * b11 - a03 * b09) * s;
  const Real i02 = (a31 * b05 - a32 * b04 + a33 * b03) * s;
  const Real i03 = (a22 * b04 - a21 * b05 - a23 * b03) * s;
  const Real i11 = (a00 * b11 - a02 * b08 + a03 * b07) * s;
  const Real i12 = (a32 * b02 - a30 * b05 - a33 * b01) * s;
  const Real i13 = (a20 * b05 - a22 * b02 + a23 * b01) * s;
  const Real i22 = (a30 * b04 - a31 * b02 + a33 * b00) * s;
  const Real i23 = (a21 * b02 - a20 * b04 - a23 * b00) * s;
  const Real i33 = (a20 * b03 - a21 * b01 + a22 * b00) * s;
  const Real k0 = i00 * y[4] + i01 * y[5] + i02 * y[6] + i03 * y[7];
  const Real k1 = i01 * y[4] + i11 * y[5] + i12 * y[6] + i13 * y[7];
  const Real k2 = i02 * y[4] + i12 * y[5] + i22 * y[6] + i23 * y[7];
  const Real k3 = i03 * y[4] + i13 * y[5] + i23 * y[6] + i33 * y[7];
  dy[0] = k0;
  dy[1] = k1;
  dy[2] = k2;
  dy[3] = k3;
  const Real kk00 = k0 * k0;
  const Real kk01 = k0 * k1;
  const Real kk02 = k0 * k2;
  const Real kk03 = k0 * k3;
  const Real kk11 = k1 * k1;
  const Real kk12 = k1 * k2;
  const Real kk13 = k1 * k3;
  const Real kk22 = k2 * k2;
  const Real kk23 = k2 * k3;
  const Real kk33 = k3 * k3;
  dy[4] = Real(0.0);
  dy[5] = Real(0.5) * t102 * kk00 + t108 * kk01 + t112 * kk02 + t116 * kk03 + Real(0.5) * t123 * kk11 + t128 * kk12 + t133 * kk13 + Real(0.5) * t139 * kk22 + t143 * kk23 + Real(0.5) * t149 * kk33;
  dy[6] = Real(0.5) * t155 * kk00 + t112 * kk01 + t160 * kk02 + t164 * kk03 + Real(0.5) * t169 * kk11 + t174 * kk12 + t143 * kk13 + Real(0.5) * t181 * kk22 + t185 * kk23 + Real(0.5) * t191 * kk33;
  dy[7] = Real(0.5) * t197 * kk00 + t116 * kk01 + t164 * kk02 + t202 * kk03 + Real(0.5) * t207 * kk11 + t143 * kk12 + t211 * kk13 + Real(0.5) * t216 * kk22 + t220 * kk23 + Real(0.5) * t227 * kk33;
  return Real(0.5) * (k0 * y[4] + k1 * y[5] + k2 * y[6] + k3 * y[7]);
}

template <typename Real>
RM_HD inline Real horizon_value(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t10 = x[1] * x[1];
  const Real t11 = x[2] * x[2];
  const Real t13 = x[3] * x[3];
  const Real t14 = t10 + t11 + t13;
  const Real t15 = sqrt(t14);
  const Real t8 = params[0] * params[1];
  const Real t18 = t8 * t8;
  const Real t20 = t18 * Real(-1.0);
  const Real t48 = params[0] * params[0];
  const Real t49 = t20 + t48;
  const Real t50 = sqrt(t49);
  const Real t51 = params[0] + t50;
  const Real t55 = t51 * Real(-1.01);
  const Real t56 = t15 + t55;
  return t56;
}

template <typename Real>
RM_HD inline Real disk_radius(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t10 = x[1] * x[1];
  const Real t11 = x[2] * x[2];
  const Real t13 = x[3] * x[3];
  const Real t14 = t10 + t11 + t13;
  const Real t15 = sqrt(t14);
  return t15;
}

template <typename Real>
RM_HD inline Real disk_omega(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t10 = x[1] * x[1];
  const Real t11 = x[2] * x[2];
  const Real t13 = x[3] * x[3];
  const Real t14 = t10 + t11 + t13;
  const Real t15 = sqrt(t14);
  const Real t23 = Real(1.0) / (t15 * t15);
  const Real t8 = params[0] * params[1];
  const Real t18 = t8 * t8;
  const Real t20 = t18 * Real(-1.0);
  const Real t87 = params[0] * t15;
  const Real t88 = t20 + t87;
  const Real t89 = sqrt(t88);
  const Real t90 = t23 * t89;
  return t90;
}

template <typename Real>
RM_HD inline Real disk_inner(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t57 = params[1] * params[1];
  const Real t62 = t57 * Real(-9.0);
  const Real t64 = params[1] * params[1] * params[1] * params[1];
  const Real t65 = Real(2.0) * t64;
  const Real t71 = t57 * Real(-4.0);
  const Real t72 = Real(5.0) + t71;
  const Real t73 = Real(-1.0) * t57;
  const Real t74 = Real(1.0) + t73;
  const Real t75 = t72 * t74;
  const Real t76 = sqrt(t75);
  const Real t77 = t57 * t76;
  const Real t78 = Real(8.0) + t62 + t65 + t77;
  const Real t81 = pow(t78, Real(0.3333333333333333));
  const Real t83 = Real(-1.0) * t57 * t76;
  const Real t84 = Real(8.0) + t62 + t65 + t83;
  const Real t85 = pow(t84, Real(0.3333333333333333));
  const Real t86 = Real(2.0) + t81 + t85;
  const Real t91 = params[0] * t86;
  return t91;
}

template <typename Real>
RM_HD inline Real disk_outer(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t93 = params[0] * Real(20.0);
  return t93;
}

template <typename Real>
RM_HD inline Real length_scale(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  return params[0];
}

// The functions above as one type, so generic code can take the metric as a
// template parameter (rm::trace_ray<rm::kerr::Metric>(...)).
struct Metric {
  static constexpr int PARAM_COUNT = 2;
  static constexpr bool HAS_HORIZON = true;
  static constexpr bool HAS_DISK = true;
  template <typename Real>
  RM_HD static void eval(const Real x[4], const Real* p, Real g[16], Real dg[64]) { metric_eval(x, p, g, dg); }
  template <typename Real>
  RM_HD static Real rhs(const Real y[8], const Real* p, Real dy[8]) { return reissner_nordstrom::geodesic_rhs(y, p, dy); }
  template <typename Real>
  RM_HD static Real horizon_value(const Real x[4], const Real* p) { return reissner_nordstrom::horizon_value(x, p); }
  template <typename Real>
  RM_HD static Real disk_radius(const Real x[4], const Real* p) { return reissner_nordstrom::disk_radius(x, p); }
  template <typename Real>
  RM_HD static Real disk_omega(const Real x[4], const Real* p) { return reissner_nordstrom::disk_omega(x, p); }
  template <typename Real>
  RM_HD static Real disk_inner(const Real x[4], const Real* p) { return reissner_nordstrom::disk_inner(x, p); }
  template <typename Real>
  RM_HD static Real disk_outer(const Real x[4], const Real* p) { return reissner_nordstrom::disk_outer(x, p); }
  template <typename Real>
  RM_HD static Real length_scale(const Real x[4], const Real* p) { return reissner_nordstrom::length_scale(x, p); }
};

}  // namespace reissner_nordstrom
}  // namespace rm
