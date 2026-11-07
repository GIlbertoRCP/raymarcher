// Generated from spacetime "Schwarzschild" (metrics/schwarzschild.st) by compiler/src/backend-cuda.ts.
// Do not edit; run `npm run export-cuda` instead.
#pragma once

#ifndef RM_HD
#if defined(__CUDACC__)
#define RM_HD __host__ __device__
#else
#define RM_HD
#endif
#endif

#ifndef RM_UNROLL
#if defined(__CUDACC__)
#define RM_UNROLL _Pragma("unroll")
#else
#define RM_UNROLL
#endif
#endif

#ifndef __CUDACC_RTC__
#include <cmath>
#endif

namespace rm {
namespace schwarzschild {

#ifndef __CUDACC_RTC__
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
#endif

constexpr const char* NAME = "Schwarzschild";
constexpr int PARAM_COUNT = 1;
constexpr bool HAS_HORIZON = true;
constexpr bool HAS_DISK = true;

template <typename Real>
RM_HD inline void metric_eval(const Real x[4], const Real* params, Real g[16], Real dg[64]) {
  (void)x;
  (void)params;
  const Real t8 = x[1] * x[1];
  const Real t9 = x[2] * x[2];
  const Real t11 = x[3] * x[3];
  const Real t12 = t8 + t9 + t11;
  const Real t13 = sqrt(t12);
  const Real t15 = Real(1.0) / (t13);
  const Real t16 = params[0] * Real(2.0) * t15;
  const Real t21 = t16 + Real(-1.0);
  const Real t24 = x[1] * params[0] * t15 * t15 * Real(-2.0);
  const Real t25 = x[2] * params[0] * t15 * t15 * Real(-2.0);
  const Real t26 = x[3] * params[0] * t15 * t15 * Real(-2.0);
  const Real t17 = x[1] * t15;
  const Real t27 = t17 * t17;
  const Real t28 = params[0] * Real(2.0) * t15 * t27;
  const Real t29 = Real(1.0) + t28;
  const Real t31 = x[1] * x[2] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t32 = x[1] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t18 = x[2] * t15;
  const Real t33 = t18 * t18;
  const Real t34 = params[0] * Real(2.0) * t15 * t33;
  const Real t35 = Real(1.0) + t34;
  const Real t37 = x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t19 = x[3] * t15;
  const Real t38 = t19 * t19;
  const Real t39 = params[0] * Real(2.0) * t15 * t38;
  const Real t40 = Real(1.0) + t39;
  const Real t56 = Real(1.0) / (t13 * t13);
  const Real t60 = x[1] * params[0] * t15 * Real(-2.0) * t56;
  const Real t61 = params[0] * t15 * t15 * Real(-2.0);
  const Real t62 = x[1] * x[1] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t63 = t61 + t62 + t62;
  const Real t64 = x[1] * x[2] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t65 = t64 + t64;
  const Real t66 = x[1] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t67 = t66 + t66;
  const Real t68 = x[1] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t69 = x[1] * x[1] * t15 * Real(-1.0) * t56;
  const Real t70 = t15 + t69;
  const Real t73 = x[1] * params[0] * t15 * t15 * t70 * Real(4.0);
  const Real t74 = t68 + t73;
  const Real t75 = x[2] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t76 = x[1] * x[1] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t77 = t75 + t76 + t76 + t76;
  const Real t78 = x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t79 = x[1] * x[1] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t80 = t78 + t79 + t79 + t79;
  const Real t81 = x[1] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t85 = x[1] * x[2] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t86 = t81 + t85;
  const Real t87 = x[1] * x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t88 = t87 + t87 + t87;
  const Real t89 = x[1] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t92 = x[1] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t93 = t89 + t92;
  const Real t96 = x[2] * params[0] * t15 * Real(-2.0) * t56;
  const Real t97 = x[2] * x[2] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t98 = t61 + t97 + t97;
  const Real t99 = x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t100 = t99 + t99;
  const Real t101 = x[2] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t103 = x[1] * x[1] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t104 = t101 + t103;
  const Real t105 = x[1] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t106 = x[1] * x[2] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t107 = t105 + t106 + t106 + t106;
  const Real t108 = x[2] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t109 = x[2] * x[2] * t15 * Real(-1.0) * t56;
  const Real t110 = t15 + t109;
  const Real t112 = x[2] * params[0] * t15 * t15 * Real(4.0) * t110;
  const Real t113 = t108 + t112;
  const Real t114 = x[2] * x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t115 = t78 + t114 + t114 + t114;
  const Real t116 = x[2] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t119 = x[2] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t120 = t116 + t119;
  const Real t123 = x[3] * params[0] * t15 * Real(-2.0) * t56;
  const Real t124 = x[3] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t125 = t61 + t124 + t124;
  const Real t126 = x[3] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t128 = x[1] * x[1] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t129 = t126 + t128;
  const Real t130 = x[1] * x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t131 = t105 + t130 + t130 + t130;
  const Real t132 = x[3] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t134 = x[2] * x[2] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t135 = t132 + t134;
  const Real t136 = x[2] * x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t137 = t75 + t136 + t136 + t136;
  const Real t138 = x[3] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t139 = x[3] * x[3] * t15 * Real(-1.0) * t56;
  const Real t140 = t15 + t139;
  const Real t142 = x[3] * params[0] * t15 * t15 * Real(4.0) * t140;
  const Real t143 = t138 + t142;
  g[0] = t21;
  g[1] = t24;
  g[2] = t25;
  g[3] = t26;
  g[4] = t24;
  g[5] = t29;
  g[6] = t31;
  g[7] = t32;
  g[8] = t25;
  g[9] = t31;
  g[10] = t35;
  g[11] = t37;
  g[12] = t26;
  g[13] = t32;
  g[14] = t37;
  g[15] = t40;
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
  dg[16] = t60;
  dg[17] = t63;
  dg[18] = t65;
  dg[19] = t67;
  dg[20] = t63;
  dg[21] = t74;
  dg[22] = t77;
  dg[23] = t80;
  dg[24] = t65;
  dg[25] = t77;
  dg[26] = t86;
  dg[27] = t88;
  dg[28] = t67;
  dg[29] = t80;
  dg[30] = t88;
  dg[31] = t93;
  dg[32] = t96;
  dg[33] = t65;
  dg[34] = t98;
  dg[35] = t100;
  dg[36] = t65;
  dg[37] = t104;
  dg[38] = t107;
  dg[39] = t88;
  dg[40] = t98;
  dg[41] = t107;
  dg[42] = t113;
  dg[43] = t115;
  dg[44] = t100;
  dg[45] = t88;
  dg[46] = t115;
  dg[47] = t120;
  dg[48] = t123;
  dg[49] = t67;
  dg[50] = t100;
  dg[51] = t125;
  dg[52] = t67;
  dg[53] = t129;
  dg[54] = t88;
  dg[55] = t131;
  dg[56] = t100;
  dg[57] = t88;
  dg[58] = t135;
  dg[59] = t137;
  dg[60] = t125;
  dg[61] = t131;
  dg[62] = t137;
  dg[63] = t143;
}

template <typename Real>
RM_HD inline Real geodesic_rhs(const Real y[8], const Real* params, Real dy[8]) {
  const Real* x = y;
  (void)x;
  (void)params;
  const Real t8 = x[1] * x[1];
  const Real t9 = x[2] * x[2];
  const Real t11 = x[3] * x[3];
  const Real t12 = t8 + t9 + t11;
  const Real t13 = sqrt(t12);
  const Real t15 = Real(1.0) / (t13);
  const Real t16 = params[0] * Real(2.0) * t15;
  const Real t21 = t16 + Real(-1.0);
  const Real t24 = x[1] * params[0] * t15 * t15 * Real(-2.0);
  const Real t25 = x[2] * params[0] * t15 * t15 * Real(-2.0);
  const Real t26 = x[3] * params[0] * t15 * t15 * Real(-2.0);
  const Real t17 = x[1] * t15;
  const Real t27 = t17 * t17;
  const Real t28 = params[0] * Real(2.0) * t15 * t27;
  const Real t29 = Real(1.0) + t28;
  const Real t31 = x[1] * x[2] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t32 = x[1] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t18 = x[2] * t15;
  const Real t33 = t18 * t18;
  const Real t34 = params[0] * Real(2.0) * t15 * t33;
  const Real t35 = Real(1.0) + t34;
  const Real t37 = x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t19 = x[3] * t15;
  const Real t38 = t19 * t19;
  const Real t39 = params[0] * Real(2.0) * t15 * t38;
  const Real t40 = Real(1.0) + t39;
  const Real t56 = Real(1.0) / (t13 * t13);
  const Real t60 = x[1] * params[0] * t15 * Real(-2.0) * t56;
  const Real t61 = params[0] * t15 * t15 * Real(-2.0);
  const Real t62 = x[1] * x[1] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t63 = t61 + t62 + t62;
  const Real t64 = x[1] * x[2] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t65 = t64 + t64;
  const Real t66 = x[1] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t67 = t66 + t66;
  const Real t68 = x[1] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t69 = x[1] * x[1] * t15 * Real(-1.0) * t56;
  const Real t70 = t15 + t69;
  const Real t73 = x[1] * params[0] * t15 * t15 * t70 * Real(4.0);
  const Real t74 = t68 + t73;
  const Real t75 = x[2] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t76 = x[1] * x[1] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t77 = t75 + t76 + t76 + t76;
  const Real t78 = x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t79 = x[1] * x[1] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t80 = t78 + t79 + t79 + t79;
  const Real t81 = x[1] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t85 = x[1] * x[2] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t86 = t81 + t85;
  const Real t87 = x[1] * x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t88 = t87 + t87 + t87;
  const Real t89 = x[1] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t92 = x[1] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t93 = t89 + t92;
  const Real t96 = x[2] * params[0] * t15 * Real(-2.0) * t56;
  const Real t97 = x[2] * x[2] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t98 = t61 + t97 + t97;
  const Real t99 = x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t100 = t99 + t99;
  const Real t101 = x[2] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t103 = x[1] * x[1] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t104 = t101 + t103;
  const Real t105 = x[1] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t106 = x[1] * x[2] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t107 = t105 + t106 + t106 + t106;
  const Real t108 = x[2] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t109 = x[2] * x[2] * t15 * Real(-1.0) * t56;
  const Real t110 = t15 + t109;
  const Real t112 = x[2] * params[0] * t15 * t15 * Real(4.0) * t110;
  const Real t113 = t108 + t112;
  const Real t114 = x[2] * x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t115 = t78 + t114 + t114 + t114;
  const Real t116 = x[2] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t119 = x[2] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t120 = t116 + t119;
  const Real t123 = x[3] * params[0] * t15 * Real(-2.0) * t56;
  const Real t124 = x[3] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t125 = t61 + t124 + t124;
  const Real t126 = x[3] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t128 = x[1] * x[1] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t129 = t126 + t128;
  const Real t130 = x[1] * x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t131 = t105 + t130 + t130 + t130;
  const Real t132 = x[3] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t134 = x[2] * x[2] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t135 = t132 + t134;
  const Real t136 = x[2] * x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t137 = t75 + t136 + t136 + t136;
  const Real t138 = x[3] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t139 = x[3] * x[3] * t15 * Real(-1.0) * t56;
  const Real t140 = t15 + t139;
  const Real t142 = x[3] * params[0] * t15 * t15 * Real(4.0) * t140;
  const Real t143 = t138 + t142;
  const Real a00 = t21, a01 = t24, a02 = t25, a03 = t26;
  const Real a10 = t24, a11 = t29, a12 = t31, a13 = t32;
  const Real a20 = t25, a21 = t31, a22 = t35, a23 = t37;
  const Real a30 = t26, a31 = t32, a32 = t37, a33 = t40;
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
  dy[5] = Real(0.5) * t60 * kk00 + t63 * kk01 + t65 * kk02 + t67 * kk03 + Real(0.5) * t74 * kk11 + t77 * kk12 + t80 * kk13 + Real(0.5) * t86 * kk22 + t88 * kk23 + Real(0.5) * t93 * kk33;
  dy[6] = Real(0.5) * t96 * kk00 + t65 * kk01 + t98 * kk02 + t100 * kk03 + Real(0.5) * t104 * kk11 + t107 * kk12 + t88 * kk13 + Real(0.5) * t113 * kk22 + t115 * kk23 + Real(0.5) * t120 * kk33;
  dy[7] = Real(0.5) * t123 * kk00 + t67 * kk01 + t100 * kk02 + t125 * kk03 + Real(0.5) * t129 * kk11 + t88 * kk12 + t131 * kk13 + Real(0.5) * t135 * kk22 + t137 * kk23 + Real(0.5) * t143 * kk33;
  return Real(0.5) * (k0 * y[4] + k1 * y[5] + k2 * y[6] + k3 * y[7]);
}

template <typename Real>
RM_HD inline Real horizon_value(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t8 = x[1] * x[1];
  const Real t9 = x[2] * x[2];
  const Real t11 = x[3] * x[3];
  const Real t12 = t8 + t9 + t11;
  const Real t13 = sqrt(t12);
  const Real t45 = params[0] * Real(-2.02);
  const Real t46 = t13 + t45;
  return t46;
}

template <typename Real>
RM_HD inline Real disk_radius(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t8 = x[1] * x[1];
  const Real t9 = x[2] * x[2];
  const Real t11 = x[3] * x[3];
  const Real t12 = t8 + t9 + t11;
  const Real t13 = sqrt(t12);
  return t13;
}

template <typename Real>
RM_HD inline Real disk_omega(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t47 = sqrt(params[0]);
  const Real t8 = x[1] * x[1];
  const Real t9 = x[2] * x[2];
  const Real t11 = x[3] * x[3];
  const Real t12 = t8 + t9 + t11;
  const Real t13 = sqrt(t12);
  const Real t50 = Real(1.0) / (t13 * sqrt(t13));
  const Real t51 = t47 * t50;
  return t51;
}

template <typename Real>
RM_HD inline Real disk_inner(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t53 = params[0] * Real(6.0);
  return t53;
}

template <typename Real>
RM_HD inline Real disk_outer(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t55 = params[0] * Real(20.0);
  return t55;
}

template <typename Real>
RM_HD inline Real length_scale(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  return params[0];
}

// Jacobian-vector product of metric_eval along P directions (compiler/src/tangent.ts):
// dx[4 d + i], dparams[PARAM_COUNT d + k] -> tg[16 d + 4 mu + nu], tdg[64 d + 16 c + 4 mu + nu].
template <typename Real, int P>
RM_HD inline void metric_jvp(const Real x[4], const Real* params, const Real* dx, const Real* dparams,
                             Real g[16], Real dg[64], Real* tg, Real* tdg) {
  (void)x;
  (void)params;
  (void)dx;
  (void)dparams;
  constexpr int NP = 1;
  (void)NP;
  const Real t8 = x[1] * x[1];
  const Real t9 = x[2] * x[2];
  const Real t11 = x[3] * x[3];
  const Real t12 = t8 + t9 + t11;
  const Real t13 = sqrt(t12);
  const Real t15 = Real(1.0) / (t13);
  const Real t16 = params[0] * Real(2.0) * t15;
  const Real t21 = t16 + Real(-1.0);
  const Real t24 = x[1] * params[0] * t15 * t15 * Real(-2.0);
  const Real t25 = x[2] * params[0] * t15 * t15 * Real(-2.0);
  const Real t26 = x[3] * params[0] * t15 * t15 * Real(-2.0);
  const Real t17 = x[1] * t15;
  const Real t27 = t17 * t17;
  const Real t28 = params[0] * Real(2.0) * t15 * t27;
  const Real t29 = Real(1.0) + t28;
  const Real t31 = x[1] * x[2] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t32 = x[1] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t18 = x[2] * t15;
  const Real t33 = t18 * t18;
  const Real t34 = params[0] * Real(2.0) * t15 * t33;
  const Real t35 = Real(1.0) + t34;
  const Real t37 = x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t19 = x[3] * t15;
  const Real t38 = t19 * t19;
  const Real t39 = params[0] * Real(2.0) * t15 * t38;
  const Real t40 = Real(1.0) + t39;
  const Real t56 = Real(1.0) / (t13 * t13);
  const Real t60 = x[1] * params[0] * t15 * Real(-2.0) * t56;
  const Real t61 = params[0] * t15 * t15 * Real(-2.0);
  const Real t62 = x[1] * x[1] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t63 = t61 + t62 + t62;
  const Real t64 = x[1] * x[2] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t65 = t64 + t64;
  const Real t66 = x[1] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t67 = t66 + t66;
  const Real t68 = x[1] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t69 = x[1] * x[1] * t15 * Real(-1.0) * t56;
  const Real t70 = t15 + t69;
  const Real t73 = x[1] * params[0] * t15 * t15 * t70 * Real(4.0);
  const Real t74 = t68 + t73;
  const Real t75 = x[2] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t76 = x[1] * x[1] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t77 = t75 + t76 + t76 + t76;
  const Real t78 = x[3] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t79 = x[1] * x[1] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t80 = t78 + t79 + t79 + t79;
  const Real t81 = x[1] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t85 = x[1] * x[2] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t86 = t81 + t85;
  const Real t87 = x[1] * x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t88 = t87 + t87 + t87;
  const Real t89 = x[1] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t92 = x[1] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t93 = t89 + t92;
  const Real t96 = x[2] * params[0] * t15 * Real(-2.0) * t56;
  const Real t97 = x[2] * x[2] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t98 = t61 + t97 + t97;
  const Real t99 = x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t100 = t99 + t99;
  const Real t101 = x[2] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t103 = x[1] * x[1] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t104 = t101 + t103;
  const Real t105 = x[1] * params[0] * Real(2.0) * t15 * t15 * t15;
  const Real t106 = x[1] * x[2] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t107 = t105 + t106 + t106 + t106;
  const Real t108 = x[2] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t109 = x[2] * x[2] * t15 * Real(-1.0) * t56;
  const Real t110 = t15 + t109;
  const Real t112 = x[2] * params[0] * t15 * t15 * Real(4.0) * t110;
  const Real t113 = t108 + t112;
  const Real t114 = x[2] * x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t115 = t78 + t114 + t114 + t114;
  const Real t116 = x[2] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t119 = x[2] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t120 = t116 + t119;
  const Real t123 = x[3] * params[0] * t15 * Real(-2.0) * t56;
  const Real t124 = x[3] * x[3] * params[0] * Real(2.0) * t15 * t15 * t56;
  const Real t125 = t61 + t124 + t124;
  const Real t126 = x[3] * params[0] * t15 * Real(-2.0) * t27 * t56;
  const Real t128 = x[1] * x[1] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t129 = t126 + t128;
  const Real t130 = x[1] * x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t131 = t105 + t130 + t130 + t130;
  const Real t132 = x[3] * params[0] * t15 * Real(-2.0) * t33 * t56;
  const Real t134 = x[2] * x[2] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0);
  const Real t135 = t132 + t134;
  const Real t136 = x[2] * x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56;
  const Real t137 = t75 + t136 + t136 + t136;
  const Real t138 = x[3] * params[0] * t15 * Real(-2.0) * t38 * t56;
  const Real t139 = x[3] * x[3] * t15 * Real(-1.0) * t56;
  const Real t140 = t15 + t139;
  const Real t142 = x[3] * params[0] * t15 * t15 * Real(4.0) * t140;
  const Real t143 = t138 + t142;
  const Real t212 = Real(1.0) / (t13 * t13 * t13);
  g[0] = t21;
  g[1] = t24;
  g[2] = t25;
  g[3] = t26;
  g[4] = t24;
  g[5] = t29;
  g[6] = t31;
  g[7] = t32;
  g[8] = t25;
  g[9] = t31;
  g[10] = t35;
  g[11] = t37;
  g[12] = t26;
  g[13] = t32;
  g[14] = t37;
  g[15] = t40;
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
  dg[16] = t60;
  dg[17] = t63;
  dg[18] = t65;
  dg[19] = t67;
  dg[20] = t63;
  dg[21] = t74;
  dg[22] = t77;
  dg[23] = t80;
  dg[24] = t65;
  dg[25] = t77;
  dg[26] = t86;
  dg[27] = t88;
  dg[28] = t67;
  dg[29] = t80;
  dg[30] = t88;
  dg[31] = t93;
  dg[32] = t96;
  dg[33] = t65;
  dg[34] = t98;
  dg[35] = t100;
  dg[36] = t65;
  dg[37] = t104;
  dg[38] = t107;
  dg[39] = t88;
  dg[40] = t98;
  dg[41] = t107;
  dg[42] = t113;
  dg[43] = t115;
  dg[44] = t100;
  dg[45] = t88;
  dg[46] = t115;
  dg[47] = t120;
  dg[48] = t123;
  dg[49] = t67;
  dg[50] = t100;
  dg[51] = t125;
  dg[52] = t67;
  dg[53] = t129;
  dg[54] = t88;
  dg[55] = t131;
  dg[56] = t100;
  dg[57] = t88;
  dg[58] = t135;
  dg[59] = t137;
  dg[60] = t125;
  dg[61] = t131;
  dg[62] = t137;
  dg[63] = t143;
RM_UNROLL
  for (int d = 0; d < P; ++d) {
    const Real d145 = Real(2.0) * t15 * dparams[NP * d + 0];
    const Real d147 = x[1] * Real(2.0) * dx[4 * d + 1];
    const Real d149 = x[2] * Real(2.0) * dx[4 * d + 2];
    const Real d151 = x[3] * Real(2.0) * dx[4 * d + 3];
    const Real d152 = d147 + d149 + d151;
    const Real d156 = params[0] * t15 * Real(-1.0) * t56 * d152;
    const Real d157 = d145 + d156;
    const Real d158 = params[0] * t15 * t15 * Real(-2.0) * dx[4 * d + 1];
    const Real d159 = x[1] * t15 * t15 * Real(-2.0) * dparams[NP * d + 0];
    const Real d160 = x[1] * params[0] * t15 * t15 * t56 * d152;
    const Real d161 = d158 + d159 + d160 + d160;
    const Real d162 = params[0] * t15 * t15 * Real(-2.0) * dx[4 * d + 2];
    const Real d163 = x[2] * t15 * t15 * Real(-2.0) * dparams[NP * d + 0];
    const Real d164 = x[2] * params[0] * t15 * t15 * t56 * d152;
    const Real d165 = d162 + d163 + d164 + d164;
    const Real d166 = params[0] * t15 * t15 * Real(-2.0) * dx[4 * d + 3];
    const Real d167 = x[3] * t15 * t15 * Real(-2.0) * dparams[NP * d + 0];
    const Real d168 = x[3] * params[0] * t15 * t15 * t56 * d152;
    const Real d169 = d166 + d167 + d168 + d168;
    const Real d170 = Real(2.0) * t15 * t27 * dparams[NP * d + 0];
    const Real d171 = params[0] * t15 * Real(-1.0) * t27 * t56 * d152;
    const Real d172 = t15 * dx[4 * d + 1];
    const Real d173 = x[1] * t15 * t56 * d152 * Real(-0.5);
    const Real d174 = d172 + d173;
    const Real d176 = x[1] * params[0] * t15 * t15 * Real(4.0) * d174;
    const Real d177 = d170 + d171 + d176;
    const Real d178 = x[2] * params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 1];
    const Real d179 = x[1] * params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 2];
    const Real d180 = x[1] * x[2] * Real(2.0) * t15 * t15 * t15 * dparams[NP * d + 0];
    const Real d181 = x[1] * x[2] * params[0] * t15 * t15 * t15 * Real(-1.0) * t56 * d152;
    const Real d182 = d178 + d179 + d180 + d181 + d181 + d181;
    const Real d183 = x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 1];
    const Real d184 = x[1] * params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 3];
    const Real d185 = x[1] * x[3] * Real(2.0) * t15 * t15 * t15 * dparams[NP * d + 0];
    const Real d186 = x[1] * x[3] * params[0] * t15 * t15 * t15 * Real(-1.0) * t56 * d152;
    const Real d187 = d183 + d184 + d185 + d186 + d186 + d186;
    const Real d188 = Real(2.0) * t15 * t33 * dparams[NP * d + 0];
    const Real d189 = params[0] * t15 * Real(-1.0) * t33 * t56 * d152;
    const Real d190 = t15 * dx[4 * d + 2];
    const Real d191 = x[2] * t15 * t56 * d152 * Real(-0.5);
    const Real d192 = d190 + d191;
    const Real d194 = x[2] * params[0] * t15 * t15 * Real(4.0) * d192;
    const Real d195 = d188 + d189 + d194;
    const Real d196 = x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 2];
    const Real d197 = x[2] * params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 3];
    const Real d198 = x[2] * x[3] * Real(2.0) * t15 * t15 * t15 * dparams[NP * d + 0];
    const Real d199 = x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-1.0) * t56 * d152;
    const Real d200 = d196 + d197 + d198 + d199 + d199 + d199;
    const Real d201 = Real(2.0) * t15 * t38 * dparams[NP * d + 0];
    const Real d202 = params[0] * t15 * Real(-1.0) * t38 * t56 * d152;
    const Real d203 = t15 * dx[4 * d + 3];
    const Real d204 = x[3] * t15 * t56 * d152 * Real(-0.5);
    const Real d205 = d203 + d204;
    const Real d207 = x[3] * params[0] * t15 * t15 * Real(4.0) * d205;
    const Real d208 = d201 + d202 + d207;
    const Real d209 = params[0] * t15 * Real(-2.0) * t56 * dx[4 * d + 1];
    const Real d210 = x[1] * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d211 = x[1] * params[0] * t15 * t56 * t56 * d152;
    const Real d214 = x[1] * params[0] * Real(2.0) * t15 * t15 * d152 * t212;
    const Real d215 = d209 + d210 + d211 + d214;
    const Real d216 = t15 * t15 * Real(-2.0) * dparams[NP * d + 0];
    const Real d217 = params[0] * t15 * t15 * t56 * d152;
    const Real d219 = x[1] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 1];
    const Real d220 = x[1] * x[1] * Real(2.0) * t15 * t15 * t56 * dparams[NP * d + 0];
    const Real d221 = x[1] * x[1] * params[0] * t15 * t15 * Real(-1.0) * t56 * t56 * d152;
    const Real d222 = x[1] * x[1] * params[0] * t15 * t15 * t15 * Real(-2.0) * d152 * t212;
    const Real d224 = d216 + d217 + d217 + d219 + d219 + d219 + d219 + d220 + d220 + d221 + d221 + d221 + d221 + d222 + d222;
    const Real d225 = x[2] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 1];
    const Real d226 = x[1] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 2];
    const Real d227 = x[1] * x[2] * Real(2.0) * t15 * t15 * t56 * dparams[NP * d + 0];
    const Real d228 = x[1] * x[2] * params[0] * t15 * t15 * Real(-1.0) * t56 * t56 * d152;
    const Real d229 = x[1] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * d152 * t212;
    const Real d231 = d225 + d225 + d226 + d226 + d227 + d227 + d228 + d228 + d228 + d228 + d229 + d229;
    const Real d232 = x[3] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 1];
    const Real d233 = x[1] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 3];
    const Real d234 = x[1] * x[3] * Real(2.0) * t15 * t15 * t56 * dparams[NP * d + 0];
    const Real d235 = x[1] * x[3] * params[0] * t15 * t15 * Real(-1.0) * t56 * t56 * d152;
    const Real d236 = x[1] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * d152 * t212;
    const Real d238 = d232 + d232 + d233 + d233 + d234 + d234 + d235 + d235 + d235 + d235 + d236 + d236;
    const Real d239 = params[0] * t15 * Real(-2.0) * t27 * t56 * dx[4 * d + 1];
    const Real d240 = x[1] * t15 * Real(-2.0) * t27 * t56 * dparams[NP * d + 0];
    const Real d241 = x[1] * params[0] * t15 * t27 * t56 * t56 * d152;
    const Real d242 = x[1] * x[1] * params[0] * t15 * t15 * t56 * Real(-4.0) * d174;
    const Real d243 = x[1] * params[0] * Real(2.0) * t15 * t15 * t27 * d152 * t212;
    const Real d245 = params[0] * t15 * t15 * t70 * Real(4.0) * dx[4 * d + 1];
    const Real d246 = x[1] * t15 * t15 * t70 * Real(4.0) * dparams[NP * d + 0];
    const Real d247 = x[1] * params[0] * t15 * t15 * Real(-2.0) * t56 * t70 * d152;
    const Real d155 = t15 * t56 * d152 * Real(-0.5);
    const Real d248 = x[1] * t15 * Real(-1.0) * t56 * dx[4 * d + 1];
    const Real d249 = x[1] * x[1] * t15 * t56 * t56 * Real(0.5) * d152;
    const Real d250 = x[1] * x[1] * t15 * t15 * d152 * t212;
    const Real d252 = d155 + d248 + d248 + d249 + d250;
    const Real d253 = x[1] * params[0] * t15 * t15 * Real(4.0) * d252;
    const Real d255 = d239 + d240 + d241 + d242 + d243 + d245 + d246 + d247 + d247 + d253;
    const Real d256 = params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 2];
    const Real d257 = x[2] * Real(2.0) * t15 * t15 * t15 * dparams[NP * d + 0];
    const Real d258 = x[2] * params[0] * t15 * t15 * t15 * Real(-1.0) * t56 * d152;
    const Real d260 = x[1] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 1];
    const Real d261 = x[1] * x[1] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 2];
    const Real d262 = x[1] * x[1] * x[2] * t15 * t15 * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d263 = x[1] * x[1] * x[2] * params[0] * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d264 = x[1] * x[1] * x[2] * params[0] * Real(2.0) * t15 * t15 * t15 * t15 * d152 * t212;
    const Real d266 = d256 + d257 + d258 + d258 + d258 + d260 + d260 + d260 + d260 + d260 + d260 + d261 + d261 + d261 + d262 + d262 + d262 + d263 + d263 + d263 + d263 + d263 + d263 + d263 + d263 + d263 + d264 + d264 + d264;
    const Real d267 = params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 3];
    const Real d268 = x[3] * Real(2.0) * t15 * t15 * t15 * dparams[NP * d + 0];
    const Real d269 = x[3] * params[0] * t15 * t15 * t15 * Real(-1.0) * t56 * d152;
    const Real d271 = x[1] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 1];
    const Real d272 = x[1] * x[1] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 3];
    const Real d273 = x[1] * x[1] * x[3] * t15 * t15 * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d274 = x[1] * x[1] * x[3] * params[0] * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d275 = x[1] * x[1] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t15 * d152 * t212;
    const Real d277 = d267 + d268 + d269 + d269 + d269 + d271 + d271 + d271 + d271 + d271 + d271 + d272 + d272 + d272 + d273 + d273 + d273 + d274 + d274 + d274 + d274 + d274 + d274 + d274 + d274 + d274 + d275 + d275 + d275;
    const Real d278 = params[0] * t15 * Real(-2.0) * t33 * t56 * dx[4 * d + 1];
    const Real d279 = x[1] * t15 * Real(-2.0) * t33 * t56 * dparams[NP * d + 0];
    const Real d280 = x[1] * params[0] * t15 * t33 * t56 * t56 * d152;
    const Real d281 = x[1] * x[2] * params[0] * t15 * t15 * t56 * Real(-4.0) * d192;
    const Real d282 = x[1] * params[0] * Real(2.0) * t15 * t15 * t33 * d152 * t212;
    const Real d284 = x[2] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 1];
    const Real d285 = x[1] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 2];
    const Real d286 = x[1] * x[2] * x[2] * t15 * t15 * t15 * t56 * Real(-4.0) * dparams[NP * d + 0];
    const Real d287 = x[1] * x[2] * x[2] * params[0] * Real(2.0) * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d288 = x[1] * x[2] * x[2] * params[0] * t15 * t15 * t15 * t15 * Real(4.0) * d152 * t212;
    const Real d290 = d278 + d279 + d280 + d281 + d282 + d284 + d285 + d285 + d286 + d287 + d287 + d287 + d288;
    const Real d291 = x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 1];
    const Real d292 = x[1] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 2];
    const Real d293 = x[1] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 3];
    const Real d294 = x[1] * x[2] * x[3] * t15 * t15 * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d295 = x[1] * x[2] * x[3] * params[0] * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d296 = x[1] * x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t15 * d152 * t212;
    const Real d298 = d291 + d291 + d291 + d292 + d292 + d292 + d293 + d293 + d293 + d294 + d294 + d294 + d295 + d295 + d295 + d295 + d295 + d295 + d295 + d295 + d295 + d296 + d296 + d296;
    const Real d299 = params[0] * t15 * Real(-2.0) * t38 * t56 * dx[4 * d + 1];
    const Real d300 = x[1] * t15 * Real(-2.0) * t38 * t56 * dparams[NP * d + 0];
    const Real d301 = x[1] * params[0] * t15 * t38 * t56 * t56 * d152;
    const Real d302 = x[1] * x[3] * params[0] * t15 * t15 * t56 * Real(-4.0) * d205;
    const Real d303 = x[1] * params[0] * Real(2.0) * t15 * t15 * t38 * d152 * t212;
    const Real d305 = x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 1];
    const Real d306 = x[1] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 3];
    const Real d307 = x[1] * x[3] * x[3] * t15 * t15 * t15 * t56 * Real(-4.0) * dparams[NP * d + 0];
    const Real d308 = x[1] * x[3] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d309 = x[1] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t15 * Real(4.0) * d152 * t212;
    const Real d311 = d299 + d300 + d301 + d302 + d303 + d305 + d306 + d306 + d307 + d308 + d308 + d308 + d309;
    const Real d312 = params[0] * t15 * Real(-2.0) * t56 * dx[4 * d + 2];
    const Real d313 = x[2] * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d314 = x[2] * params[0] * t15 * t56 * t56 * d152;
    const Real d315 = x[2] * params[0] * Real(2.0) * t15 * t15 * d152 * t212;
    const Real d316 = d312 + d313 + d314 + d315;
    const Real d317 = x[2] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 2];
    const Real d318 = x[2] * x[2] * Real(2.0) * t15 * t15 * t56 * dparams[NP * d + 0];
    const Real d319 = x[2] * x[2] * params[0] * t15 * t15 * Real(-1.0) * t56 * t56 * d152;
    const Real d320 = x[2] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * d152 * t212;
    const Real d322 = d216 + d217 + d217 + d317 + d317 + d317 + d317 + d318 + d318 + d319 + d319 + d319 + d319 + d320 + d320;
    const Real d323 = x[3] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 2];
    const Real d324 = x[2] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 3];
    const Real d325 = x[2] * x[3] * Real(2.0) * t15 * t15 * t56 * dparams[NP * d + 0];
    const Real d326 = x[2] * x[3] * params[0] * t15 * t15 * Real(-1.0) * t56 * t56 * d152;
    const Real d327 = x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * d152 * t212;
    const Real d329 = d323 + d323 + d324 + d324 + d325 + d325 + d326 + d326 + d326 + d326 + d327 + d327;
    const Real d330 = params[0] * t15 * Real(-2.0) * t27 * t56 * dx[4 * d + 2];
    const Real d331 = x[2] * t15 * Real(-2.0) * t27 * t56 * dparams[NP * d + 0];
    const Real d332 = x[2] * params[0] * t15 * t27 * t56 * t56 * d152;
    const Real d333 = x[1] * x[2] * params[0] * t15 * t15 * t56 * Real(-4.0) * d174;
    const Real d334 = x[2] * params[0] * Real(2.0) * t15 * t15 * t27 * d152 * t212;
    const Real d336 = x[1] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 1];
    const Real d337 = x[1] * x[1] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 2];
    const Real d338 = x[1] * x[1] * x[2] * t15 * t15 * t15 * t56 * Real(-4.0) * dparams[NP * d + 0];
    const Real d339 = x[1] * x[1] * x[2] * params[0] * Real(2.0) * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d340 = x[1] * x[1] * x[2] * params[0] * t15 * t15 * t15 * t15 * Real(4.0) * d152 * t212;
    const Real d342 = d330 + d331 + d332 + d333 + d334 + d336 + d336 + d337 + d338 + d339 + d339 + d339 + d340;
    const Real d343 = params[0] * Real(2.0) * t15 * t15 * t15 * dx[4 * d + 1];
    const Real d344 = x[1] * Real(2.0) * t15 * t15 * t15 * dparams[NP * d + 0];
    const Real d345 = x[1] * params[0] * t15 * t15 * t15 * Real(-1.0) * t56 * d152;
    const Real d347 = x[2] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 1];
    const Real d348 = x[1] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 2];
    const Real d349 = x[1] * x[2] * x[2] * t15 * t15 * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d350 = x[1] * x[2] * x[2] * params[0] * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d351 = x[1] * x[2] * x[2] * params[0] * Real(2.0) * t15 * t15 * t15 * t15 * d152 * t212;
    const Real d353 = d343 + d344 + d345 + d345 + d345 + d347 + d347 + d347 + d348 + d348 + d348 + d348 + d348 + d348 + d349 + d349 + d349 + d350 + d350 + d350 + d350 + d350 + d350 + d350 + d350 + d350 + d351 + d351 + d351;
    const Real d354 = params[0] * t15 * Real(-2.0) * t33 * t56 * dx[4 * d + 2];
    const Real d355 = x[2] * t15 * Real(-2.0) * t33 * t56 * dparams[NP * d + 0];
    const Real d356 = x[2] * params[0] * t15 * t33 * t56 * t56 * d152;
    const Real d357 = x[2] * x[2] * params[0] * t15 * t15 * t56 * Real(-4.0) * d192;
    const Real d358 = x[2] * params[0] * Real(2.0) * t15 * t15 * t33 * d152 * t212;
    const Real d360 = params[0] * t15 * t15 * Real(4.0) * t110 * dx[4 * d + 2];
    const Real d361 = x[2] * t15 * t15 * Real(4.0) * t110 * dparams[NP * d + 0];
    const Real d362 = x[2] * params[0] * t15 * t15 * Real(-2.0) * t56 * t110 * d152;
    const Real d363 = x[2] * t15 * Real(-1.0) * t56 * dx[4 * d + 2];
    const Real d364 = x[2] * x[2] * t15 * t56 * t56 * Real(0.5) * d152;
    const Real d365 = x[2] * x[2] * t15 * t15 * d152 * t212;
    const Real d367 = d155 + d363 + d363 + d364 + d365;
    const Real d368 = x[2] * params[0] * t15 * t15 * Real(4.0) * d367;
    const Real d370 = d354 + d355 + d356 + d357 + d358 + d360 + d361 + d362 + d362 + d368;
    const Real d371 = x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 2];
    const Real d372 = x[2] * x[2] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 3];
    const Real d373 = x[2] * x[2] * x[3] * t15 * t15 * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d374 = x[2] * x[2] * x[3] * params[0] * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d375 = x[2] * x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t15 * d152 * t212;
    const Real d377 = d267 + d268 + d269 + d269 + d269 + d371 + d371 + d371 + d371 + d371 + d371 + d372 + d372 + d372 + d373 + d373 + d373 + d374 + d374 + d374 + d374 + d374 + d374 + d374 + d374 + d374 + d375 + d375 + d375;
    const Real d378 = params[0] * t15 * Real(-2.0) * t38 * t56 * dx[4 * d + 2];
    const Real d379 = x[2] * t15 * Real(-2.0) * t38 * t56 * dparams[NP * d + 0];
    const Real d380 = x[2] * params[0] * t15 * t38 * t56 * t56 * d152;
    const Real d381 = x[2] * x[3] * params[0] * t15 * t15 * t56 * Real(-4.0) * d205;
    const Real d382 = x[2] * params[0] * Real(2.0) * t15 * t15 * t38 * d152 * t212;
    const Real d384 = x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 2];
    const Real d385 = x[2] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 3];
    const Real d386 = x[2] * x[3] * x[3] * t15 * t15 * t15 * t56 * Real(-4.0) * dparams[NP * d + 0];
    const Real d387 = x[2] * x[3] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d388 = x[2] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t15 * Real(4.0) * d152 * t212;
    const Real d390 = d378 + d379 + d380 + d381 + d382 + d384 + d385 + d385 + d386 + d387 + d387 + d387 + d388;
    const Real d391 = params[0] * t15 * Real(-2.0) * t56 * dx[4 * d + 3];
    const Real d392 = x[3] * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d393 = x[3] * params[0] * t15 * t56 * t56 * d152;
    const Real d394 = x[3] * params[0] * Real(2.0) * t15 * t15 * d152 * t212;
    const Real d395 = d391 + d392 + d393 + d394;
    const Real d396 = x[3] * params[0] * Real(2.0) * t15 * t15 * t56 * dx[4 * d + 3];
    const Real d397 = x[3] * x[3] * Real(2.0) * t15 * t15 * t56 * dparams[NP * d + 0];
    const Real d398 = x[3] * x[3] * params[0] * t15 * t15 * Real(-1.0) * t56 * t56 * d152;
    const Real d399 = x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * d152 * t212;
    const Real d401 = d216 + d217 + d217 + d396 + d396 + d396 + d396 + d397 + d397 + d398 + d398 + d398 + d398 + d399 + d399;
    const Real d402 = params[0] * t15 * Real(-2.0) * t27 * t56 * dx[4 * d + 3];
    const Real d403 = x[3] * t15 * Real(-2.0) * t27 * t56 * dparams[NP * d + 0];
    const Real d404 = x[3] * params[0] * t15 * t27 * t56 * t56 * d152;
    const Real d405 = x[1] * x[3] * params[0] * t15 * t15 * t56 * Real(-4.0) * d174;
    const Real d406 = x[3] * params[0] * Real(2.0) * t15 * t15 * t27 * d152 * t212;
    const Real d408 = x[1] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 1];
    const Real d409 = x[1] * x[1] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 3];
    const Real d410 = x[1] * x[1] * x[3] * t15 * t15 * t15 * t56 * Real(-4.0) * dparams[NP * d + 0];
    const Real d411 = x[1] * x[1] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d412 = x[1] * x[1] * x[3] * params[0] * t15 * t15 * t15 * t15 * Real(4.0) * d152 * t212;
    const Real d414 = d402 + d403 + d404 + d405 + d406 + d408 + d408 + d409 + d410 + d411 + d411 + d411 + d412;
    const Real d415 = x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 1];
    const Real d416 = x[1] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 3];
    const Real d417 = x[1] * x[3] * x[3] * t15 * t15 * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d418 = x[1] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d419 = x[1] * x[3] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t15 * d152 * t212;
    const Real d421 = d343 + d344 + d345 + d345 + d345 + d415 + d415 + d415 + d416 + d416 + d416 + d416 + d416 + d416 + d417 + d417 + d417 + d418 + d418 + d418 + d418 + d418 + d418 + d418 + d418 + d418 + d419 + d419 + d419;
    const Real d422 = params[0] * t15 * Real(-2.0) * t33 * t56 * dx[4 * d + 3];
    const Real d423 = x[3] * t15 * Real(-2.0) * t33 * t56 * dparams[NP * d + 0];
    const Real d424 = x[3] * params[0] * t15 * t33 * t56 * t56 * d152;
    const Real d425 = x[2] * x[3] * params[0] * t15 * t15 * t56 * Real(-4.0) * d192;
    const Real d426 = x[3] * params[0] * Real(2.0) * t15 * t15 * t33 * d152 * t212;
    const Real d428 = x[2] * x[3] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 2];
    const Real d429 = x[2] * x[2] * params[0] * t15 * t15 * t15 * t56 * Real(-4.0) * dx[4 * d + 3];
    const Real d430 = x[2] * x[2] * x[3] * t15 * t15 * t15 * t56 * Real(-4.0) * dparams[NP * d + 0];
    const Real d431 = x[2] * x[2] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d432 = x[2] * x[2] * x[3] * params[0] * t15 * t15 * t15 * t15 * Real(4.0) * d152 * t212;
    const Real d434 = d422 + d423 + d424 + d425 + d426 + d428 + d428 + d429 + d430 + d431 + d431 + d431 + d432;
    const Real d435 = x[3] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 2];
    const Real d436 = x[2] * x[3] * params[0] * t15 * t15 * t15 * Real(-2.0) * t56 * dx[4 * d + 3];
    const Real d437 = x[2] * x[3] * x[3] * t15 * t15 * t15 * Real(-2.0) * t56 * dparams[NP * d + 0];
    const Real d438 = x[2] * x[3] * x[3] * params[0] * t15 * t15 * t15 * t56 * t56 * d152;
    const Real d439 = x[2] * x[3] * x[3] * params[0] * Real(2.0) * t15 * t15 * t15 * t15 * d152 * t212;
    const Real d441 = d256 + d257 + d258 + d258 + d258 + d435 + d435 + d435 + d436 + d436 + d436 + d436 + d436 + d436 + d437 + d437 + d437 + d438 + d438 + d438 + d438 + d438 + d438 + d438 + d438 + d438 + d439 + d439 + d439;
    const Real d442 = params[0] * t15 * Real(-2.0) * t38 * t56 * dx[4 * d + 3];
    const Real d443 = x[3] * t15 * Real(-2.0) * t38 * t56 * dparams[NP * d + 0];
    const Real d444 = x[3] * params[0] * t15 * t38 * t56 * t56 * d152;
    const Real d445 = x[3] * x[3] * params[0] * t15 * t15 * t56 * Real(-4.0) * d205;
    const Real d446 = x[3] * params[0] * Real(2.0) * t15 * t15 * t38 * d152 * t212;
    const Real d448 = params[0] * t15 * t15 * Real(4.0) * t140 * dx[4 * d + 3];
    const Real d449 = x[3] * t15 * t15 * Real(4.0) * t140 * dparams[NP * d + 0];
    const Real d450 = x[3] * params[0] * t15 * t15 * Real(-2.0) * t56 * t140 * d152;
    const Real d451 = x[3] * t15 * Real(-1.0) * t56 * dx[4 * d + 3];
    const Real d452 = x[3] * x[3] * t15 * t56 * t56 * Real(0.5) * d152;
    const Real d453 = x[3] * x[3] * t15 * t15 * d152 * t212;
    const Real d455 = d155 + d451 + d451 + d452 + d453;
    const Real d456 = x[3] * params[0] * t15 * t15 * Real(4.0) * d455;
    const Real d458 = d442 + d443 + d444 + d445 + d446 + d448 + d449 + d450 + d450 + d456;
    tg[16 * d + 0] = d157;
    tg[16 * d + 1] = d161;
    tg[16 * d + 2] = d165;
    tg[16 * d + 3] = d169;
    tg[16 * d + 4] = d161;
    tg[16 * d + 5] = d177;
    tg[16 * d + 6] = d182;
    tg[16 * d + 7] = d187;
    tg[16 * d + 8] = d165;
    tg[16 * d + 9] = d182;
    tg[16 * d + 10] = d195;
    tg[16 * d + 11] = d200;
    tg[16 * d + 12] = d169;
    tg[16 * d + 13] = d187;
    tg[16 * d + 14] = d200;
    tg[16 * d + 15] = d208;
    tdg[64 * d + 0] = Real(0.0);
    tdg[64 * d + 1] = Real(0.0);
    tdg[64 * d + 2] = Real(0.0);
    tdg[64 * d + 3] = Real(0.0);
    tdg[64 * d + 4] = Real(0.0);
    tdg[64 * d + 5] = Real(0.0);
    tdg[64 * d + 6] = Real(0.0);
    tdg[64 * d + 7] = Real(0.0);
    tdg[64 * d + 8] = Real(0.0);
    tdg[64 * d + 9] = Real(0.0);
    tdg[64 * d + 10] = Real(0.0);
    tdg[64 * d + 11] = Real(0.0);
    tdg[64 * d + 12] = Real(0.0);
    tdg[64 * d + 13] = Real(0.0);
    tdg[64 * d + 14] = Real(0.0);
    tdg[64 * d + 15] = Real(0.0);
    tdg[64 * d + 16] = d215;
    tdg[64 * d + 17] = d224;
    tdg[64 * d + 18] = d231;
    tdg[64 * d + 19] = d238;
    tdg[64 * d + 20] = d224;
    tdg[64 * d + 21] = d255;
    tdg[64 * d + 22] = d266;
    tdg[64 * d + 23] = d277;
    tdg[64 * d + 24] = d231;
    tdg[64 * d + 25] = d266;
    tdg[64 * d + 26] = d290;
    tdg[64 * d + 27] = d298;
    tdg[64 * d + 28] = d238;
    tdg[64 * d + 29] = d277;
    tdg[64 * d + 30] = d298;
    tdg[64 * d + 31] = d311;
    tdg[64 * d + 32] = d316;
    tdg[64 * d + 33] = d231;
    tdg[64 * d + 34] = d322;
    tdg[64 * d + 35] = d329;
    tdg[64 * d + 36] = d231;
    tdg[64 * d + 37] = d342;
    tdg[64 * d + 38] = d353;
    tdg[64 * d + 39] = d298;
    tdg[64 * d + 40] = d322;
    tdg[64 * d + 41] = d353;
    tdg[64 * d + 42] = d370;
    tdg[64 * d + 43] = d377;
    tdg[64 * d + 44] = d329;
    tdg[64 * d + 45] = d298;
    tdg[64 * d + 46] = d377;
    tdg[64 * d + 47] = d390;
    tdg[64 * d + 48] = d395;
    tdg[64 * d + 49] = d238;
    tdg[64 * d + 50] = d329;
    tdg[64 * d + 51] = d401;
    tdg[64 * d + 52] = d238;
    tdg[64 * d + 53] = d414;
    tdg[64 * d + 54] = d298;
    tdg[64 * d + 55] = d421;
    tdg[64 * d + 56] = d329;
    tdg[64 * d + 57] = d298;
    tdg[64 * d + 58] = d434;
    tdg[64 * d + 59] = d441;
    tdg[64 * d + 60] = d401;
    tdg[64 * d + 61] = d421;
    tdg[64 * d + 62] = d441;
    tdg[64 * d + 63] = d458;
  }
}

// The disk functions and their tangents: out = (radius, omega, inner, outer), tout[4 d + j].
template <typename Real, int P>
RM_HD inline void disk_jvp(const Real x[4], const Real* params, const Real* dx, const Real* dparams,
                           Real out[4], Real* tout) {
  (void)x;
  (void)params;
  (void)dx;
  (void)dparams;
  constexpr int NP = 1;
  (void)NP;
  const Real t8 = x[1] * x[1];
  const Real t9 = x[2] * x[2];
  const Real t11 = x[3] * x[3];
  const Real t12 = t8 + t9 + t11;
  const Real t13 = sqrt(t12);
  const Real t47 = sqrt(params[0]);
  const Real t50 = Real(1.0) / (t13 * sqrt(t13));
  const Real t51 = t47 * t50;
  const Real t53 = params[0] * Real(6.0);
  const Real t55 = params[0] * Real(20.0);
  const Real t15 = Real(1.0) / (t13);
  const Real t459 = Real(1.0) / (t47);
  const Real t463 = Real(1.0) / (t13 * t13 * sqrt(t13));
  out[0] = t13;
  out[1] = t51;
  out[2] = t53;
  out[3] = t55;
RM_UNROLL
  for (int d = 0; d < P; ++d) {
    const Real d147 = x[1] * Real(2.0) * dx[4 * d + 1];
    const Real d149 = x[2] * Real(2.0) * dx[4 * d + 2];
    const Real d151 = x[3] * Real(2.0) * dx[4 * d + 3];
    const Real d152 = d147 + d149 + d151;
    const Real d153 = t15 * Real(0.5) * d152;
    const Real d461 = t50 * Real(0.5) * dparams[NP * d + 0] * t459;
    const Real d466 = t15 * t47 * d152 * t463 * Real(-0.75);
    const Real d467 = d461 + d466;
    const Real d468 = Real(6.0) * dparams[NP * d + 0];
    const Real d469 = Real(20.0) * dparams[NP * d + 0];
    tout[4 * d + 0] = d153;
    tout[4 * d + 1] = d467;
    tout[4 * d + 2] = d468;
    tout[4 * d + 3] = d469;
  }
}

// The functions above as one type, so generic code can take the metric as a
// template parameter (rm::trace_ray<rm::kerr::Metric>(...)).
struct Metric {
  static constexpr int PARAM_COUNT = 1;
  static constexpr bool HAS_HORIZON = true;
  static constexpr bool HAS_DISK = true;
  template <typename Real>
  RM_HD static void eval(const Real x[4], const Real* p, Real g[16], Real dg[64]) { metric_eval(x, p, g, dg); }
  template <typename Real>
  RM_HD static Real rhs(const Real y[8], const Real* p, Real dy[8]) { return schwarzschild::geodesic_rhs(y, p, dy); }
  template <typename Real>
  RM_HD static Real horizon_value(const Real x[4], const Real* p) { return schwarzschild::horizon_value(x, p); }
  template <typename Real>
  RM_HD static Real disk_radius(const Real x[4], const Real* p) { return schwarzschild::disk_radius(x, p); }
  template <typename Real>
  RM_HD static Real disk_omega(const Real x[4], const Real* p) { return schwarzschild::disk_omega(x, p); }
  template <typename Real>
  RM_HD static Real disk_inner(const Real x[4], const Real* p) { return schwarzschild::disk_inner(x, p); }
  template <typename Real>
  RM_HD static Real disk_outer(const Real x[4], const Real* p) { return schwarzschild::disk_outer(x, p); }
  template <typename Real>
  RM_HD static Real length_scale(const Real x[4], const Real* p) { return schwarzschild::length_scale(x, p); }
  template <typename Real, int P>
  RM_HD static void jvp(const Real x[4], const Real* p, const Real* dx, const Real* dp, Real g[16], Real dg[64], Real* tg, Real* tdg) {
    schwarzschild::metric_jvp<Real, P>(x, p, dx, dp, g, dg, tg, tdg);
  }
  template <typename Real, int P>
  RM_HD static void disk_jvp(const Real x[4], const Real* p, const Real* dx, const Real* dp, Real out[4], Real* tout) {
    schwarzschild::disk_jvp<Real, P>(x, p, dx, dp, out, tout);
  }
};

}  // namespace schwarzschild
}  // namespace rm
