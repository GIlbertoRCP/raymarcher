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
};

}  // namespace schwarzschild
}  // namespace rm
