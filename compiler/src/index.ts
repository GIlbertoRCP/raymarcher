export { compileMetric, type CompiledMetric } from './compile.ts';
export { parseMetric, MetricSyntaxError, type ParamSpec, type ParsedMetric } from './parser.ts';
export { compileToJs, type JsMetric } from './backend-js.ts';
export { Spacetime, invert4, dot4, lower, type RayResult, type Vec3, type Vec4 } from './geodesic.ts';
export { compileToWgsl, MAX_PARAMS } from './backend-wgsl.ts';
