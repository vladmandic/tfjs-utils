/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { env, util, backend_util, Reshape, _FusedMatMul, Identity, Complex, upcastType, buffer, slice_util, TensorBuffer, Abs, Add, AddN, Transpose, ArgMax, ArgMin, AvgPool, BatchMatMul, Slice, BatchToSpaceND, NotEqual, Real, Cast, zeros, Ceil, ClipByValue, Imag, Concat, Conv2D, Conv2DBackpropInput, CropAndResize, DepthwiseConv2dNative, Multiply, sumOutType, Sum, Einsum, Elu, Equal, Exp, ExpandDims, Expm1, Fill, Floor, FloorDiv, FromPixels, FusedBatchNorm, FusedConv2D, FusedDepthwiseConv2D, GatherNd, GatherV2, Greater, GreaterEqual, Less, LessEqual, Log, LogicalAnd, Max, Maximum, MaxPool, Mean, Min, Minimum, MirrorPad, Neg, NonMaxSuppressionV3, kernel_impls, NonMaxSuppressionV5, ZerosLike, OnesLike, Pack, PadV2, Pow, Prelu, Prod, Range, RealDiv, Relu, Relu6, ResizeBilinear, ResizeNearestNeighbor, Rsqrt, Select, Sigmoid, Sub, Softmax, SpaceToBatchND, Sqrt, Square, SquaredDifference, StridedSlice, StringNGrams, Tanh, Tile, Transform, Unpack, registerKernel, KernelBackend, DataStorage, engine, device_util, registerBackend } from '@tensorflow/tfjs-core';

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const ENV = env();
/** The batched command encoders size in the device queue. */
ENV.registerFlag('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', () => 15);
/**
 * Whether we forward execution to the CPU backend if tensors are small and
 * reside on the CPU.
 */
ENV.registerFlag('WEBGPU_CPU_FORWARD', () => true);
/**
 * Thread register block size for matmul kernel.
 */
ENV.registerFlag('WEBGPU_MATMUL_WORK_PER_THREAD', () => 4);
/**
 * Whether to use conv2d_naive which directly implement the conv2d logic rather
 * than using a matmul to simulate.
 */
ENV.registerFlag('WEBGPU_USE_NAIVE_CONV2D', () => false);
/**
 * Whether to use GLSL shading language.
 */
ENV.registerFlag('WEBGPU_USE_GLSL', () => true);
/**
 * Whether to use conv2dTranspose_naive which directly implement the
 * conv2dTranspose logic rather than using a matmul to simulate.
 */
ENV.registerFlag('WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE', () => false);
/**
 * Whether we will run im2col as a separate shader for convolution.
 */
ENV.registerFlag('WEBGPU_CONV_SEPARATE_IM2COL_SHADER', () => false);
/**
 * Whether we use low power GPU. Otherwise, a high performance GPU will be
 * requested.
 */
ENV.registerFlag('WEBGPU_USE_LOW_POWER_GPU', () => false);
/**
 * Threshold for input tensor size that determines whether WebGPU backend will
 * delegate computation to CPU.
 *
 * Default value is 128.
 */
ENV.registerFlag('CPU_HANDOFF_SIZE_THRESHOLD', () => 128);
/**
 * Whether to use a dummy canvas to make profiling tools like PIX work with
 * TFJS webgpu backend.
 */
ENV.registerFlag('WEBGPU_USE_PROFILE_TOOL', () => false);
/**
 * Whether to use import API.
 */
ENV.registerFlag('WEBGPU_USE_IMPORT', () => true);

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var BinaryOpType;
(function (BinaryOpType) {
    BinaryOpType[BinaryOpType["MUL"] = 0] = "MUL";
    BinaryOpType[BinaryOpType["ADD"] = 1] = "ADD";
    BinaryOpType[BinaryOpType["SUB"] = 2] = "SUB";
    BinaryOpType[BinaryOpType["DIV"] = 3] = "DIV";
    BinaryOpType[BinaryOpType["EQUAL"] = 4] = "EQUAL";
    BinaryOpType[BinaryOpType["GREATER"] = 5] = "GREATER";
    BinaryOpType[BinaryOpType["GREATER_EQUAL"] = 6] = "GREATER_EQUAL";
    BinaryOpType[BinaryOpType["LESS"] = 7] = "LESS";
    BinaryOpType[BinaryOpType["LESS_EQUAL"] = 8] = "LESS_EQUAL";
    BinaryOpType[BinaryOpType["LOGICAL_AND"] = 9] = "LOGICAL_AND";
    BinaryOpType[BinaryOpType["NOT_EQUAL"] = 10] = "NOT_EQUAL";
    BinaryOpType[BinaryOpType["SQUARED_DIFFERENCE"] = 11] = "SQUARED_DIFFERENCE";
    BinaryOpType[BinaryOpType["INT_DIV"] = 12] = "INT_DIV";
    BinaryOpType[BinaryOpType["POW"] = 13] = "POW";
    BinaryOpType[BinaryOpType["PRELU"] = 14] = "PRELU";
    BinaryOpType[BinaryOpType["MAX"] = 15] = "MAX";
    BinaryOpType[BinaryOpType["MIN"] = 16] = "MIN";
    BinaryOpType[BinaryOpType["COMPLEX_MULTIPLY_REAL"] = 17] = "COMPLEX_MULTIPLY_REAL";
    BinaryOpType[BinaryOpType["COMPLEX_MULTIPLY_IMAG"] = 18] = "COMPLEX_MULTIPLY_IMAG";
})(BinaryOpType || (BinaryOpType = {}));
// GLSL shader.
const CHECK_NAN_SNIPPET = `
  if (isnan(a)) return a;
  if (isnan(b)) return b;
  `;
const CHECK_NAN_SNIPPET_VEC4 = `
  result.r = isNaN.r > 0. ? NAN : result.r;
  result.g = isNaN.g > 0. ? NAN : result.g;
  result.b = isNaN.b > 0. ? NAN : result.b;
  result.a = isNaN.a > 0. ? NAN : result.a;
  `;
const ADD = 'return a + b;';
// (Ar + Ai)(Br + Bi) =
// ArBr + ArBi + AiBr + AiBi = ArBr - AB + ArBi + AiBr
// Yr = ArBr - AB
// Yi = ArBi + AiBr
const COMPLEX_MULTIPLY_REAL = 'return areal * breal - aimag * bimag;';
const COMPLEX_MULTIPLY_IMAG = 'return areal * bimag + aimag * breal;';
const DIV = 'return a / b;';
const EQUAL = 'return float(a == b);';
const EQUAL_VEC4 = 'return vec4(equal(a, b));';
const GREATER = 'return float(a > b);';
const GREATER_VEC4 = 'return vec4(greaterThan(a, b));';
const GREATER_EQUAL = 'return float(a >= b);';
const GREATER_EQUAL_VEC4 = 'return vec4(greaterThanEqual(a, b));';
const INT_DIV = `
  float s = sign(a) * sign(b);
  int ia = int(round(a));
  int ib = int(round(b));
  return float(idiv(ia, ib, s));
  `;
const INT_DIV_VEC4 = `
  ivec4 ia = ivec4(round(a));
  ivec4 ib = ivec4(round(b));
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
  `;
const LESS = 'return float(a < b);';
const LESS_VEC4 = 'return vec4(lessThan(a, b));';
const LESS_EQUAL = 'return float(a <= b);';
const LESS_EQUAL_VEC4 = 'return vec4(lessThanEqual(a, b));';
const LOGICAL_AND = 'return float(float(a) >= 1.0 && float(b) >= 1.0);';
const LOGICAL_AND_VEC4 = `return vec4(
  vec4(greaterThanEqual(a, vec4(1.0))) *
  vec4(greaterThanEqual(b, vec4(1.0))));`;
const MUL = 'return a * b;';
const NOT_EQUAL = 'return float(a != b);';
const NOT_EQUAL_VEC4 = 'return vec4(notEqual(a, b));';
const POW = `
  if(a < 0.0 && floor(b) < b) {
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
  `;
const POW_VEC4 = `
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  vec4 isNaN = vec4(lessThan(a, vec4(0.0))) * vec4(lessThan(floor(b), b));
  ${CHECK_NAN_SNIPPET_VEC4}
  return result;
  `;
const PRELU = 'return (a < 0.) ? b * a : a;';
const PRELU_VEC4 = `
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
  `;
const SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';
const SUB = 'return a - b;';
// WGSL shader.
const EQUAL_WGSL = 'return f32(a == b);';
const EQUAL_VEC4_WGSL = 'return vec4<f32>(a == b);';
const GREATER_WGSL = 'return f32(a > b);';
const GREATER_VEC4_WGSL = 'return vec4<f32>(a > b);';
const GREATER_EQUAL_WGSL = 'return f32(a >= b);';
const GREATER_EQUAL_VEC4_WGSL = 'return vec4<f32>(a >= b);';
const LESS_WGSL = 'return f32(a < b);';
const LESS_VEC4_WGSL = 'return vec4<f32>(a < b);';
const LESS_EQUAL_WGSL = 'return f32(a <= b);';
const LESS_EQUAL_VEC4_WGSL = 'return vec4<f32>(a <= b);';
const LOGICAL_AND_WGSL = 'return f32(f32(a) >= 1.0 && f32(b) >= 1.0);';
const LOGICAL_AND_VEC4_WGSL = `return (vec4<f32>(a >= vec4<f32>(1.0)) *
  vec4<f32>(b >= vec4<f32>(1.0)));`;
const CHECK_NAN_SNIPPET_WGSL = `
  if (isNanCustom(a)) { return a; }
  if (isNanCustom(b)) { return b; }
  `;
const CHECK_NAN_SNIPPET_VEC4_WGSL = `
  if (isNaN.r > 0.) {
    resultTemp.r = uniforms.NAN;
  }
  if (isNaN.g > 0.) {
    resultTemp.g = uniforms.NAN;
  }
  if (isNaN.b > 0.) {
    resultTemp.b = uniforms.NAN;
  }
  if (isNaN.a > 0.) {
    resultTemp.a = uniforms.NAN;
  }
  `;
const INT_DIV_WGSL = `
  let s = sign(a) * sign(b);
  let ia = i32(round(a));
  let ib = i32(round(b));
  return f32(idiv(ia, ib, s));
  `;
const INT_DIV_VEC4_WGSL = `
  let ia = vec4<i32>(round(a));
  let ib = vec4<i32>(round(b));
  let cond = ib != vec4<i32>(0);
  var resultTemp = vec4<i32>(0);
  let s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    resultTemp[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    resultTemp[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    resultTemp[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    resultTemp[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4<f32>(resultTemp);
  `;
const NOT_EQUAL_WGSL = 'return f32(a != b);';
const NOT_EQUAL_VEC4_WGSL = 'return vec4<f32>(a != b);';
const POW_WGSL = `
  if(a < 0.0 && floor(b) < b) {
    return f32(uniforms.NAN);
  }
  if (b == 0.0) {
    return 1.0;
  }
  if (i32(round(b % 2.0)) != 1) {
    return pow(abs(a), b);
  }
  return sign(a) * pow(abs(a), b);
  `;
const POW_VEC4_WGSL = `
  let isModRound1Bool = vec4<i32>(round(b % vec4<f32>(2.0))) == vec4<i32>(1);
  let isModRound1 = vec4<f32>(isModRound1Bool);
  let multiplier = sign(a) * isModRound1 + (vec4<f32>(1.0) - isModRound1);
  var resultTemp = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  let isExpZero = b == vec4<f32>(0.0);
  if (isExpZero.r) {
    resultTemp.r = 1.0;
  }
  if (isExpZero.g) {
    resultTemp.g = 1.0;
  }
  if (isExpZero.b) {
    resultTemp.b = 1.0;
  }
  if (isExpZero.a) {
    resultTemp.a = 1.0;
  }
  let isNaN = vec4<f32>(a < vec4<f32>(0.0)) * vec4<f32>(floor(b) < b);
  ${CHECK_NAN_SNIPPET_VEC4_WGSL}
  return resultTemp;
  `;
const PRELU_WGSL = `if (a < 0.0) { return b * a; }  return a;`;
const PRELU_VEC4_WGSL = `
  let aLessThanZero : vec4<bool> = vec4<bool>(a < vec4<f32>(0.0));
  let aLessThanZeroF32 = vec4<f32>(aLessThanZero);
  return (vec4<f32>(aLessThanZeroF32) * (b * a)) + ((vec4<f32>(1.0) - vec4<f32>(aLessThanZeroF32)) * a);
  `;
function getMinMaxString(op, useVec4, useWGSL = false) {
    if (useWGSL) {
        const checkNanSnippetWgsl = useVec4 ? CHECK_NAN_SNIPPET_VEC4_WGSL : CHECK_NAN_SNIPPET_WGSL;
        return useVec4 ? `
    var resultTemp = vec4<f32>(${op}(a, b));
    let isNaN = min(vec4<f32>(isNanCustomVec4F32(a)) + vec4<f32>(isNanCustomVec4F32(b)), vec4<f32>(1.0));
    ` + checkNanSnippetWgsl +
            `
    return resultTemp;
  ` :
            checkNanSnippetWgsl + `
    return ${op}(a, b);
  `;
    }
    const checkNanSnippet = useVec4 ? CHECK_NAN_SNIPPET_VEC4 : CHECK_NAN_SNIPPET;
    return useVec4 ? `
    vec4 result = vec4(${op}(a, b));
    vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
    ` + checkNanSnippet +
        `
    return result;
  ` :
        checkNanSnippet + `
    return ${op}(a, b);
  `;
}
function getBinaryOpString(type, useVec4, useWgsl) {
    switch (type) {
        case BinaryOpType.MUL:
            return MUL;
        case BinaryOpType.ADD:
            return ADD;
        case BinaryOpType.SUB:
            return SUB;
        case BinaryOpType.DIV:
            return DIV;
        case BinaryOpType.EQUAL:
            if (useWgsl) {
                return useVec4 ? EQUAL_VEC4_WGSL : EQUAL_WGSL;
            }
            else {
                return useVec4 ? EQUAL_VEC4 : EQUAL;
            }
        case BinaryOpType.GREATER:
            if (useWgsl) {
                return useVec4 ? GREATER_VEC4_WGSL : GREATER_WGSL;
            }
            else {
                return useVec4 ? GREATER_VEC4 : GREATER;
            }
        case BinaryOpType.GREATER_EQUAL:
            if (useWgsl) {
                return useVec4 ? GREATER_EQUAL_VEC4_WGSL : GREATER_EQUAL_WGSL;
            }
            else {
                return useVec4 ? GREATER_EQUAL_VEC4 : GREATER_EQUAL;
            }
        case BinaryOpType.LESS:
            if (useWgsl) {
                return useVec4 ? LESS_VEC4_WGSL : LESS_WGSL;
            }
            else {
                return useVec4 ? LESS_VEC4 : LESS;
            }
        case BinaryOpType.LESS_EQUAL:
            if (useWgsl) {
                return useVec4 ? LESS_EQUAL_VEC4_WGSL : LESS_EQUAL_WGSL;
            }
            else {
                return useVec4 ? LESS_EQUAL_VEC4 : LESS_EQUAL;
            }
        case BinaryOpType.LOGICAL_AND:
            if (useWgsl) {
                return useVec4 ? LOGICAL_AND_VEC4_WGSL : LOGICAL_AND_WGSL;
            }
            else {
                return useVec4 ? LOGICAL_AND_VEC4 : LOGICAL_AND;
            }
        case BinaryOpType.NOT_EQUAL:
            if (useWgsl) {
                return useVec4 ? NOT_EQUAL_VEC4_WGSL : NOT_EQUAL_WGSL;
            }
            else {
                return useVec4 ? NOT_EQUAL_VEC4 : NOT_EQUAL;
            }
        case BinaryOpType.SQUARED_DIFFERENCE:
            return SQUARED_DIFFERENCE;
        case BinaryOpType.INT_DIV:
            if (useWgsl) {
                return useVec4 ? INT_DIV_VEC4_WGSL : INT_DIV_WGSL;
            }
            else {
                return useVec4 ? INT_DIV_VEC4 : INT_DIV;
            }
        case BinaryOpType.PRELU:
            if (useWgsl) {
                return useVec4 ? PRELU_VEC4_WGSL : PRELU_WGSL;
            }
            else {
                return useVec4 ? PRELU_VEC4 : PRELU;
            }
        case BinaryOpType.MAX:
            return getMinMaxString('max', useVec4, useWgsl);
        case BinaryOpType.MIN:
            return getMinMaxString('min', useVec4, useWgsl);
        case BinaryOpType.POW:
            if (useWgsl) {
                return useVec4 ? POW_VEC4_WGSL : POW_WGSL;
            }
            else {
                return useVec4 ? POW_VEC4 : POW;
            }
        case BinaryOpType.COMPLEX_MULTIPLY_REAL:
            return COMPLEX_MULTIPLY_REAL;
        case BinaryOpType.COMPLEX_MULTIPLY_IMAG:
            return COMPLEX_MULTIPLY_IMAG;
        default:
            throw new Error(`BinaryType ${type} is not implemented!`);
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var UnaryOpType;
(function (UnaryOpType) {
    UnaryOpType[UnaryOpType["ABS"] = 0] = "ABS";
    UnaryOpType[UnaryOpType["CEIL"] = 1] = "CEIL";
    UnaryOpType[UnaryOpType["ELU"] = 2] = "ELU";
    UnaryOpType[UnaryOpType["EXP"] = 3] = "EXP";
    UnaryOpType[UnaryOpType["EXPM1"] = 4] = "EXPM1";
    UnaryOpType[UnaryOpType["FLOOR"] = 5] = "FLOOR";
    UnaryOpType[UnaryOpType["LINEAR"] = 6] = "LINEAR";
    UnaryOpType[UnaryOpType["LOG"] = 7] = "LOG";
    UnaryOpType[UnaryOpType["NEG"] = 8] = "NEG";
    UnaryOpType[UnaryOpType["PRELU"] = 9] = "PRELU";
    UnaryOpType[UnaryOpType["RELU"] = 10] = "RELU";
    UnaryOpType[UnaryOpType["RELU6"] = 11] = "RELU6";
    UnaryOpType[UnaryOpType["RSQRT"] = 12] = "RSQRT";
    UnaryOpType[UnaryOpType["SIGMOID"] = 13] = "SIGMOID";
    UnaryOpType[UnaryOpType["SQRT"] = 14] = "SQRT";
    UnaryOpType[UnaryOpType["SQUARE"] = 15] = "SQUARE";
    UnaryOpType[UnaryOpType["TANH"] = 16] = "TANH";
    UnaryOpType[UnaryOpType["TO_INT"] = 17] = "TO_INT";
})(UnaryOpType || (UnaryOpType = {}));
// GLSL shader.
const ABS = `return abs(a);`;
const CEIL = `return ceil(a);`;
const EXPM1 = `return exp(a) - 1.0;`;
const ELU = `return (a >= 0.0) ? a : (exp(a) - 1.0);`;
const ELU_VEC4 = `
  vec4 result;

  result.r = (a.r >= 0.0) ? a.r : (exp(a.r) - 1.0);
  result.g = (a.g >= 0.0) ? a.g : (exp(a.g) - 1.0);
  result.b = (a.b >= 0.0) ? a.b : (exp(a.b) - 1.0);
  result.a = (a.a >= 0.0) ? a.a : (exp(a.a) - 1.0);

  return result;
`;
const EXP = `return exp(a);`;
const FLOOR = `return floor(a);`;
const LINEAR = `return a;`;
const LOG = `if (a < 0.0) { return 1.0/0.0; }
  return log(a);`;
const NEG = `return -a;`;
const PRELU$1 = `return (a < 0.0) ? b * a : a;`;
const RELU = 'return max(a, 0.0);';
const RELU6 = 'return clamp(a, 0.0, 6.0);';
const RELU_VEC4 = `
  vec4 result = a * vec4(greaterThanEqual(a, vec4(0.0)));
  bvec4 isNaN = isnan(a);

  result.r = isNaN.r ? a.r : result.r;
  result.g = isNaN.g ? a.g : result.g;
  result.b = isNaN.b ? a.b : result.b;
  result.a = isNaN.a ? a.a : result.a;

  return result;
`;
const RSQRT = `return 1.0/sqrt(a);`;
const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * a));`;
const SQRT = `return sqrt(a);`;
const SQUARE = `return a * a;`;
const TANH = `
  float e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`;
const TO_INT = `return float(int(a));`;
// WGSL shader.
const ELU_WGSL = `if (a >= 0.0) { return a; }  return (exp(a) - 1.0);`;
const RELU_WGSL = 'return max(a, 0.0);';
const RELU6_VEC4_WGSL = 'return clamp(a, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(6.0, 6.0, 6.0, 6.0));';
const RELU_VEC4_WGSL = `
  var resBool : vec4<bool> = vec4<bool>(a >= vec4<f32>(0.0, 0.0, 0.0, 0.0));
  let isNaN : vec4<bool> = isNan(a);
  var resFloat : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  for (var i:u32 = 0u; i< 4u; i = i+1u ) {
    if (resBool[i]) {
      resFloat[i] = 1.0;
    }
  }
  resFloat = a * resFloat;
  if (isNaN.r) {
    resFloat.r = a.r;
  }
  if (isNaN.g) {
    resFloat.g = a.g;
  }
  if (isNaN.b) {
    resFloat.b = a.b;
  }
  if (isNaN.a) {
    resFloat.a = a.a;
  }
  return resFloat;
`;
const TO_INT_WGSL = `return f32(i32((a)));`;
function getUnaryOpString(type, useVec4, useWgsl) {
    switch (type) {
        case UnaryOpType.ABS:
            return ABS;
        case UnaryOpType.CEIL:
            return CEIL;
        case UnaryOpType.ELU:
            if (useWgsl) {
                if (useVec4) {
                    throw new Error(`UnaryOpType ELU vec4 for WGSL is not implemented!`);
                }
                return ELU_WGSL;
            }
            else {
                return useVec4 ? ELU_VEC4 : ELU;
            }
        case UnaryOpType.EXP:
            return EXP;
        case UnaryOpType.EXPM1:
            return EXPM1;
        case UnaryOpType.FLOOR:
            return FLOOR;
        case UnaryOpType.LINEAR:
            return LINEAR;
        case UnaryOpType.LOG:
            return LOG;
        case UnaryOpType.NEG:
            return NEG;
        case UnaryOpType.PRELU:
            return PRELU$1;
        case UnaryOpType.RELU:
            if (useWgsl) {
                return useVec4 ? RELU_VEC4_WGSL : RELU_WGSL;
            }
            else {
                return useVec4 ? RELU_VEC4 : RELU;
            }
        case UnaryOpType.RELU6:
            if (useWgsl) {
                return useVec4 ? RELU6_VEC4_WGSL : RELU6;
            }
            else {
                return RELU6;
            }
        case UnaryOpType.RSQRT:
            return RSQRT;
        case UnaryOpType.SIGMOID:
            return SIGMOID;
        case UnaryOpType.SQRT:
            return SQRT;
        case UnaryOpType.SQUARE:
            return SQUARE;
        case UnaryOpType.TANH:
            return TANH;
        case UnaryOpType.TO_INT:
            return useWgsl ? TO_INT_WGSL : TO_INT;
        default:
            throw new Error(`BinaryType ${type} is not implemented!`);
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function mapActivationToShaderProgram(activation, packed = false, useWgsl = false) {
    if (activation === null) {
        return null;
    }
    else if (activation === 'linear') {
        return getUnaryOpString(UnaryOpType.LINEAR);
    }
    else if (activation === 'relu') {
        return getUnaryOpString(UnaryOpType.RELU, packed, useWgsl);
    }
    else if (activation === 'elu') {
        return getUnaryOpString(UnaryOpType.ELU, packed);
    }
    else if (activation === 'relu6') {
        return getUnaryOpString(UnaryOpType.RELU6, packed, useWgsl);
    }
    else if (activation === 'prelu') {
        return getBinaryOpString(BinaryOpType.PRELU, packed, useWgsl);
    }
    else if (activation === 'sigmoid') {
        return getUnaryOpString(UnaryOpType.SIGMOID);
    }
    throw new Error(`Activation ${activation} has not been implemented for the WebGPU backend.`);
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function makeMatMulSmallOutputSizeSource() {
    return `
  float mm_readA(int row, int col);
  float mm_readB(int row, int col);
  void mm_write(int row, int col, float value);
  const int TileAOuter = int(gl_WorkGroupSize.y / 2);
  const int TileBOuter = int(gl_WorkGroupSize.x);
  const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

  shared float mm_Asub1[TileAOuter][TileInner];
  shared float mm_Bsub1[TileInner][TileBOuter];
  shared float mm_Asub2[TileAOuter][TileInner];
  shared float mm_Bsub2[TileInner][TileBOuter];

  // If the output size is small for matrix multiplication, avoid to use vec4
  // and handle some elements per thread to optimally utilize the ALU.
  // Introduces two shared memory buffers, some logical threads could handle
  // arithmetic operations and others handle IO operations between barrier api,
  // makes ALUs and load/store units work simultaneously, could improves
  // the performance.
  void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
    int tileRow = int(gl_LocalInvocationID.y);
    int tileCol = int(gl_LocalInvocationID.x);
    int globalRow = int(gl_GlobalInvocationID.y);
    int globalCol = int(gl_GlobalInvocationID.x);

    int numTiles = (dimInner - 1) / TileInner + 1;
    float acc = 0.0;

    int globalColA = tileCol;
    int globalRowB = tileRow;
    int tileColA = int(gl_LocalInvocationID.x);
    int tileRowB = int(gl_LocalInvocationID.y);
    for (int t = 0; t < numTiles; t++) {
      if (t == 0) {
        if (tileRow < TileAOuter) {
          // Load one tile of A and B into local memory.
          mm_Asub1[tileRow][tileColA] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA);
          globalColA += TileInner;
          mm_Bsub1[tileRowB][tileCol] = mm_readB(globalRowB, globalCol);
          globalRowB += TileInner;
        }
      } else {
        if (tileRow < TileAOuter) {
          // Load one tile of A and B into local memory.
          mm_Asub1[tileRow][tileColA] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA);
          globalColA += TileInner;
          mm_Bsub1[tileRowB][tileCol] = mm_readB(globalRowB, globalCol);
          globalRowB += TileInner;
        } else {
          // Compute acc values for a single thread.
          for (int k = 0; k < TileInner; k++) {
            acc += mm_Asub2[tileRow - TileAOuter][k] * mm_Bsub2[k][tileCol];
          }
        }
      }
      barrier();
      if (t != 0) {
        t++;
      }

      if (t < numTiles) {
        if (tileRow < TileAOuter) {
          // Load one tile of A and B into local memory.
          mm_Asub2[tileRow][tileColA] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA);
          globalColA += TileInner;
          mm_Bsub2[tileRowB][tileCol] = mm_readB(globalRowB, globalCol);
          globalRowB += TileInner;
        } else {
          // Compute acc values for a single thread.
          for (int k = 0; k < TileInner; k++) {
            acc += mm_Asub1[tileRow - TileAOuter][k] * mm_Bsub1[k][tileCol];
          }
        }
      }
      barrier();
    }
    if (tileRow >= TileAOuter) {
      mm_write((globalRow - tileRow) / 2 + tileRow - TileAOuter,
          globalCol, acc);
    }
  }
  `;
}
class MatMulSmallOutputSizeProgram {
    constructor(aShape, bShape, outputShape, bias = null, activation = null, preluActivationWeights = null) {
        this.variableNames = ['A', 'B'];
        this.workGroupSize = [8, 16, 1];
        util.assert(aShape[1] <= 16 || bShape[2] <= 16, () => 'This program can be only used when A width is small.');
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [2], y: [1], z: [0] };
        this.dispatch = [Math.ceil(outputShape[2] / this.workGroupSize[0]),
            Math.ceil(outputShape[1] * 2 / this.workGroupSize[1]), outputShape[0]];
        const addBias = bias != null;
        if (addBias) {
            this.variableNames.push('bias');
        }
        const hasPreluActivationWeights = preluActivationWeights != null;
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        this.shaderKey = `matMulSmallOutputSize_${this.activation}`;
    }
    getUserCode() {
        let sampleA;
        sampleA = `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner + col] : 0`;
        let sampleB;
        sampleB = `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter + col] : 0`;
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation);
            if (this.hasPreluActivationWeights) {
                activationSnippet = `float activation(float a, ivec3 outCoord) {
            float b = getPreluActivationWeightsAtOutCoords(outCoord);
            ${activationOp}
            }`;
            }
            else {
                activationSnippet = `float activation(float a, ivec3 outCoord) {
            ${activationOp}
        }`;
            }
            applyActivationSnippet = 'value = activation(value, outCoord);';
        }
        const addBiasSnippet = this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';
        const userCode = `
      ${activationSnippet}

      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];
      int batch;
      ${makeMatMulSmallOutputSizeSource()}
      float mm_readA(int row, int col) {
        int batchASize = aShape[1] * aShape[2];
        return ${sampleA};
      }
      float mm_readB(int row, int col) {
        int batchBSize = bShape[1] * bShape[2];
        return ${sampleB};
      }
      void mm_write(int row, int col, float value) {
        if (coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimBOuter))) {
          ivec3 outCoord = ivec3(batch, row, col);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(batch, row, col, value);
        }
      }
      void main() {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Generates GLSL that computes strides.
function symbolicallyComputeStrides(indicesArr, variableName) {
    if (Math.max(...indicesArr) > 3) {
        throw new Error('Cannot symbolically compute strides for rank > 4 tensor.');
    }
    const numCoords = indicesArr.length;
    const shape = indicesArr.map(d => `${variableName}[${d}]`);
    const strides = new Array(numCoords - 1);
    strides[numCoords - 2] = shape[numCoords - 1];
    for (let i = numCoords - 3; i >= 0; --i) {
        strides[i] = `(${strides[i + 1]} * ${shape[i + 1]})`;
    }
    return strides;
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function getCoordsDataTypeWgsl(rank) {
    if (rank <= 1) {
        return 'u32';
    }
    else if (rank === 2) {
        return 'vec2<u32>';
    }
    else if (rank === 3) {
        return 'vec3<u32>';
    }
    else if (rank === 4) {
        return 'vec4<u32>';
    }
    else {
        throw Error(`GPU for rank ${rank} is not yet supported`);
    }
}
function mapToTypesWgsl(type, isVec4) {
    if (type === 'float32') {
        return isVec4 ? 'vec4<f32>' : 'f32';
    }
    else if (type === 'int32') {
        return isVec4 ? 'vec4<i32>' : 'i32';
    }
    else if (type === 'bool') {
        // Type 'bool' cannot be used in storage class,
        // https://www.w3.org/TR/WGSL/#host-shareable-types.
        return isVec4 ? 'vec4<i32>' : 'i32';
    }
    return type;
}
function getWorkGroupSizeStringWgsl(workGroupSize) {
    return `
  [[stage(compute), workgroup_size(${workGroupSize[0]}, ${workGroupSize[1]}, ${workGroupSize[2]})]]
`;
}
function getGlobalIndexStringWgsl(workGroupSize) {
    return `
  let index = getGlobalIndex(globalId, localId, vec3<u32>(${workGroupSize[0]}u, ${workGroupSize[1]}u, ${workGroupSize[2]}u));
`;
}
function getMainHeaderStringWgsl(workGroupSize) {
    return `
  [[stage(compute), workgroup_size(${workGroupSize[0]}, ${workGroupSize[1]}, ${workGroupSize[2]})]]
  fn main([[builtin(local_invocation_id)]] localId : vec3<u32>, [[builtin(global_invocation_id)]] globalId : vec3<u32>)
`;
}
function makeShader(inputInfo, outputData, program, isFromPixel = false) {
    if (isFromPixel === true) {
        const getCoords = generateGetCoordsFromFlatIndex(outputData.shape);
        const outputBufferStr = `
      [[block]] struct Matrix0 {
        numbers: array<${mapToTypesWgsl(outputData.dtype, program.isVec4)}>;
      };
      [[block]] struct Uniform {
        size            : i32;
        numChannels     : i32;
        outShapeStrides : vec2<u32>;
      };

      [[group(0), binding(0)]] var<storage, write> result : Matrix0;
      [[group(0), binding(2)]] var<uniform> uniforms: Uniform;
    `;
        return [
            SHADER_PREFIX,
            outputBufferStr,
            getCoords,
            program.getUserCodeWgsl(),
        ].join('\n');
    }
    const prefixSnippets = [];
    let uniformDeclaration = '[[block]] struct Uniforms { NAN : f32; ';
    program.variableNames.forEach((x, i) => {
        uniformDeclaration += `${x.charAt(0).toLowerCase() + x.slice(1)}Shape : ${getCoordsDataTypeWgsl(inputInfo[i].shape.length)}; `;
    });
    uniformDeclaration +=
        `outShape : ${getCoordsDataTypeWgsl(outputData.shape.length)} ; `;
    const stridesLength = outputData.shape.length - 1;
    uniformDeclaration += `
       outShapeStrides: ${getCoordsDataTypeWgsl(stridesLength)}; `;
    if (program.size != null) {
        uniformDeclaration += 'size : u32; ';
    }
    uniformDeclaration += 'dispatchSize : vec3<u32>; ';
    if (program.uniformsWgsl) {
        uniformDeclaration += program.uniformsWgsl;
    }
    uniformDeclaration += '};';
    prefixSnippets.push(uniformDeclaration);
    // Output buffer.
    prefixSnippets.push(`
    [[block]] struct Matrix0 {
        numbers: array<${mapToTypesWgsl(outputData.dtype, program.isVec4)}>;
    };

    [[group(0), binding(0)]] var<storage, write> result : Matrix0;
  `);
    program.variableNames.forEach((x, i) => {
        prefixSnippets.push(`
    [[block]] struct Matrix${1 + i} {
      numbers: array<${mapToTypesWgsl(inputInfo[i].dtype, program.isVec4)}>;
    };
    [[group(0), binding(${1 + i})]] var<storage, read> ${x} : Matrix${1 + i};
    `);
    });
    if (uniformDeclaration !== '') {
        prefixSnippets.push(`
    [[group(0), binding(${1 + program.variableNames.length})]] var<uniform> uniforms : Uniforms;
    `);
    }
    const [getOutputCoords, dispatchLayoutRank] = generateGetOutputCoords(outputData.shape, program.dispatchLayout);
    const getCoords = generateGetCoordsFromFlatIndex(outputData.shape);
    const sources = [
        SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS, getCoords,
        getOutputCoords,
        getSetOutputSnippet(outputData.shape, outputData.dtype, program.isVec4)
    ];
    if (dispatchLayoutRank === outputData.shape.length) {
        // Input sampling snippet is only meaningful when the output isn't getting
        // implicitly reshaped (like it does in conv2d_matmul).
        const inputSamplingSnippet = inputInfo
            .map(x => getInputSamplingSnippet(x, outputData.shape, program.isVec4, program.dispatchLayout.x.length ===
            outputData.shape.length))
            .join('\n');
        sources.push(inputSamplingSnippet);
    }
    sources.push(program.getUserCodeWgsl());
    const source = sources.join('\n');
    return source;
}
const SHADER_PREFIX = `
  fn idiv(a: i32, b: i32, sign: f32) -> i32 {
    var res: i32 = a / b;
    let mod: i32 = a % b;
    if (sign < 0. && mod != 0) {
      res = res - 1;
    }
    return res;
  }

  fn isNanCustom(val : f32) -> bool {
    if (val > 0.0) {
      return false;
    }
    if (val < 0.0) {
      return false;
    }
    if (val == 0.0) {
      return false;
    }
    return true;
  }

  fn isNanCustomVec4F32(val : vec4<f32>) -> vec4<f32> {
    var res = vec4<f32> (0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
      if (isNanCustom(val[i])) {
        res[i] = 1.0;
      } else {
        res[i] = 0.0;
      }
    }
    return res;
  }

  // Checks whether coordinates lie within the bounds of the shape.
  fn coordsInBounds4D(coord : vec4<u32>, shape : vec4<u32>) -> bool {
    return all(coord >= vec4<u32>(0u)) &&
        all(coord < shape);
  }

  fn coordsInBounds3D(coord : vec3<u32>, shape : vec3<u32>) -> bool {
    return all(coord >= vec3<u32>(0u)) &&
        all(coord < shape);
  }

  fn coordsInBounds2D(coord : vec2<u32>, shape : vec2<u32>) -> bool {
    return all(coord >= vec2<u32>(0u)) &&
        all(coord < shape);
  }
  `;
const SAMPLING_SNIPPETS = `
  fn getFlatIndex1D(coord : u32, shape : u32) -> u32 {
    return coord;
  }

  fn getFlatIndex2D(coords : vec2<u32>, shape : vec2<u32>) -> u32 {
    return u32(dot(vec2<f32>(coords), vec2<f32>(f32(shape.y), 1.0)));
  }

  fn getFlatIndex3D(coords : vec3<u32>, shape : vec3<u32>) -> u32 {
    return u32(dot(vec3<f32>(coords), vec3<f32>(f32(shape.y) * f32(shape.z), f32(shape.z), 1.0)));
  }

  fn getFlatIndex4D(coords : vec4<u32>, shape : vec4<u32>) -> u32 {
    return u32(dot(vec4<f32>(coords), vec4<f32>(
        f32(shape.y) * f32(shape.z) * f32(shape.w), f32(shape.z) * f32(shape.w), f32(shape.w), 1.0)));
  }

  // Only used when the y/z dimension of workgroup size is 1.
  fn getGlobalIndex(globalId : vec3<u32>, localId : vec3<u32>, workGroupSize : vec3<u32>) -> u32 {
    if (uniforms.dispatchSize.y == 1u && uniforms.dispatchSize.z == 1u) {
      return globalId.x;
    }
    let localInvocationIndex = localId.z * workGroupSize.x * workGroupSize.y +
      localId.y * workGroupSize.x + localId.x;
    let workGroupID = (globalId - localId)/vec3<u32>(
      workGroupSize[0], workGroupSize[1], workGroupSize[2]);
    return (workGroupID.z * uniforms.dispatchSize.x * uniforms.dispatchSize.y +
      workGroupID.y * uniforms.dispatchSize.x + workGroupID.x) *
      (workGroupSize.x * workGroupSize.y * workGroupSize.z) +
      localInvocationIndex;
  }
`;
function getSetOutputSnippet(outShape, outBufferType, isVec4) {
    const outRank = outShape.length;
    const wgslType = mapToTypesWgsl(outBufferType, isVec4);
    let snippet;
    if (isVec4) {
        snippet = `fn setOutputFlat(flatIndex : u32, value : vec4<f32>) {
      result.numbers[flatIndex] = ${wgslType}(value);
    }
    fn setOutputFlatI32(flatIndex : u32, value : vec4<i32>) {
      result.numbers[flatIndex] = ${wgslType}(value);
    }`;
    }
    else {
        snippet = `fn setOutputFlat(flatIndex : u32, value : f32) {
      result.numbers[flatIndex] = ${wgslType}(value);
    }
    fn setOutputFlatI32(flatIndex : u32, value : i32) {
      result.numbers[flatIndex] = ${wgslType}(value);
    }`;
    }
    if (outRank >= 2) {
        switch (outRank) {
            case 2:
                snippet += `
        fn getOutputFlatIndex(coords : vec2<u32>) -> u32 {
          return u32(dot(vec2<f32>(coords), vec2<f32>(f32(uniforms.outShapeStrides), 1.0)));
        }
        `;
                break;
            case 3:
                snippet += `
        fn getOutputFlatIndex(coords : vec3<u32>) -> u32 {
          return u32(dot(vec3<f32>(coords), vec3<f32>(f32(uniforms.outShapeStrides.x), f32(uniforms.outShapeStrides.y), 1.0)));
        }
        `;
                break;
            case 4:
                snippet += `
        fn getOutputFlatIndex(coords : vec4<u32>) -> u32 {
          return u32(dot(vec4<f32>(coords), vec4<f32>(
            f32(uniforms.outShapeStrides.x), f32(uniforms.outShapeStrides.y), f32(uniforms.outShapeStrides.z), 1.0)));
        }
        `;
                break;
            default:
                util.assert(false, () => `Unsupported ${outRank}D shape`);
                break;
        }
        const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
        const type = getCoordsDataTypeWgsl(outRank);
        if (isVec4) {
            snippet += `
      fn setOutput(${dims.map(d => `${d} : u32`).join(', ')}, value : vec4<f32>) {
        let flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutputFlat(flatIndex / 4u, value);
      }
      fn setOutputVectorI32(${dims.map(d => `${d} : u32`).join(', ')}, value : vec4<i32>) {
        let flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutputFlatI32(flatIndex / 4u, value);
      }
    `;
        }
        else {
            snippet += `
      fn setOutput(${dims.map(d => `${d} : u32`).join(', ')}, value : f32) {
        let flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutputFlat(flatIndex, value);
      }
      fn setOutputI32(${dims.map(d => `${d} : u32`).join(', ')}, value : i32) {
        let flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutputFlatI32(flatIndex, value);
      }
    `;
        }
    }
    return snippet;
}
function getInputSamplingSnippet(inInfo, outShape, isVec4, isFlatDispatchLayout) {
    let res = getSamplerFromInInfo(inInfo, isVec4);
    const inShape = inInfo.shape;
    if (inShape.length <= outShape.length) {
        res += getSamplerAtOutputCoords(inInfo, outShape, isVec4, isFlatDispatchLayout);
    }
    return res;
}
function getSamplerFromInInfo(inInfo, isVec4) {
    const texName = inInfo.name;
    const rank = inInfo.shape.length;
    const type = getCoordsDataTypeWgsl(rank);
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, rank);
    const inputs = dims.map(d => `${d} : u32`).join(', ');
    if (rank < 1) {
        if (isVec4) {
            return `
        fn ${funcName}() -> vec4<f32> {
          return vec4<f32>(${texName}.numbers[0]);
        }
      `;
        }
        return `
      fn ${funcName}() ->f32 {
        return f32(${texName}.numbers[0]);
      }
    `;
    }
    const shapeStr = `uniforms.${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;
    let rankStr = `${rank}D`;
    if (rank === 0) {
        rankStr = '1D';
    }
    if (isVec4) {
        return `
      fn ${funcName}(${inputs}) -> vec4<f32> {
        return vec4<f32>(${texName}.numbers[getFlatIndex${rankStr}(${type}(${dims.join(',')}),
          ${shapeStr}) / 4u]);
      }
      `;
    }
    return `
    fn ${funcName}(${inputs}) -> f32 {
      return f32(${texName}.numbers[getFlatIndex${rankStr}(${type}(${dims.join(',')}),
        ${shapeStr})]);
    }
   `;
}
// TODO: Implement getXXXFromFlatIndex, use it instead of getXXXAtOutCoords when
// it's flat dispatch layout.
function getSamplerAtOutputCoords(inInfo, outShape, isVec4, isFlatDispatchLayout) {
    const texName = inInfo.name;
    const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
    const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
    const inRank = inInfo.shape.length;
    const outRank = outShape.length;
    const type = getCoordsDataTypeWgsl(outRank);
    // If the inShape equals the outShape and the dispatch layout is flat, we can
    // directly use |gl_GlobalInvocationID.x| as the index and don't need coords
    // conversion between these two shapes.
    if (util.arraysEqual(inInfo.shape, outShape) && isFlatDispatchLayout) {
        if (isVec4) {
            return `
        fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : u32) -> vec4<f32> {
          return vec4<f32>(${texName}.numbers[globalIndex]);
        }

        fn ${funcName}ByCoords(coords : ${type}) -> vec4<f32> {
          return vec4<f32>(${texName}.numbers[${outRank > 1 ? 'getOutputFlatIndex(coords)' : 'coords'} / 4u]);
        }
        `;
        }
        else {
            return `
      fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : u32) -> f32 {
        return f32(${texName}.numbers[globalIndex]);
      }

      fn ${funcName}ByCoords(coords : ${type}) -> f32 {
        return f32(${texName}.numbers[${outRank > 1 ? 'getOutputFlatIndex(coords)' : 'coords'}]);
      }
      `;
        }
    }
    const broadcastDims = backend_util.getBroadcastDims(inInfo.shape, outShape);
    const rankDiff = outRank - inRank;
    let coordsSnippet = '';
    if (inRank === 0) {
        if (isVec4) {
            return `
      fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : u32) -> vec4<f32> {
        return get${texFuncSnippet}();
      }

      fn ${funcName}ByCoords(coords : ${type}) -> vec4<f32> {
        return get${texFuncSnippet}();
      }
    `;
        }
        return `
      fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : u32) -> f32{
        return get${texFuncSnippet}();
      }

      fn ${funcName}ByCoords(coords : ${type}) -> f32{
        return get${texFuncSnippet}();
      }
    `;
    }
    else {
        if (outRank < 2 && broadcastDims.length >= 1) {
            coordsSnippet = 'coords = 0u;';
        }
        else {
            coordsSnippet =
                broadcastDims.map(d => `coords[${d + rankDiff}u] = 0u;`).join('\n');
        }
    }
    let unpackedCoordsSnippet = '';
    if (outRank < 2 && inRank > 0) {
        unpackedCoordsSnippet = 'coords';
    }
    else {
        if (outRank > 1) {
            const coordsType = getCoordsDataTypeWgsl(inRank);
            const coordsValues = inInfo.shape.map((s, i) => `coords[${i + rankDiff}u]`).join(', ');
            unpackedCoordsSnippet = `${coordsType}(${coordsValues})`;
        }
        else {
            unpackedCoordsSnippet = 'coords';
        }
    }
    const shapeStr = `uniforms.${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;
    const rankStr = `${inRank}D`;
    if (isVec4) {
        return `
      fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : u32) -> vec4<f32> {
        var coords = getOutputCoords(globalId, globalIndex);
        ${coordsSnippet}
        return ${texName}.numbers[getFlatIndex${rankStr}(${unpackedCoordsSnippet}, ${shapeStr}) / 4u];
      }

      fn ${funcName}ByCoords(coordsIn : ${type}) -> vec4<f32> {
        var coords = coordsIn;
        ${coordsSnippet}
        return ${texName}.numbers[getFlatIndex${rankStr}(${unpackedCoordsSnippet}, ${shapeStr}) / 4u];
      }
    `;
    }
    return `
    fn ${funcName}ByGlobalId(globalId : vec3<u32>, globalIndex : u32) -> f32 {
      var coords = getOutputCoords(globalId, globalIndex);
      ${coordsSnippet}
      return f32(${texName}.numbers[getFlatIndex${rankStr}(${unpackedCoordsSnippet}, ${shapeStr})]);
    }

    fn ${funcName}ByCoords(coordsIn : ${type}) -> f32 {
      var coords = coordsIn;
      ${coordsSnippet}
      return f32(${texName}.numbers[getFlatIndex${rankStr}(${unpackedCoordsSnippet}, ${shapeStr})]);
    }
  `;
}
/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
function generateGetOutputCoords(outShape, dispatchLayout) {
    const { x, y = [], z = [] } = dispatchLayout;
    const outRank = outShape.length;
    if (x.length === outRank) {
        const dtype = getCoordsDataTypeWgsl(outRank);
        const snippet = `fn getOutputCoords(globalId : vec3<u32>, globalIndex : u32) -> ${dtype}{
      return getCoordsFromFlatIndex(u32(globalIndex));
    }
    `;
        return [snippet, outRank];
    }
    let gatherDimensionsStr = '';
    const dims = [x, y, z];
    let rank = 0;
    for (let i = 0; i < dims.length; i++) {
        const arr = dims[i];
        if (arr.length === 0) {
            continue;
        }
        rank += arr.length;
        if (arr.length === 1) {
            gatherDimensionsStr += `let d${arr[0]} = globalId[${i}];`;
        }
        else {
            const strides = symbolicallyComputeStrides(arr, 'uniforms.outShape');
            gatherDimensionsStr += `var index${i} = globalId[${i}];`;
            for (let j = 0; j < strides.length; j++) {
                gatherDimensionsStr += `let d${arr[j]} = index${i} / ${strides[j]};`;
                if (j === strides.length - 1) {
                    gatherDimensionsStr += `let d${arr[j + 1]} = ` +
                        `index${i} - d${arr[j]} * ${strides[j]};`;
                }
                else {
                    gatherDimensionsStr +=
                        `index${i} = index${i} - d${arr[j]} * ${strides[j]};`;
                }
            }
        }
    }
    const dimensions = [];
    for (let i = 0; i < rank; i++) {
        dimensions.push(`d${i}`);
    }
    const dtype = getCoordsDataTypeWgsl(rank);
    let snippet = `fn getOutputCoords(globalId : vec3<u32>, globalIndex : u32) -> ${dtype} {
    ${gatherDimensionsStr}
  `;
    if (dimensions.length === 0) {
        snippet += `return ${dtype}(0); }`;
    }
    else {
        snippet += `return ${dtype}(${dimensions.join(',')}); }`;
    }
    return [snippet, rank];
}
/**
 * Derives logical coordinates from a flat index. Performs integer division
 * with each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
function generateGetCoordsFromFlatIndex(shape) {
    const rank = shape.length;
    if (rank <= 1) {
        return `fn getCoordsFromFlatIndex(index : u32) -> u32 { return index; }`;
    }
    const strides = util.computeStrides(shape);
    const dtype = getCoordsDataTypeWgsl(rank);
    const coords = [];
    for (let i = 0; i < rank; i++) {
        coords.push(`d${i}`);
    }
    if (strides.length === 1) {
        return `    fn getCoordsFromFlatIndex(index : u32) -> vec2<u32> {
      let d0 = index / uniforms.outShapeStrides; let d1 = index - d0 * uniforms.outShapeStrides;
      return vec2<u32>(d0, d1);
    }`;
    }
    const snippet = 'var index2 = index;' +
        strides
            .map((_, i) => {
            const line1 = `let ${coords[i]} = index2 / uniforms.outShapeStrides[${i}]`;
            const line2 = i === strides.length - 1 ?
                `let ${coords[i + 1]} = index2 - ${coords[i]} * uniforms.outShapeStrides[${i}]` :
                `index2 = index2 - ${coords[i]} * uniforms.outShapeStrides[${i}]`;
            return `${line1}; ${line2};`;
        })
            .join('');
    return `
    fn getCoordsFromFlatIndex(index : u32) -> ${dtype} {
      ${snippet}
      return ${dtype}(${coords.join(',')});
    }
  `;
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE = 65535;

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const arrayProduct = (arr) => {
    let product = 1;
    for (let i = 0; i < arr.length; i++) {
        product *= arr[i];
    }
    return product;
};
function tilesFitEvenlyIntoShape(tileSize, shape) {
    if (tileSize.length !== shape.length) {
        throw new Error(`Cannot compute whether rank ${tileSize.length}` +
            ` tiles fit evenly into rank ${shape.length} shape` +
            ` - ranks must match.`);
    }
    return shape.every((dim, dimIdx) => dim % tileSize[dimIdx] === 0);
}
// Computes dispatch geometry based on layout of output dimensions and
// workGroupSize.
function computeDispatch(layout, outputShape, workGroupSize = [1, 1, 1], elementsPerThread = [1, 1, 1]) {
    const [dispatchX, dispatchY, dispatchZ] = [
        Math.ceil(arrayProduct(layout.x.map(d => outputShape[d])) /
            (workGroupSize[0] * elementsPerThread[0])),
        layout.y ? Math.ceil(arrayProduct(layout.y.map(d => outputShape[d])) /
            (workGroupSize[1] * elementsPerThread[1])) :
            1,
        layout.z ? Math.ceil(arrayProduct(layout.z.map(d => outputShape[d])) /
            (workGroupSize[2] * elementsPerThread[2])) :
            1
    ];
    if (dispatchX <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE &&
        dispatchY <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE &&
        dispatchZ <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE) {
        return [dispatchX, dispatchY, dispatchZ];
    }
    util.assert(dispatchX > MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE &&
        layout.y === undefined && layout.z === undefined, () => 'Dispatch size exceeds WebGPU limits in Y or Z dimension.');
    let dispatchAverage = Math.ceil(Math.sqrt(dispatchX));
    if (dispatchAverage > MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE) {
        dispatchAverage = Math.ceil(Math.cbrt(dispatchX));
        util.assert(dispatchAverage <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE, () => 'Total dispatch size exceeds WebGPU maximum.');
        return [dispatchAverage, dispatchAverage, dispatchAverage];
    }
    else {
        return [dispatchAverage, dispatchAverage, 1];
    }
}
function computeWorkGroupSizeForConv2d(layout, outputShape) {
    const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
    const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
    // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
    // These are experimental values. Usually, we need to adjust the work group
    // size based on the output shape. For example, when one dimension is smaller
    // than 4, it will be wasteful if we assign a larger size for this dimension,
    // which results lots of threads doing useless work and reduces parallelism
    // of hardware threads. But it is always a balance between work group size
    // and shared memory. If one dimension is too small, such as 1, shared memory
    // will won't be fully utilized.
    if (dim0 <= 4) {
        return [4, 16, 1];
    }
    if (dim1 <= 4) {
        return [16, 4, 1];
    }
    return [16, 16, 1];
}
function computeWorkGroupSizeForMatMul(dimAOuter, dimInner, dimBOuter) {
    // These are experimental values. Usually, we need to adjust the work group
    // size based on the input shapes to improve the EU occupancy.
    // TODO: WebGPU limits the maximum allowed shared memory size as 16K. To make
    // sure it doesn't exceed this limitations. Temporarily reduce the work group
    // size to [8, 8, 1] and the work per thread size is [4, 4, 1]. But we should
    // revisit it and find the balance between work group size and work per thread
    // size.
    if (dimAOuter === 1) {
        return [32, 1, 1];
    }
    else if (dimBOuter === 1) {
        return [1, 32, 1];
    }
    return [8, 8, 1];
}
function computeWorkPerThreadForConv2d(layout, outputShape) {
    const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
    const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
    // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
    // The following conditions correspond to the values set in
    // computeWorkGroupSizeForConv2d.
    if (dim0 <= 4) {
        return [1, 2, 1];
    }
    if (dim1 <= 4) {
        return [2, 1, 1];
    }
    return [2, 2, 1];
}
function flatDispatchLayout(shape) {
    return { x: shape.map((d, i) => i) };
}
function GPUBytesPerElement(dtype) {
    if (dtype === 'float32' || dtype === 'int32' || dtype === 'bool' ||
        dtype === 'string') {
        return 4;
    }
    else if (dtype === 'complex64') {
        return 8;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
function ArrayBufferToTypedArray(data, dtype) {
    if (dtype === 'float32') {
        return new Float32Array(data);
    }
    else if (dtype === 'int32') {
        return new Int32Array(data);
    }
    else if (dtype === 'bool' || dtype === 'string') {
        const dataAsInt32Array = new Int32Array(data);
        const boolData = new ArrayBuffer(dataAsInt32Array.length);
        const dataAsTypedArray = new Uint8Array(boolData);
        for (let i = 0; i < dataAsInt32Array.length; i++) {
            dataAsTypedArray[i] = dataAsInt32Array[i];
        }
        return dataAsTypedArray;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
function isWebGPUSupported() {
    if (!navigator.gpu) {
        return false;
    }
    return true;
}

var webgpu_util = /*#__PURE__*/Object.freeze({
  __proto__: null,
  tilesFitEvenlyIntoShape: tilesFitEvenlyIntoShape,
  computeDispatch: computeDispatch,
  computeWorkGroupSizeForConv2d: computeWorkGroupSizeForConv2d,
  computeWorkGroupSizeForMatMul: computeWorkGroupSizeForMatMul,
  computeWorkPerThreadForConv2d: computeWorkPerThreadForConv2d,
  flatDispatchLayout: flatDispatchLayout,
  GPUBytesPerElement: GPUBytesPerElement,
  ArrayBufferToTypedArray: ArrayBufferToTypedArray,
  isWebGPUSupported: isWebGPUSupported
});

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function getGlslDifferences() {
    const defineSpecialNaN = `
      bool isnan_custom(float val) {
        // logical or has undefined behavior, https://bugs.chromium.org/p/tint/issues/detail?id=976.
        if (val > 0.0) {
          return false;
        }
        if (val < 0.0) {
          return false;
        }
        if (val == 0.0) {
          return false;
        }
        return true;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `;
    return { defineSpecialNaN };
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function getCoordsDataType(rank) {
    if (rank <= 1) {
        return 'int';
    }
    else if (rank === 2) {
        return 'ivec2';
    }
    else if (rank === 3) {
        return 'ivec3';
    }
    else if (rank === 4) {
        return 'ivec4';
    }
    else {
        throw Error(`GPU for rank ${rank} is not yet supported`);
    }
}
function mapToGlslTypes(type, isVec4) {
    if (type === 'float32') {
        return isVec4 ? 'vec4' : 'float';
    }
    else if (type === 'int32') {
        return isVec4 ? 'ivec4' : 'int';
    }
    else if (type === 'bool') {
        return isVec4 ? 'bvec4' : 'bool';
    }
    return type;
}
function makeShader$1(inputInfo, outputData, program, isFromPixel = false) {
    const outputBufferStr = `    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      ${mapToGlslTypes(outputData.dtype, program.isVec4)} result[];
    };`;
    if (isFromPixel === true) {
        const getCoords = generateGetCoordsFromFlatIndex$1(outputData.shape);
        return [
            SHADER_PREFIX$1, outputBufferStr, program.getUserCode(), getCoords
        ].join('\n');
    }
    const prefixSnippets = [];
    if (program.workGroupSize != null) {
        prefixSnippets.push(`
      layout (local_size_x = ${program.workGroupSize[0]},
              local_size_y = ${program.workGroupSize[1]},
              local_size_z = ${program.workGroupSize[2]}) in;
    `);
    }
    // Output buffer.
    prefixSnippets.push(`
    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      ${mapToGlslTypes(outputData.dtype, program.isVec4)} result[];
    };
  `);
    program.variableNames.forEach((x, i) => {
        prefixSnippets.push(`
      layout(std430, set = 0, binding = ${1 + i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputInfo[i].dtype, program.isVec4)} ${x}[];
      };
    `);
    });
    let uniformDeclaration = 'float NAN; ';
    program.variableNames.forEach((x, i) => {
        uniformDeclaration += `${getCoordsDataType(inputInfo[i].shape.length)} ${x.charAt(0).toLowerCase() + x.slice(1)}Shape; `;
    });
    uniformDeclaration +=
        `${getCoordsDataType(outputData.shape.length)} outShape; `;
    const stridesLength = outputData.shape.length - 1;
    uniformDeclaration += `${getCoordsDataType(stridesLength)} outShapeStrides; `;
    if (program.size != null) {
        uniformDeclaration += 'int size; ';
    }
    uniformDeclaration += 'ivec3 dispatchSize; ';
    if (program.uniforms) {
        uniformDeclaration += program.uniforms;
    }
    if (uniformDeclaration !== '') {
        prefixSnippets.push(`
        layout(std140, set = 0, binding = ${1 + program.variableNames.length}) uniform Uniforms {
            ${uniformDeclaration}
        };
    `);
    }
    prefixSnippets.push(getGlslDifferences().defineSpecialNaN);
    const [getOutputCoords, dispatchLayoutRank] = generateGetOutputCoords$1(outputData.shape, program.dispatchLayout);
    const getCoords = generateGetCoordsFromFlatIndex$1(outputData.shape);
    const sources = [
        SHADER_PREFIX$1, prefixSnippets.join('\n'), SAMPLING_SNIPPETS$1, getCoords,
        getOutputCoords,
        getSetOutputSnippet$1(outputData.shape, outputData.dtype, program.isVec4)
    ];
    if (dispatchLayoutRank === outputData.shape.length) {
        // Input sampling snippet is only meaningful when the output isn't getting
        // implicitly reshaped (like it does in conv2d_matmul).
        const inputSamplingSnippet = inputInfo
            .map(x => getInputSamplingSnippet$1(x, outputData.shape, program.isVec4, program.dispatchLayout.x.length ===
            outputData.shape.length))
            .join('\n');
        sources.push(inputSamplingSnippet);
    }
    sources.push(program.getUserCode());
    const source = sources.join('\n');
    return source;
}
const SHADER_PREFIX$1 = `#version 450

  int idiv(int a, int b, float sign) {
    int res = a / b;
    int mod = a % b;
    if (sign < 0. && mod != 0) {
      res -= 1;
    }
    return res;
  }

  // Checks whether coordinates lie within the bounds of the shape.
  bool coordsInBounds(ivec4 coord, ivec4 shape) {
    return all(greaterThanEqual(coord, ivec4(0))) &&
        all(lessThan(coord, shape));
  }

  bool coordsInBounds(ivec3 coord, ivec3 shape) {
    return all(greaterThanEqual(coord, ivec3(0))) &&
        all(lessThan(coord, shape));
  }

  bool coordsInBounds(ivec2 coord, ivec2 shape) {
    return all(greaterThanEqual(coord, ivec2(0))) &&
        all(lessThan(coord, shape));
  }
`;
const SAMPLING_SNIPPETS$1 = `
  int getFlatIndex(int coord, int shape) {
    return coord;
  }

  int getFlatIndex(ivec2 coords, ivec2 shape) {
    return int(dot(coords, ivec2(shape.y, 1.)));
  }

  int getFlatIndex(ivec3 coords, ivec3 shape) {
    return int(dot(coords, ivec3(shape.y * shape.z, shape.z, 1.)));
  }

  int getFlatIndex(ivec4 coords, ivec4 shape) {
    return int(dot(coords, ivec4(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
  }

  // Only used when the y/z dimension of workgroup size is 1.
  int getGlobalIndex() {
    if (dispatchSize.y == 1 && dispatchSize.z == 1) {
      return int(gl_GlobalInvocationID.x);
    } else {
      return int((gl_WorkGroupID.z * dispatchSize.x * dispatchSize.y +
        gl_WorkGroupID.y * dispatchSize.x + gl_WorkGroupID.x) *
        (gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z) +
        gl_LocalInvocationIndex);
    }
  }
`;
function getSetOutputSnippet$1(outShape, outBufferType, isVec4) {
    const outRank = outShape.length;
    const glslType = mapToGlslTypes(outBufferType, isVec4);
    let snippet;
    if (isVec4) {
        snippet = `void setOutput(int flatIndex, vec4 value) {
      result[flatIndex] = ${glslType === 'ivec4' ?
            'ivec4(value)' :
            (glslType === 'bvec4' ? 'bvec4(value)' : 'value')};
    }
    void setOutput(int flatIndex, ivec4 value) {
      result[flatIndex] = ${glslType === 'vec4' ?
            'vec4(value)' :
            (glslType === 'bvec4' ? 'bvec4(value)' : 'value')};
    }`;
    }
    else {
        snippet = `void setOutput(int flatIndex, float value) {
      result[flatIndex] = ${glslType === 'int' ? 'int(value)' :
            (glslType === 'bool' ? 'bool(value)' : 'value')};
    }
    void setOutput(int flatIndex, int value) {
      result[flatIndex] = ${glslType === 'float' ?
            'float(value)' :
            (glslType === 'bool' ? 'bool(value)' : 'value')};
    }`;
    }
    if (outRank >= 2) {
        switch (outRank) {
            case 2:
                snippet += `
        int getOutputFlatIndex(ivec2 coords) {
          return int(dot(coords, ivec2(outShapeStrides, 1)));
        }
        `;
                break;
            case 3:
                snippet += `
        int getOutputFlatIndex(ivec3 coords) {
          return int(dot(coords, ivec3(outShapeStrides.x, outShapeStrides.y, 1)));
        }
        `;
                break;
            case 4:
                snippet += `
        int getOutputFlatIndex(ivec4 coords) {
          return int(dot(coords, ivec4(
            outShapeStrides.x, outShapeStrides.y, outShapeStrides.z, 1)));
        }
        `;
                break;
            default:
                util.assert(false, () => `Unsupported ${outRank}D shape`);
                break;
        }
        const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
        const type = getCoordsDataType(outRank);
        if (isVec4) {
            snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, vec4 value) {
        int flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutput(flatIndex / 4, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, ivec4 value) {
        int flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutput(flatIndex / 4, value);
      }
    `;
        }
        else {
            snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, float value) {
        int flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutput(flatIndex, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, int value) {
        int flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutput(flatIndex, value);
      }
    `;
        }
    }
    return snippet;
}
function getInputSamplingSnippet$1(inInfo, outShape, isVec4, isFlatDispatchLayout) {
    let res = getSamplerFromInInfo$1(inInfo, isVec4);
    const inShape = inInfo.shape;
    if (inShape.length <= outShape.length) {
        res += getSamplerAtOutputCoords$1(inInfo, outShape, isVec4, isFlatDispatchLayout);
    }
    return res;
}
function getSamplerFromInInfo$1(inInfo, isVec4) {
    const texName = inInfo.name;
    const rank = inInfo.shape.length;
    const type = getCoordsDataType(rank);
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, rank);
    const inputs = dims.map(d => `int ${d}`).join(', ');
    if (rank < 1) {
        if (isVec4) {
            return `
        vec4 ${funcName}() {
          return vec4(${texName}[0]);
        }
      `;
        }
        return `
      float ${funcName}() {
        return float(${texName}[0]);
      }
    `;
    }
    const shapeStr = `${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;
    if (isVec4) {
        return `
      vec4 ${funcName}(${inputs}) {
        return vec4(${texName}[getFlatIndex(${type}(${dims.join(',')}),
          ${shapeStr}) / 4]);
      }
      `;
    }
    return `
    float ${funcName}(${inputs}) {
      return float(${texName}[getFlatIndex(${type}(${dims.join(',')}),
        ${shapeStr})]);
    }
   `;
}
function getSamplerAtOutputCoords$1(inInfo, outShape, isVec4, isFlatDispatchLayout) {
    const texName = inInfo.name;
    const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
    const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
    const inRank = inInfo.shape.length;
    const outRank = outShape.length;
    const type = getCoordsDataType(outRank);
    // If the inShape equals the outShape and the dispatch layout is flat, we can
    // directly use |gl_GlobalInvocationID.x| as the index and don't need coords
    // conversion between these two shapes.
    if (util.arraysEqual(inInfo.shape, outShape) && isFlatDispatchLayout) {
        if (isVec4) {
            return `
        vec4 ${funcName}() {
          return vec4(${texName}[getGlobalIndex()]);
        }

        vec4 ${funcName}(${type} coords) {
          return vec4(${texName}[${outRank > 1 ? 'getOutputFlatIndex(coords)' : 'coords'} / 4]);
        }
        `;
        }
        else {
            return `
      float ${funcName}() {
        return float(${texName}[getGlobalIndex()]);
      }

      float ${funcName}(${type} coords) {
        return float(${texName}[${outRank > 1 ? 'getOutputFlatIndex(coords)' : 'coords'}]);
      }
      `;
        }
    }
    const broadcastDims = backend_util.getBroadcastDims(inInfo.shape, outShape);
    const rankDiff = outRank - inRank;
    let coordsSnippet = '';
    if (inRank === 0) {
        if (isVec4) {
            return `
      vec4 ${funcName}() {
        return get${texFuncSnippet}();
      }

      vec4 ${funcName}(${type} coords) {
        return get${texFuncSnippet}();
      }
    `;
        }
        return `
      float ${funcName}() {
        return get${texFuncSnippet}();
      }

      float ${funcName}(${type} coords) {
        return get${texFuncSnippet}();
      }
    `;
    }
    else {
        if (outRank < 2 && broadcastDims.length >= 1) {
            coordsSnippet = 'coords = 0;';
        }
        else {
            coordsSnippet =
                broadcastDims.map(d => `coords[${d + rankDiff}] = 0;`).join('\n');
        }
    }
    let unpackedCoordsSnippet = '';
    if (outRank < 2 && inRank > 0) {
        unpackedCoordsSnippet = 'coords';
    }
    else {
        if (outRank > 1) {
            const coordsType = getCoordsDataType(inRank);
            const coordsValues = inInfo.shape.map((s, i) => `coords[${i + rankDiff}]`).join(', ');
            unpackedCoordsSnippet = `${coordsType}(${coordsValues})`;
        }
        else {
            unpackedCoordsSnippet = 'coords';
        }
    }
    const shapeStr = `${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;
    if (isVec4) {
        return `
      vec4 ${funcName}() {
        ${type} coords = getOutputCoords();
        ${coordsSnippet}
        return ${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${shapeStr}) / 4];
      }

      vec4 ${funcName}(${type} coords) {
        ${coordsSnippet}
        return ${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${shapeStr}) / 4];
      }
    `;
    }
    return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${shapeStr})]);
    }

    float ${funcName}(${type} coords) {
      ${coordsSnippet}
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${shapeStr})]);
    }
  `;
}
/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
function generateGetOutputCoords$1(outShape, dispatchLayout) {
    const { x, y = [], z = [] } = dispatchLayout;
    const outRank = outShape.length;
    if (x.length === outRank) {
        const dtype = getCoordsDataType(outRank);
        const snippet = `${dtype} getOutputCoords() {
      return getCoordsFromFlatIndex(getGlobalIndex());
    }
    `;
        return [snippet, outRank];
    }
    let gatherDimensionsStr = '';
    const dims = [x, y, z];
    let rank = 0;
    for (let i = 0; i < dims.length; i++) {
        const arr = dims[i];
        if (arr.length === 0) {
            continue;
        }
        rank += arr.length;
        if (arr.length === 1) {
            gatherDimensionsStr += `int d${arr[0]} =
        int(gl_GlobalInvocationID[${i}]);`;
        }
        else {
            const strides = symbolicallyComputeStrides(arr, 'outShape');
            gatherDimensionsStr += `int index${i} =
          int(gl_GlobalInvocationID[${i}]);`;
            for (let j = 0; j < strides.length; j++) {
                gatherDimensionsStr += `int d${arr[j]} = index${i} / ${strides[j]};`;
                if (j === strides.length - 1) {
                    gatherDimensionsStr += `int d${arr[j + 1]} = ` +
                        `index${i} - d${arr[j]} * ${strides[j]};`;
                }
                else {
                    gatherDimensionsStr += `index${i} -= d${arr[j]} * ${strides[j]};`;
                }
            }
        }
    }
    const dimensions = [];
    for (let i = 0; i < rank; i++) {
        dimensions.push(`d${i}`);
    }
    const dtype = getCoordsDataType(rank);
    let snippet = `${dtype} getOutputCoords() {
    ${gatherDimensionsStr}
  `;
    if (dimensions.length === 0) {
        snippet += `return ${dtype}(0);}`;
    }
    else {
        snippet += `return ${dtype}(${dimensions.join(',')});}`;
    }
    return [snippet, rank];
}
/**
 * Derives logical coordinates from a flat index. Performs integer division
 * with each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
function generateGetCoordsFromFlatIndex$1(shape) {
    const rank = shape.length;
    if (rank <= 1) {
        return `int getCoordsFromFlatIndex(int index) {return index; }`;
    }
    const strides = util.computeStrides(shape);
    const dtype = getCoordsDataType(rank);
    const coords = [];
    for (let i = 0; i < rank; i++) {
        coords.push(`d${i}`);
    }
    if (strides.length === 1) {
        return `    ivec2 getCoordsFromFlatIndex(int index) {
      int d0 = index / outShapeStrides; int d1 = index - d0 * outShapeStrides;
      return ivec2(d0,d1);
    }`;
    }
    const snippet = strides
        .map((_, i) => {
        const line1 = `int ${coords[i]} = index / outShapeStrides[${i}]`;
        const line2 = i === strides.length - 1 ?
            `int ${coords[i + 1]} = index - ${coords[i]} * outShapeStrides[${i}]` :
            `index -= ${coords[i]} * outShapeStrides[${i}]`;
        return `${line1}; ${line2};`;
    })
        .join('');
    return `
    ${dtype} getCoordsFromFlatIndex(int index) {
      ${snippet}
      return ${dtype}(${coords.join(',')});
    }
  `;
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const makeBindGroup = (device, bindGroupLayout, inputs, output, uniforms) => {
    const bindings = [output, ...inputs];
    if (uniforms) {
        bindings.push(uniforms);
    }
    return device.createBindGroup({
        layout: bindGroupLayout,
        entries: bindings.map((b, i) => ({ binding: i, resource: b })),
    });
};
const compileProgram = (glslang, device, program, pipelineLayout, inputsData, output, isFromPixel = false) => {
    const outputData = { dtype: output.dtype, shape: output.shape };
    let source;
    let module;
    if (program.useWgsl) {
        source = makeShader(inputsData, outputData, program, isFromPixel);
        module = device.createShaderModule({ code: source });
    }
    else {
        source = makeShader$1(inputsData, outputData, program, isFromPixel);
        const result = glslang.compileGLSLZeroCopy(source, 'compute', false);
        if (result.data.length === 0) {
            throw new Error('Shader compilation failed');
        }
        result.free();
        module = device.createShaderModule({ code: result.data });
    }
    const pipeline = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: 'main' } });
    return pipeline;
};
function makeShaderKey(program, shapes, types, broadcastDimsKey = '', inputShapesEqualsOutShape = '') {
    let useWgslKey = '';
    if (program.useWgsl) {
        useWgslKey = '_1';
    }
    const key = (program.workGroupSize ? program.workGroupSize.join(',') : '') +
        shapes.map(shape => shape.length).join(',') + types.join(',') +
        program.variableNames.join(',') + broadcastDimsKey +
        inputShapesEqualsOutShape + program.shaderKey + useWgslKey;
    return key;
}
// This is global flag, but program may ignore this flag.
function getUseWgsl() {
    return !env().getBool('WEBGPU_USE_GLSL');
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function makeMatMulPackedVec4Source(workPerThread) {
    return `
    vec4 mm_readA(int row, int col);
    vec4 mm_readB(int row, int col);
    void mm_write(int row, int col, vec4 value);

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${workPerThread[0]}; // only support ColPerThread = 4
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileBOuter;

    shared vec4 mm_Asub[TileAOuter][TileInner / ColPerThread];
    shared vec4 mm_Bsub[TileInner][TileBOuter / ColPerThread];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x);

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x);

      int numTiles = (dimInner - 1) / TileInner + 1;

      vec4 acc[RowPerThread];
      vec4 ACached;
      vec4 BCached[4];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          acc[innerRow] = vec4(0.0);
      }

      // Loop over shared dimension.
      int globalColA = tileCol;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                globalColA);
        }
        globalColA += TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol);
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner / ColPerThread; k++) {
          BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
          BCached[1] = mm_Bsub[k * ColPerThread + 1][tileCol];
          BCached[2] = mm_Bsub[k * ColPerThread + 2][tileCol];
          BCached[3] = mm_Bsub[k * ColPerThread + 3][tileCol];

          for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[tileRow + i][k];
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
            acc[i] = BCached[3] * ACached.w + acc[i];
          }
        }
        barrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        mm_write(globalRow + innerRow,
          globalCol,
          acc[innerRow]);
      }
    }
  `;
}
function makeMatMulVectorVec4Source() {
    return `
    vec4 mm_readA(int row, int col);
    vec4 mm_readB(int row, int col);
    void mm_write(int row, int col, vec4 value);

    const int TileSize = int(gl_WorkGroupSize.x) * 4;

    shared vec4 mm_Asub[TileSize / 4];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileCol = int(gl_LocalInvocationID.x);
      int globalCol = int(gl_GlobalInvocationID.x);
      int globalRow = int(gl_GlobalInvocationID.y);

      int numTiles = (dimInner - 1) / TileSize + 1;

      // Without this initialization strange values show up in acc.
      vec4 acc = vec4(0.0);

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        int colA = t * TileSize / 4 + tileCol;
        mm_Asub[tileCol] = mm_readA(globalRow, colA);
        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileSize / 4; k++) {
          int rowB = t * TileSize + k * 4;
          vec4 BCached0 = mm_readB(rowB, globalCol);
          vec4 BCached1 = mm_readB(rowB + 1, globalCol);
          vec4 BCached2 = mm_readB(rowB + 2, globalCol);
          vec4 BCached3 = mm_readB(rowB + 3, globalCol);

          vec4 ACached = mm_Asub[k];
          acc += BCached0 * ACached.x;
          acc += BCached1 * ACached.y;
          acc += BCached2 * ACached.z;
          acc += BCached3 * ACached.w;
        }

        barrier();
      }

      if (globalRow < dimAOuter && globalCol < dimBOuter) {
        mm_write(globalRow, globalCol, acc);
      }
    }
  `;
}
function makeMatMulPackedVec4SourceWgsl(workPerThread, workGroupSize) {
    const tileInfo = {
        RowPerThread: workPerThread[1],
        ColPerThread: workPerThread[0],
        TileAOuter: workGroupSize[1] * workPerThread[1],
        TileBOuter: workGroupSize[0] * workPerThread[0],
        TileInner: workGroupSize[0] * workPerThread[0]
    };
    return `
  var<workgroup> mm_Asub : array<array<vec4<f32>, ${tileInfo.TileInner / tileInfo.ColPerThread}>, ${tileInfo.TileAOuter}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${tileInfo.TileBOuter / tileInfo.ColPerThread}>, ${tileInfo.TileInner}>;

  let RowPerThread = ${tileInfo.RowPerThread}u;
  let ColPerThread = ${tileInfo.ColPerThread}u; // only support ColPerThread = 4
  let TileAOuter = ${tileInfo.TileAOuter}u;
  let TileBOuter = ${tileInfo.TileBOuter}u;
  let TileInner = ${tileInfo.TileInner}u;

  ${getWorkGroupSizeStringWgsl(workGroupSize)}
  fn main([[builtin(local_invocation_id)]] localId : vec3<u32>,
        [[builtin(global_invocation_id)]] globalId : vec3<u32>) {

    let tileRow = localId.y * RowPerThread;
    let tileCol = localId.x;

    let globalRow = globalId.y * RowPerThread;
    let globalCol = globalId.x;
    let numTiles = (uniforms.dimInner - 1u) / TileInner + 1u;

    var acc: array<vec4<f32>, ${tileInfo.RowPerThread}>;
    var ACached : vec4<f32>;
    var BCached : array<vec4<f32>, 4>;

    // Loop over shared dimension.
    var globalColA = tileCol;
    let RowPerThreadB = TileInner / ${workGroupSize[1]}u;
    let tileRowB = localId.y * RowPerThreadB;
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load one tile of A into local memory.
        for (var innerRow = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, globalColA, globalId);
        }
        globalColA = globalColA + TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (var innerRow = 0u; innerRow < RowPerThreadB; innerRow = innerRow + 1u) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol, globalId);
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0u; k < TileInner / ColPerThread; k = k + 1u) {
            BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
            BCached[1] = mm_Bsub[k * ColPerThread + 1u][tileCol];
            BCached[2] = mm_Bsub[k * ColPerThread + 2u][tileCol];
            BCached[3] = mm_Bsub[k * ColPerThread + 3u][tileCol];

            for (var i = 0u; i < RowPerThread; i = i + 1u) {
                ACached = mm_Asub[tileRow + i][k];
                acc[i] = BCached[0] * ACached.x + acc[i];
                acc[i] = BCached[1] * ACached.y + acc[i];
                acc[i] = BCached[2] * ACached.z + acc[i];
                acc[i] = BCached[3] * ACached.w + acc[i];
            }
        }

        workgroupBarrier();
    }

    for (var innerRow = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
        mm_write(globalRow + innerRow,
                 globalCol,
                 acc[innerRow], globalId);
    }
}`;
}
function makeMatMulVectorVec4SourceWgsl(workGroupSize) {
    return `
  var<workgroup> mm_Asub : array<vec4<f32>, ${workGroupSize[0]}>;
  let tileSize = ${workGroupSize[0] * 4}u;
  ${getWorkGroupSizeStringWgsl(workGroupSize)}
  fn main([[builtin(local_invocation_id)]] localId : vec3<u32>,
        [[builtin(global_invocation_id)]] globalId : vec3<u32>) {
    let tileCol = localId.x;
    let globalCol = globalId.x;
    let globalRow = globalId.y;

    let numTiles = (uniforms.dimInner - 1u) / tileSize + 1u;

    // Without this initialization strange values show up in acc.
    var acc = vec4<f32>(0.0);

    // Loop over shared dimension.
    for (var t = 0u; t < numTiles; t = t + 1u) {
      // Load one tile of A into local memory.
      let colA = t * tileSize / 4u + tileCol;
      mm_Asub[tileCol] = mm_readA(globalRow, colA, globalId);

      workgroupBarrier();

      // Compute acc values for a single thread.
      for (var k = 0u; k < tileSize / 4u; k = k + 1u) {
        let rowB = t * tileSize + k * 4u;
        let BCached0 = mm_readB(rowB, globalCol, globalId);
        let BCached1 = mm_readB(rowB + 1u, globalCol, globalId);
        let BCached2 = mm_readB(rowB + 2u, globalCol, globalId);
        let BCached3 = mm_readB(rowB + 3u, globalCol, globalId);

        let ACached = mm_Asub[k];
        acc = acc + BCached0 * ACached.x;
        acc = acc + BCached1 * ACached.y;
        acc = acc + BCached2 * ACached.z;
        acc = acc + BCached3 * ACached.w;
      }

      workgroupBarrier();
    }

    if (globalRow < uniforms.dimAOuter && globalCol < uniforms.dimBOuter) {
      mm_write(globalRow, globalCol, acc, globalId);
    }
  }
`;
}
class MatMulPackedVec4Program {
    constructor(aShape, outputShape, rowPerThread, bias = null, activation = null, preluActivationWeights = null) {
        this.variableNames = ['A', 'B'];
        this.uniformsWgsl = `dimAOuter : u32; dimBOuter : u32; dimInner : u32;`;
        this.workGroupSize = [16, 16, 1];
        this.isVec4 = true;
        this.vecSize = 4;
        this.outputShape = outputShape;
        this.workGroupSize = computeWorkGroupSizeForMatMul(outputShape[1], aShape[2], outputShape[2]);
        this.dispatchLayout = { x: [2], y: [1], z: [0] };
        if (outputShape[1] === 1) {
            rowPerThread = 1;
        }
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.vecSize, rowPerThread, 1]);
        const addBias = bias != null;
        const hasPreluActivationWeights = preluActivationWeights != null;
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.workPerThread = rowPerThread;
        this.aShape = aShape;
        this.addBias = addBias;
        this.useWgsl = getUseWgsl();
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        [this.fitA, this.fitB] = this.getShapeFit();
        this.shaderKey = `matMulPackedVec4_${rowPerThread}_${this.activation}_${this.fitA}_${this.fitB}_${this.outputShape[1] > 1}`;
    }
    getShapeFit() {
        const dimInner = this.aShape[2];
        const dimBOuter = this.outputShape[2];
        const bShape = [this.outputShape[0], dimInner, dimBOuter];
        const tileAOuter = this.workGroupSize[1] * this.workPerThread;
        const tileBOuter = this.workGroupSize[0] * this.vecSize;
        const tileInner = tileBOuter; // Make sure tileInner is divisible by 4.
        const tileSizeA = [tileAOuter, tileInner];
        const tileSizeB = [tileInner, tileBOuter];
        return [
            tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1)),
            tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1))
        ];
    }
    getUserCode() {
        const sampleA = this.fitA ?
            `A[batch * batchASize + row * dimInner / 4 + col]` :
            `coordsInBounds(ivec2(row, col * 4), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner / 4 + col] :
            vec4(0.0)`;
        const sampleB = this.fitB ?
            `B[batch * batchBSize + row * dimBOuter / 4 + col]` :
            `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter / 4 + col] :
            vec4(0.0)`;
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation, this.isVec4, this.useWgsl);
            if (this.hasPreluActivationWeights) {
                activationSnippet = `vec4 activation(vec4 a, ivec3 outCoord) {
                  vec4 b = getPreluActivationWeightsAtOutCoords(outCoord);
                  ${activationOp}
                }`;
            }
            else {
                activationSnippet = `
                vec4 activation(vec4 a, ivec3 outCoord) {
                  ${activationOp}
                }`;
            }
            applyActivationSnippet = 'value = activation(value, outCoord);';
        }
        const addBiasSnippet = this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';
        const userCode = `
      ${activationSnippet}
      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];
      int batch;

      ${this.outputShape[1] > 1 ?
            makeMatMulPackedVec4Source([this.vecSize, this.workPerThread, 1]) :
            makeMatMulVectorVec4Source()}

      vec4 mm_readA(int row, int col) {
        int batchASize = aShape[1] * aShape[2] / ${this.vecSize};
        return ${sampleA};
      }

      vec4 mm_readB(int row, int col) {
        int batchBSize = bShape[1] * bShape[2] / ${this.vecSize};
        return ${sampleB};
      }

      void mm_write(int row, int col, vec4 value) {
        if (row < dimAOuter && col * 4 < dimBOuter)
        {
          ivec3 outCoord = ivec3(batch, row, col * 4);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(outCoord[0], outCoord[1], outCoord[2], value);
        }
      }

      void main() {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const sampleA = this.fitA ?
            `return A.numbers[batch * batchASize + row * uniforms.dimInner / 4u + col]` :
            `if (coordsInBounds2D(vec2<u32>(row, col * 4u), vec2<u32>(uniforms.dimAOuter, uniforms.dimInner))) {
            return A.numbers[batch * batchASize + row * uniforms.dimInner / 4u + col];
        }
        return vec4<f32>(0.0)`;
        const sampleB = this.fitB ?
            `return B.numbers[batch * batchBSize + row * uniforms.dimBOuter / 4u + col]` :
            `if(coordsInBounds2D(vec2<u32>(row, col * 4u), vec2<u32>(uniforms.dimInner, uniforms.dimBOuter))) {
             return B.numbers[batch * batchBSize + row * uniforms.dimBOuter / 4u + col];
        }
        return vec4<f32>(0.0)`;
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation, this.isVec4, this.useWgsl);
            if (this.hasPreluActivationWeights) {
                activationSnippet =
                    `fn activation(a : vec4<f32>, outCoord : vec3<u32>) -> vec4<f32> {
                  let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
                  ${activationOp}
                }`;
            }
            else {
                activationSnippet = `
            fn activation(a : vec4<f32>, outCoord : vec3<u32>) -> vec4<f32> {
              ${activationOp}
            }`;
            }
            applyActivationSnippet = 'value = activation(value, outCoord);';
        }
        const addBiasSnippet = this.addBias ?
            'value = value + getBiasAtOutCoordsByCoords(outCoord);' :
            '';
        const userCode = `
      ${activationSnippet}
      fn mm_readA(row : u32, col : u32,  globalId : vec3<u32>) -> vec4<f32> {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2] / ${this.vecSize}u;
        let batch = globalId.z;
        ${sampleA};
      }

      fn mm_readB(row : u32, col : u32,  globalId : vec3<u32>) -> vec4<f32> {
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2] / ${this.vecSize}u;
        let batch = globalId.z;
        ${sampleB};
      }

      fn mm_write(row : u32, col : u32, valueIn : vec4<f32>, globalId : vec3<u32>) {
        if (row < uniforms.aShape[1] && col * 4u < uniforms.bShape[2])
        {
          var value = valueIn;
          let batch = globalId.z;
          let outCoord = vec3<u32>(batch, row, col * 4u);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(outCoord[0], outCoord[1], outCoord[2], value);
        }
      }
      ${this.outputShape[1] > 1 ?
            makeMatMulPackedVec4SourceWgsl([this.vecSize, this.workPerThread, 1], this.workGroupSize) :
            makeMatMulVectorVec4SourceWgsl(this.workGroupSize)}

    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function makeMatMulPackedSource(workPerThread) {
    return `
    float mm_readA(int row, int col);
    float mm_readB(int row, int col);
    void mm_write(int row, int col, float value);
    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${workPerThread[0]};
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

    shared float mm_Asub[TileAOuter][TileInner];
    shared float mm_Bsub[TileInner][TileBOuter];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

      int numTiles = (dimInner - 1) / TileInner + 1;

      float acc[RowPerThread][ColPerThread];
      float ACached;
      float BCached[ColPerThread];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      const int ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
      int tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileColA + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * TileInner + inputCol);
          }
        }
        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol + innerCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol + innerCol);;
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner; k++) {
          for (int inner = 0; inner < ColPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        barrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {

          if ((globalCol + innerCol) < dimBOuter &&
              (globalRow + innerRow) < dimAOuter) {
            mm_write(globalRow + innerRow,
                     globalCol + innerCol,
                     acc[innerRow][innerCol]);
          }
        }
      }
    }
  `;
}
function makeMatMulVectorSource() {
    return `
    float mm_readA(int row, int col);
    float mm_readB(int row, int col);
    void mm_write(int row, int col, float value);
    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);

    const int TileSize = int(gl_WorkGroupSize.x) * 4;

    shared vec4 mm_Asub[TileSize / 4];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileCol = int(gl_LocalInvocationID.x);
      int globalCol = int(gl_GlobalInvocationID.x);
      int globalRow = int(gl_GlobalInvocationID.y);

      int numTiles = (dimInner - 1) / TileSize + 1;

      // Without this initialization strange values show up in acc.
      float acc = 0.0;

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        int colA = t * TileSize + tileCol * 4;
        mm_Asub[tileCol] = vec4(mm_readA(globalRow, colA),
                                mm_readA(globalRow, colA + 1),
                                mm_readA(globalRow, colA + 2),
                                mm_readA(globalRow, colA + 3));
        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileSize / 4; k++) {
          int rowB = t * TileSize + k * 4;
          vec4 BCached = vec4(mm_readB(rowB, globalCol),
                              mm_readB(rowB + 1, globalCol),
                              mm_readB(rowB + 2, globalCol),
                              mm_readB(rowB + 3, globalCol));

          vec4 ACached = mm_Asub[k];
          acc += dot(ACached, BCached);
        }

        barrier();
      }

      if (globalRow < dimAOuter && globalCol < dimBOuter) {
        mm_write(globalRow, globalCol, acc);
      }
    }
  `;
}
class MatMulPackedProgram {
    constructor(aShape, outputShape, workPerThread, transposeA = false, transposeB = false, bias = null, activation = null, preluActivationWeights = null) {
        this.variableNames = ['A', 'B'];
        this.workGroupSize = [16, 16, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [2], y: [1], z: [0] };
        const dimInner = transposeA ? aShape[1] : aShape[2];
        this.workGroupSize =
            computeWorkGroupSizeForMatMul(outputShape[1], dimInner, outputShape[2]);
        if (outputShape[1] === 1 || outputShape[2] === 1) {
            workPerThread = 1;
        }
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [workPerThread, workPerThread, 1]);
        // If dispaching number is one, it means only one work group is running.
        // For modern GPUs, it supports multiple work groups running in parallel.
        // So there may be some idle hardware threads.
        // In this case, we prefer to reduce the work per thread and improve the
        // thread utilization
        if (util.arraysEqual(this.dispatch, [1, 1, 1])) {
            workPerThread = 1;
            this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [workPerThread, workPerThread, 1]);
        }
        const addBias = bias != null;
        const hasPreluActivationWeights = preluActivationWeights != null;
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.workPerThread = workPerThread;
        this.aShape = aShape;
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        const dimBOuter = this.outputShape[2];
        const bShape = this.transposeB ?
            [this.outputShape[0], dimBOuter, dimInner] :
            [this.outputShape[0], dimInner, dimBOuter];
        [this.fitA, this.fitB] = this.getShapeFit(bShape);
        this.shaderKey = `matMulPacked_${this.workPerThread}_${transposeA}_${transposeB}_${this.activation}_${this.fitA}_${this.fitB}_${this.outputShape[1] > 1}`;
    }
    getShapeFit(bShape) {
        const tileAOuter = this.workGroupSize[1] * this.workPerThread;
        const tileBOuter = this.workGroupSize[0] * this.workPerThread;
        let tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
        if (this.outputShape[1] === 1) {
            tileInner *=
                4; // for makeMatMulVectorSource, tileSize = gl_WorkGroupSize.x * 4.
        }
        util.assert(tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0, () => `tileInner must be multiple of workgroupsize.x ` +
            `and workgroupsize.y`);
        const tileSizeA = [tileAOuter, tileInner];
        const tileSizeB = [tileInner, tileBOuter];
        return [
            tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1)),
            tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1))
        ];
    }
    getUserCode() {
        let sampleA;
        if (this.transposeA === false) {
            sampleA = this.fitA ?
                `A[batch * batchASize + row * dimInner + col]` :
                `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner + col] : 0`;
        }
        else {
            sampleA = this.fitA ?
                `A[batch * batchASize + col * dimAOuter + row]` :
                `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch* batchASize + col * dimAOuter + row] : 0`;
        }
        let sampleB;
        if (this.transposeB === false) {
            sampleB = this.fitB ?
                `B[batch * batchBSize + row * dimBOuter + col]` :
                `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter + col] : 0`;
        }
        else {
            sampleB = this.fitB ?
                `B[batch * batchBSize + col * dimInner + row]` :
                `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + col * dimInner + row] : 0`;
        }
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation);
            if (this.hasPreluActivationWeights) {
                activationSnippet = `float activation(float a, ivec3 outCoord) {
              float b = getPreluActivationWeightsAtOutCoords(outCoord);
              ${activationOp}
            }`;
            }
            else {
                activationSnippet = `
              float activation(float a, ivec3 outCoord) {
                ${activationOp}
              }
            `;
            }
            applyActivationSnippet = 'value = activation(value, outCoord);';
        }
        const addBiasSnippet = this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';
        const userCode = `
      ${activationSnippet}

      int dimAOuter = ${this.transposeA === true ? `aShape[2]` : `aShape[1]`};
      int dimInner = ${this.transposeA === true ? `aShape[1]` : `aShape[2]`};
      int dimBOuter = ${this.transposeB === true ? `bShape[1]` : `bShape[2]`};

      int batch;

      ${this.outputShape[1] > 1 ?
            makeMatMulPackedSource([this.workPerThread, this.workPerThread, 1]) :
            makeMatMulVectorSource()}
      float mm_readA(int row, int col) {
        int batchASize = aShape[1] * aShape[2];
        return ${sampleA};
      }
      float mm_readB(int row, int col) {
        int batchBSize = bShape[1] * bShape[2];
        return ${sampleB};
      }
      void mm_write(int row, int col, float value) {
        ivec3 outCoord = ivec3(batch, row, col);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(batch, row, col, value);
      }
      void main() {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function reshape(args) {
    const { inputs, attrs } = args;
    const { x } = inputs;
    const { shape } = attrs;
    const xSize = util.sizeFromShape(x.shape);
    const $shape = util.inferFromImplicitShape(shape, xSize);
    const $xSize = util.sizeFromShape($shape);
    util.assert(xSize === $xSize, () => `The new shape (${$shape}) has ${$xSize} elements and the old ` +
        `shape (${x.shape}) has ${xSize} elements. The new shape and old ` +
        `shape must have the same number of elements.`);
    // Backend needs to track refCount for the dataId for reshape op
    args.backend.incRef(x.dataId);
    return { dataId: x.dataId, shape: $shape, dtype: x.dtype };
}
const reshapeConfig = {
    kernelName: Reshape,
    backendName: 'webgpu',
    kernelFunc: reshape
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function batchMatMulImpl({ a, b, transposeA, transposeB, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    const aRank = a.shape.length;
    const bRank = b.shape.length;
    const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
    const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];
    const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
    const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];
    const outerDimsA = a.shape.slice(0, -2);
    const outerDimsB = b.shape.slice(0, -2);
    const batchDimA = util.sizeFromShape(outerDimsA);
    const batchDimB = util.sizeFromShape(outerDimsB);
    const batchDimsCompatible = batchDimA === batchDimB || batchDimA === 1 || batchDimB === 1;
    util.assert(aRank >= 2 && bRank >= 2 && batchDimsCompatible, () => `Error in matMul: the input batch dimensions must either be the ` +
        `same or at least one input batch dimension must be 1. Got input ` +
        `batch dimensions of (${outerDimsA}) and (${outerDimsB}).`);
    const outShapeOuterDims = batchDimA > batchDimB ? a.shape.slice(0, -2) : b.shape.slice(0, -2);
    const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);
    util.assert(innerShapeA === innerShapeB, () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
        `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
        `${b.shape} and transposeA=${transposeA}` +
        ` and transposeB=${transposeB} must match.`);
    const a3dShape = transposeA ?
        [batchDimA, innerShapeA, outerShapeA] :
        [batchDimA, outerShapeA, innerShapeA];
    const b3dShape = transposeB ?
        [batchDimB, outerShapeB, innerShapeB] :
        [batchDimB, innerShapeB, outerShapeB];
    // The rest of the implementation is designed to operate on rank-3 tensors
    const a3d = reshape({ inputs: { x: a }, backend, attrs: { shape: a3dShape } });
    const b3d = reshape({ inputs: { x: b }, backend, attrs: { shape: b3dShape } });
    const intermediates = [a3d, b3d];
    const batchDim = Math.max(batchDimA, batchDimB);
    const useVec4 = a.shape[2] % 4 === 0 && b.shape[2] % 4 === 0 && !transposeA &&
        !transposeB && outerShapeB >= 32;
    let program;
    let dimensions = null;
    // When the output size is absolutely small or relatively small, we may use
    // MatMulSmallOutputSizeProgram to get better performance.
    // Absolutely small size means that the output size is smaller than [16, 512].
    // Relatively small size means that one demension size of the output is
    // smaller than 16, and the output size is also more than or equal two times
    // smaller than each of the two input sizes. For example, if input sizes are
    // [12, 2048] and [2048, 1024], the output size is [12, 1024], which is
    // relatively small compared to input sizes.
    if (!transposeA && !transposeB && ((a.shape[1] <= 16 &&
        (b.shape[2] <= 512 || b.shape[1] >= 2 * b.shape[2])) ||
        (b.shape[2] <= 16 &&
            (a.shape[1] <= 512 || a.shape[2] >= 2 * a.shape[1])))) {
        program = new MatMulSmallOutputSizeProgram(a3dShape, b3dShape, [batchDim, outerShapeA, outerShapeB], bias, activation, preluActivationWeights);
    }
    else if (useVec4) {
        // TODO: Currently we need to make sure that a.shape[2] and b.shape[2]
        // are divisible by 4 since we use vec4 to get data. In future, we can
        // remove this limitation by insert 0 to pack data.
        program = new MatMulPackedVec4Program(a3dShape, [batchDim, outerShapeA, outerShapeB], env().get('WEBGPU_MATMUL_WORK_PER_THREAD'), bias, activation, preluActivationWeights);
        if (program.useWgsl) {
            const dimAOuter = a3d.shape[1];
            const dimInner = a3d.shape[2];
            const dimBOuter = b3d.shape[2];
            dimensions = [
                { type: 'uint32', data: [dimAOuter] },
                { type: 'uint32', data: [dimBOuter] }, { type: 'uint32', data: [dimInner] }
            ];
        }
    }
    else {
        program = new MatMulPackedProgram(a3dShape, [batchDim, outerShapeA, outerShapeB], env().get('WEBGPU_MATMUL_WORK_PER_THREAD'), transposeA, transposeB, bias, activation, preluActivationWeights);
    }
    const inputs = [a3d, b3d];
    if (bias) {
        inputs.push(bias);
    }
    if (preluActivationWeights) {
        inputs.push(preluActivationWeights);
    }
    const out = backend.runWebGPUProgram(program, inputs, a.dtype, dimensions);
    const outReshaped = reshape({ inputs: { x: out }, backend, attrs: { shape: outShape } });
    intermediates.push(out);
    for (const i of intermediates) {
        backend.disposeData(i.dataId);
    }
    return outReshaped;
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function _fusedMatMul(args) {
    const { inputs, backend, attrs } = args;
    const { a, b, bias, preluActivationWeights } = inputs;
    const { transposeA, transposeB, activation, leakyreluAlpha } = attrs;
    return batchMatMulImpl({
        a,
        b,
        transposeA,
        transposeB,
        backend,
        bias,
        preluActivationWeights,
        leakyreluAlpha,
        activation
    });
}
const _fusedMatMulConfig = {
    kernelName: _FusedMatMul,
    backendName: 'webgpu',
    kernelFunc: _fusedMatMul,
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class BinaryOpComplexProgram {
    constructor(op, aShape, bShape) {
        this.variableNames = ['AReal', 'AImag', 'BReal', 'BImag'];
        this.workGroupSize = [128, 1, 1];
        this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.shaderKey = `binaryOpComplex_${op}`;
        this.op = op;
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const opStr = getBinaryOpString(this.op);
        const userCode = `
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${opStr}
      }

      void main() {
        int index = getGlobalIndex();
        if(index < size) {
          float areal = getARealAtOutCoords();
          float aimag = getAImagAtOutCoords();
          float breal = getBRealAtOutCoords();
          float bimag = getBImagAtOutCoords();
          setOutput(index, binaryOpComplex(areal, aimag, breal, bimag));
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const opStr = getBinaryOpString(this.op, false, true);
        const userCode = `
      fn binaryOpComplex(
          areal : f32, aimag : f32, breal : f32, bimag : f32) -> f32 {
        ${opStr}
      }

      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if(index < uniforms.size) {
          let areal = getARealAtOutCoordsByGlobalId(globalId, index);
          let aimag = getAImagAtOutCoordsByGlobalId(globalId, index);
          let breal = getBRealAtOutCoordsByGlobalId(globalId, index);
          let bimag = getBImagAtOutCoordsByGlobalId(globalId, index);
          setOutputFlat(index, binaryOpComplex(areal, aimag, breal, bimag));
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class BinaryOpSharedProgram {
    constructor(op, aShape, bShape, useSharedMemoryWithB) {
        this.variableNames = ['A', 'B'];
        // This is an experimental value when using shared memory.
        // Note that the maximum of workgroup X dimension is 256.
        const workGroupSizeX = 256;
        this.workGroupSize = [workGroupSizeX, 1, 1];
        this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.lastDimensionSize = useSharedMemoryWithB ? bShape[0] : aShape[0];
        if (this.lastDimensionSize < 256) {
            this.workPerThread = 1;
        }
        else if (this.lastDimensionSize < 512) {
            this.workPerThread = 2;
        }
        else {
            this.workPerThread = 4;
        }
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.useSharedMemoryWithB = useSharedMemoryWithB;
        this.op = op;
        this.useWgsl = getUseWgsl();
        this.size = util.sizeFromShape(this.outputShape);
        this.sizeFit =
            this.size % (this.workGroupSize[0] * this.workPerThread) === 0;
        // this.lastDimensionSize is used as sharedBuf array size, so can not be
        // used as uniform.
        this.shaderKey = `binaryShared_${op}_${this.lastDimensionSize}_${this.useSharedMemoryWithB}_${this.sizeFit}`;
    }
    getUserCode() {
        const type = getCoordsDataType(this.outputShape.length);
        const sharedIndexSnippet = this.lastDimensionSize > 1 ?
            `coords[${this.outputShape.length - 1}]` :
            '0';
        const accessDataSnippet = this.useSharedMemoryWithB ?
            `float a = getAAtOutCoords(coords);
         float b = sharedBuf[${sharedIndexSnippet}];` :
            `float a = sharedBuf[${sharedIndexSnippet}];
         float b = getBAtOutCoords(coords);`;
        const writeDataSnippet = this.sizeFit ?
            `${type} coords = getCoordsFromFlatIndex(flatIndex);

         ${accessDataSnippet}
         setOutput(flatIndex, binaryOperation(a, b));` :
            `if(flatIndex < size) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            ${accessDataSnippet}
            setOutput(flatIndex, binaryOperation(a, b));
          }`;
        const opStr = getBinaryOpString(this.op);
        const userCode = `
        float binaryOperation(float a, float b) {
          ${opStr}
        }

        shared float sharedBuf[${this.lastDimensionSize}];
        void main() {
          int index = getGlobalIndex();
          int localIndex = int(gl_LocalInvocationIndex);

          // Fill in the shared memory buffer. Here we need a loop to make sure
          // that all data in A|B are uploaded when |sharedMemorySize| is larger
          // than work group size.
          while(localIndex < ${this.lastDimensionSize})
          {
            sharedBuf[localIndex] = ${this.useSharedMemoryWithB ? 'B' : 'A'}[localIndex];
            localIndex += int(gl_WorkGroupSize.x);
          }
          barrier();

          for(int i = 0; i < ${this.workPerThread}; i++) {
            int flatIndex = index * ${this.workPerThread} + i;

            ${writeDataSnippet}
          }
        }
        `;
        return userCode;
    }
    getUserCodeWgsl() {
        const sharedIndexSnippet = this.lastDimensionSize > 1 ?
            `coords[${this.outputShape.length - 1}]` :
            '0';
        const accessDataSnippet = this.useSharedMemoryWithB ?
            `let a = getAAtOutCoordsByCoords(coords);
         let b = sharedBuf[${sharedIndexSnippet}];` :
            `let a = sharedBuf[${sharedIndexSnippet}];
         let b = getBAtOutCoordsByCoords(coords);`;
        const writeDataSnippet = this.sizeFit ?
            `let coords = getCoordsFromFlatIndex(flatIndex);

         ${accessDataSnippet}
         setOutputFlat(flatIndex, binaryOperation(a, b));` :
            `if(flatIndex < uniforms.size) {
            let coords = getCoordsFromFlatIndex(flatIndex);

            ${accessDataSnippet}
            setOutputFlat(flatIndex, binaryOperation(a, b));
          }`;
        const opStr = getBinaryOpString(this.op, false, this.useWgsl);
        const userCode = `
        fn binaryOperation(a : f32, b : f32) -> f32 {
          ${opStr}
        }
        var<workgroup> sharedBuf : array<f32, ${this.lastDimensionSize}>;
        ${getMainHeaderStringWgsl(this.workGroupSize)} {
          ${getGlobalIndexStringWgsl(this.workGroupSize)}

          // Fill in the shared memory buffer. Here we need a loop to make sure
          // that all data in A|B are uploaded when |sharedMemorySize| is larger
          // than work group size.
          for(var localIndex = localId.x; localIndex < ${this.lastDimensionSize}u; localIndex = localIndex + ${this.workGroupSize[0]}u) {
            sharedBuf[localIndex] = f32(${this.useSharedMemoryWithB ? 'B' : 'A'}.numbers[localIndex]);
          }
          workgroupBarrier();

          for(var i = 0u; i < ${this.workPerThread}u; i = i + 1u) {
            let flatIndex = index * ${this.workPerThread}u + i;

            ${writeDataSnippet}
          }
        }
        `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class BinaryOpVec4Program {
    constructor(op, aShape, bShape) {
        this.variableNames = ['A', 'B'];
        this.workPerThread = 4;
        this.isVec4 = true;
        // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
        const workGroupSizeX = 128;
        this.workGroupSize = [workGroupSizeX, 1, 1];
        this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.op = op;
        this.fitShape = this.size % this.workGroupSize[0] === 0;
        this.shaderKey = `binaryVec4_${op}_${this.fitShape}`;
        this.size = util.sizeFromShape(this.outputShape) / this.workPerThread;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        let userCode;
        const opStr = getBinaryOpString(this.op, this.isVec4);
        if (this.fitShape) {
            userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${opStr}
      }

      void main() {
        int index = getGlobalIndex();
        vec4 a = vec4(A[index]);
        vec4 b = vec4(B[index]);
        setOutput(index, binaryOperation(a, b));
      }
    `;
        }
        else {
            userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${opStr}
      }

      void main() {
        int index = getGlobalIndex();
        if (index < size)
        {
          vec4 a = vec4(A[index]);
          vec4 b = vec4(B[index]);
          setOutput(index, binaryOperation(a, b));
        }
      }
    `;
        }
        return userCode;
    }
    getUserCodeWgsl() {
        let userCode;
        const opStr = getBinaryOpString(this.op, this.isVec4, this.useWgsl);
        const miscStr = `fn binaryOperation(a : vec4<f32>, b : vec4<f32>) -> vec4<f32> {
          ${opStr}
        }`;
        if (this.fitShape) {
            userCode = `
      ${miscStr}
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let a = vec4<f32>(A.numbers[index]);
        let b = vec4<f32>(B.numbers[index]);
        setOutputFlat(index, binaryOperation(a, b));
      }
    `;
        }
        else {
            userCode = `
      ${miscStr}
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if (index < uniforms.size) {
          let a = vec4<f32>(A.numbers[index]);
          let b = vec4<f32>(B.numbers[index]);
          setOutputFlat(index, binaryOperation(a, b));
        }
      }
    `;
        }
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class BinaryOpProgram {
    constructor(op, aShape, bShape) {
        this.variableNames = ['A', 'B'];
        // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
        const workGroupSizeX = 128;
        this.workGroupSize = [workGroupSizeX, 1, 1];
        this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.size = util.sizeFromShape(this.outputShape);
        this.sizeFit = this.size % workGroupSizeX === 0;
        this.shapesFit = util.arraysEqual(aShape, bShape) && this.sizeFit;
        this.workPerThread = this.sizeFit || this.shapesFit ? 1 : 2;
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.shaderKey = `binary_${op}_${this.sizeFit}_${this.shapesFit}`;
        this.useWgsl = getUseWgsl();
        this.op = op;
    }
    getUserCode() {
        let userCode;
        const opStr = getBinaryOpString(this.op);
        if (this.shapesFit) {
            userCode = `
          float binaryOperation(float a, float b) {
            ${opStr}
          }

          void main() {
            int index = getGlobalIndex();

            float a = float(A[index]);
            float b = float(B[index]);
            setOutput(index, binaryOperation(a, b));
          }
        `;
        }
        else if (this.sizeFit) {
            const type = getCoordsDataType(this.outputShape.length);
            userCode = `
      float binaryOperation(float a, float b) {
        ${opStr}
      }

      void main() {
        int index = getGlobalIndex();

        ${type} coords = getCoordsFromFlatIndex(index);

        float a = getAAtOutCoords(coords);
        float b = getBAtOutCoords(coords);
        setOutput(index, binaryOperation(a, b));
      }
      `;
        }
        else {
            const type = getCoordsDataType(this.outputShape.length);
            userCode = `
      float binaryOperation(float a, float b) {
        ${opStr}
      }

      void main() {
        int index = getGlobalIndex();

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if(flatIndex < size) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            float a = getAAtOutCoords(coords);
            float b = getBAtOutCoords(coords);
            setOutput(flatIndex, binaryOperation(a, b));
          }
        }
      }
      `;
        }
        return userCode;
    }
    getUserCodeWgsl() {
        let userCode;
        const opStr = getBinaryOpString(this.op, false, this.useWgsl);
        const miscStr = `          fn binaryOperation(a : f32, b : f32) -> f32 {
      ${opStr}
    }`;
        if (this.shapesFit) {
            userCode = `
          ${miscStr}
          ${getMainHeaderStringWgsl(this.workGroupSize)} {
            ${getGlobalIndexStringWgsl(this.workGroupSize)}

            let a = f32(A[index]);
            let b = f32(B[index]);
            setOutputFlat(index, binaryOperation(a, b));
          }
        `;
        }
        else if (this.sizeFit) {
            userCode = `
      ${miscStr}
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}

        let coords = getCoordsFromFlatIndex(index);

        let a = getAAtOutCoordsByCoords(coords);
        let b = getBAtOutCoordsByCoords(coords);
        setOutputFlat(index, binaryOperation(a, b));
      }
      `;
        }
        else {
            userCode = `
      ${miscStr}
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        for (var i = 0u; i < ${this.workPerThread}u; i = i + 1u ) {
          let flatIndex = index * ${this.workPerThread}u + i;

          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromFlatIndex(flatIndex);

            let a = getAAtOutCoordsByCoords(coords);
            let b = getBAtOutCoordsByCoords(coords);
            setOutputFlat(flatIndex, binaryOperation(a, b));
          }
        }
      }
      `;
        }
        return userCode;
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function getBinaryProgram(op, aShape, bShape) {
    const useVec4 = util.arraysEqual(aShape, bShape) && util.sizeFromShape(aShape) % 4 === 0;
    if (useVec4) {
        return new BinaryOpVec4Program(op, aShape, bShape);
    }
    const useSharedMemoryWithA = aShape.length === 1 && bShape.length > 1 && aShape[0] < 1024;
    const useSharedMemoryWithB = bShape.length === 1 && aShape.length > 1 && bShape[0] < 1024;
    if (useSharedMemoryWithA || useSharedMemoryWithB) {
        return new BinaryOpSharedProgram(op, aShape, bShape, useSharedMemoryWithB);
    }
    else {
        return new BinaryOpProgram(op, aShape, bShape);
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function identity(args) {
    const { inputs } = args;
    const { x } = inputs;
    args.backend.incRef(x.dataId);
    return { dataId: x.dataId, shape: x.shape, dtype: x.dtype };
}
const identityConfig = {
    kernelName: Identity,
    backendName: 'webgpu',
    kernelFunc: identity
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * Complex tensors share data with their real and imaginary components. Complex
 * tensors' reference to the components is tracked by refCount on the individual
 * component. The refCounts are increased by the identity call.
 *
 * When a complex tensor is disposed, it will reduce the refCount on the
 * components by calling disposeData on each.
 */
function complex(args) {
    const { inputs, backend } = args;
    const { real, imag } = inputs;
    const complexInfo = backend.makeTensorInfo(real.shape, 'complex64');
    const complex = backend.tensorMap.get(complexInfo.dataId);
    const realTensorInfo = identity({ inputs: { x: real }, backend });
    const imagTensorInfo = identity({ inputs: { x: imag }, backend });
    complex.complexTensorInfos = { real: realTensorInfo, imag: imagTensorInfo };
    return complexInfo;
}
const complexConfig = {
    kernelName: Complex,
    backendName: 'webgpu',
    kernelFunc: complex
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class UnaryOpProgram {
    constructor(outputShape, op) {
        this.variableNames = ['A'];
        // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
        const workGroupSizeX = 128;
        this.workGroupSize = [workGroupSizeX, 1, 1];
        this.outputShape = outputShape;
        this.size = util.sizeFromShape(this.outputShape);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.useWgsl = getUseWgsl();
        this.op = op;
        this.shaderKey = `unary_${op}`;
    }
    getUserCode() {
        return `
      float unaryOperation(float a) {
        ${getUnaryOpString(this.op)}
      }

      void main() {
        int index = getGlobalIndex();
        if (index < size)
        {
          float a = getAAtOutCoords();
          setOutput(index, unaryOperation(a));
        }
      }
      `;
    }
    getUserCodeWgsl() {
        return `
      fn unaryOperation(a : f32) -> f32 {
        ${getUnaryOpString(this.op, false, true)}
      }
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if (index < uniforms.size) {
          let a = getAAtOutCoordsByGlobalId(globalId, index);
          setOutputFlat(index, unaryOperation(a));
        }
      }
      `;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param opSnippet Op snippet to create `UnaryOpProgram`.
 * @param cpuKernelImpl Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
function unaryKernelFunc({ opType, cpuKernelImpl, dtype }) {
    return ({ inputs, backend }) => {
        const { x } = inputs;
        const webgpuBackend = backend;
        const $dtype = dtype || x.dtype;
        if (webgpuBackend.shouldExecuteOnCPU([x]) && cpuKernelImpl != null) {
            const xData = webgpuBackend.tensorMap.get(x.dataId);
            const outValues = cpuKernelImpl(xData.values, $dtype);
            return webgpuBackend.makeTensorInfo(x.shape, $dtype, outValues);
        }
        const program = new UnaryOpProgram(x.shape, opType);
        return webgpuBackend.runWebGPUProgram(program, [x], $dtype);
    };
}
/**
 * Template that creates a `KernelFunc` for binary ops.
 * @param opSnippet Op snippet to create `BinaryOpProgram`.
 * @param cpuKernelImpl Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
function binaryKernelFunc({ opSnippet, cpuKernelImpl, supportsComplex = false, dtype }) {
    return ({ inputs, backend }) => {
        const { a, b } = inputs;
        const webgpuBackend = backend;
        if (supportsComplex && a.dtype === 'complex64') {
            const aData = webgpuBackend.tensorMap.get(a.dataId);
            const bData = webgpuBackend.tensorMap.get(b.dataId);
            let real, imag;
            if (opSnippet !== BinaryOpType.MUL) {
                [real, imag] = [
                    [aData.complexTensorInfos.real, bData.complexTensorInfos.real],
                    [aData.complexTensorInfos.imag, bData.complexTensorInfos.imag]
                ].map(complexParts => {
                    const [aPart, bPart] = complexParts;
                    const aHandle = {
                        dataId: aPart.dataId,
                        dtype: aPart.dtype,
                        shape: a.shape
                    };
                    const bHandle = {
                        dataId: bPart.dataId,
                        dtype: bPart.dtype,
                        shape: b.shape
                    };
                    const program = getBinaryProgram(opSnippet, a.shape, b.shape);
                    return webgpuBackend.runWebGPUProgram(program, [aHandle, bHandle], upcastType(aPart.dtype, bPart.dtype));
                });
            }
            else {
                const realProgram = new BinaryOpComplexProgram(BinaryOpType.COMPLEX_MULTIPLY_REAL, a.shape, b.shape);
                const imagProgram = new BinaryOpComplexProgram(BinaryOpType.COMPLEX_MULTIPLY_IMAG, a.shape, b.shape);
                const inputs = [
                    {
                        dataId: aData.complexTensorInfos.real.dataId,
                        dtype: aData.complexTensorInfos.real.dtype,
                        shape: a.shape
                    },
                    {
                        dataId: aData.complexTensorInfos.imag.dataId,
                        dtype: aData.complexTensorInfos.imag.dtype,
                        shape: a.shape
                    },
                    {
                        dataId: bData.complexTensorInfos.real.dataId,
                        dtype: bData.complexTensorInfos.real.dtype,
                        shape: b.shape
                    },
                    {
                        dataId: bData.complexTensorInfos.imag.dataId,
                        dtype: bData.complexTensorInfos.imag.dtype,
                        shape: b.shape
                    }
                ];
                real = webgpuBackend.runWebGPUProgram(realProgram, inputs, 'float32');
                imag = webgpuBackend.runWebGPUProgram(imagProgram, inputs, 'float32');
            }
            const complexOutput = complex({ inputs: { real, imag }, backend: webgpuBackend });
            webgpuBackend.disposeData(real.dataId);
            webgpuBackend.disposeData(imag.dataId);
            // TODO: Implement CPU forwarding for complex inputs.
            return complexOutput;
        }
        const $dtype = dtype || upcastType(a.dtype, b.dtype);
        if ((a.dtype === 'string' || b.dtype === 'string' ||
            webgpuBackend.shouldExecuteOnCPU([a, b])) &&
            cpuKernelImpl != null) {
            const aData = webgpuBackend.tensorMap.get(a.dataId).values;
            const bData = webgpuBackend.tensorMap.get(b.dataId).values;
            const decodedAVals = a.dtype === 'string' ?
                // tslint:disable-next-line: no-any
                backend_util.fromUint8ToStringArray(aData) :
                aData;
            const decodedBVals = a.dtype === 'string' ?
                // tslint:disable-next-line: no-any
                backend_util.fromUint8ToStringArray(bData) :
                bData;
            const [outValues, outShape] = cpuKernelImpl(a.shape, b.shape, decodedAVals, decodedBVals, $dtype);
            return webgpuBackend.makeTensorInfo(outShape, $dtype, outValues);
        }
        const program = getBinaryProgram(opSnippet, a.shape, b.shape);
        return webgpuBackend.runWebGPUProgram(program, [a, b], $dtype);
    };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function simpleAbsImpl(vals) {
    const resultValues = new Float32Array(vals.length);
    for (let i = 0; i < vals.length; ++i) {
        resultValues[i] = Math.abs(vals[i]);
    }
    return resultValues;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * Template that creates implementation for binary ops. Supports broadcast.
 */
function createSimpleBinaryKernelImpl(op) {
    return (aShape, bShape, aVals, bVals, dtype) => {
        const newShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        const resultRank = newShape.length;
        const resultStrides = util.computeStrides(newShape);
        const resultSize = util.sizeFromShape(newShape);
        const result = util.getTypedArrayFromDType(dtype, resultSize);
        const aRank = aShape.length;
        const bRank = bShape.length;
        const aStrides = util.computeStrides(aShape);
        const bStrides = util.computeStrides(bShape);
        const aBroadcastDims = backend_util.getBroadcastDims(aShape, newShape);
        const bBroadcastDims = backend_util.getBroadcastDims(bShape, newShape);
        if (aBroadcastDims.length + bBroadcastDims.length === 0) {
            for (let i = 0; i < result.length; ++i) {
                result[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
            }
        }
        else {
            for (let i = 0; i < result.length; ++i) {
                const loc = util.indexToLoc(i, resultRank, resultStrides);
                const aLoc = loc.slice(-aRank);
                aBroadcastDims.forEach(d => aLoc[d] = 0);
                const aIndex = util.locToIndex(aLoc, aRank, aStrides);
                const bLoc = loc.slice(-bRank);
                bBroadcastDims.forEach(d => bLoc[d] = 0);
                const bIndex = util.locToIndex(bLoc, bRank, bStrides);
                result[i] = op(aVals[aIndex], bVals[bIndex]);
            }
        }
        return [result, newShape];
    };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const addImpl = createSimpleBinaryKernelImpl(((a, b) => a + b));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function bincountImpl(xVals, weightsVals, weightsDtype, weightsShape, size) {
    const weightsSize = util.sizeFromShape(weightsShape);
    const outVals = util.makeZerosTypedArray(size, weightsDtype);
    for (let i = 0; i < xVals.length; i++) {
        const value = xVals[i];
        if (value < 0) {
            throw new Error('Input x must be non-negative!');
        }
        if (value >= size) {
            continue;
        }
        if (weightsSize > 0) {
            outVals[value] += weightsVals[i];
        }
        else {
            outVals[value] += 1;
        }
    }
    return outVals;
}
function bincountReduceImpl(xBuf, weightsBuf, size, binaryOutput = false) {
    const numRows = xBuf.shape[0];
    const numCols = xBuf.shape[1];
    const outBuf = buffer([numRows, size], weightsBuf.dtype);
    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols; j++) {
            const value = xBuf.get(i, j);
            if (value < 0) {
                throw new Error('Input x must be non-negative!');
            }
            if (value >= size) {
                continue;
            }
            if (binaryOutput) {
                outBuf.set(1, i, value);
            }
            else {
                if (weightsBuf.size > 0) {
                    outBuf.set(outBuf.get(i, value) + weightsBuf.get(i, j), i, value);
                }
                else {
                    outBuf.set(outBuf.get(i, value) + 1, i, value);
                }
            }
        }
    }
    return outBuf;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * Template that creates implementation for unary op.
 */
function createSimpleUnaryImpl(op) {
    return (values, dtype, attrs) => {
        const newValues = util.getTypedArrayFromDType(dtype, values.length);
        for (let i = 0; i < values.length; ++i) {
            newValues[i] = op(values[i], attrs);
        }
        return newValues;
    };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const ceilImpl = createSimpleUnaryImpl((xi) => Math.ceil(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function concatImpl(inputs, outShape, dtype, simplyConcat) {
    const outVals = util.getArrayFromDType(dtype, util.sizeFromShape(outShape));
    if (simplyConcat && dtype !== 'string') {
        // Use built-in TypedArray.set() method for speed.
        let offset = 0;
        inputs.forEach(input => {
            const size = util.sizeFromShape(input.shape);
            outVals.set(input.vals, offset);
            offset += size;
        });
    }
    else {
        let colOffset = 0;
        inputs.forEach(input => {
            const decodedData = dtype === 'string' ?
                backend_util.fromUint8ToStringArray(input.vals) :
                input.vals;
            let tIdx = 0;
            for (let row = 0; row < input.shape[0]; ++row) {
                const resIdx = row * outShape[1] + colOffset;
                for (let col = 0; col < input.shape[1]; ++col) {
                    outVals[resIdx + col] = decodedData[tIdx++];
                }
            }
            colOffset += input.shape[1];
        });
    }
    return outVals;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const equalImpl = createSimpleBinaryKernelImpl((a, b) => (a === b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const expImpl = createSimpleUnaryImpl((xi) => Math.exp(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const expm1Impl = createSimpleUnaryImpl((xi) => Math.expm1(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const floorImpl = createSimpleUnaryImpl((xi) => Math.floor(xi));

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function gatherNdImpl(indicesData, paramsBuf, dtype, numSlices, sliceRank, sliceSize, strides, paramsShape, paramsSize) {
    const outBuf = buffer([numSlices, sliceSize], dtype);
    for (let i = 0; i < numSlices; i++) {
        const index = [];
        let flattenIndex = 0;
        for (let j = 0; j < sliceRank; j++) {
            const dim = indicesData[i * sliceRank + j];
            flattenIndex += dim * strides[j];
            index.push(dim);
        }
        if (flattenIndex < 0 || flattenIndex >= paramsSize / sliceSize) {
            throw new Error(`Invalid indices: ${index} does not index into ${paramsShape}`);
        }
        for (let k = 0; k < sliceSize; k++) {
            outBuf.values[i * sliceSize + k] =
                paramsBuf.get(...paramsBuf.indexToLoc(flattenIndex * sliceSize + k));
        }
    }
    return outBuf;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function gatherV2Impl(xBuf, indicesBuf, flattenOutputShape) {
    const outBuf = buffer(flattenOutputShape, xBuf.dtype);
    for (let i = 0; i < outBuf.size; ++i) {
        const newLoc = outBuf.indexToLoc(i);
        const originalLoc = newLoc.slice();
        const batchIdx = originalLoc[0];
        const indicesIdx = originalLoc[2];
        const indicesIndex = indicesBuf.locToIndex([batchIdx, indicesIdx]);
        originalLoc[2] = indicesBuf.values[indicesIndex];
        const originalIndex = xBuf.locToIndex(originalLoc);
        outBuf.values[i] = xBuf.values[originalIndex];
    }
    return outBuf;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const greaterImpl = createSimpleBinaryKernelImpl((a, b) => (a > b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const greaterEqualImpl = createSimpleBinaryKernelImpl((a, b) => (a >= b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const lessImpl = createSimpleBinaryKernelImpl((a, b) => (a < b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const lessEqualImpl = createSimpleBinaryKernelImpl((a, b) => (a <= b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function linSpaceImpl(start, stop, num) {
    const step = (stop - start) / (num - 1);
    const values = util.makeZerosTypedArray(num, 'float32');
    values[0] = start;
    for (let i = 1; i < values.length; i++) {
        values[i] = values[i - 1] + step;
    }
    return values;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const logImpl = createSimpleUnaryImpl((xi) => Math.log(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function maxImpl(aVals, reduceSize, outShape, dtype) {
    const vals = util.getTypedArrayFromDType(dtype, util.sizeFromShape(outShape));
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let max = aVals[offset];
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            if (Number.isNaN(value) ||
                value > max) { // comparison with NaN always return false
                max = value;
            }
        }
        vals[i] = max;
    }
    return vals;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const maximumImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => Math.max(aValue, bValue)));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const minimumImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => Math.min(aValue, bValue)));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const multiplyImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => aValue * bValue));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function negImpl(xVals, xShape, xDtype) {
    const minusOne = util.createScalarValue(-1, xDtype);
    return multiplyImpl([], xShape, minusOne, xVals, xDtype);
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const notEqualImpl = createSimpleBinaryKernelImpl(((a, b) => (a !== b) ? 1 : 0));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function transposeImpl(xVals, xShape, dtype, perm, newShape) {
    const xRank = xShape.length;
    const xSize = util.sizeFromShape(xShape);
    const xStrides = util.computeStrides(xShape);
    const newStrides = util.computeStrides(newShape);
    const result = util.getTypedArrayFromDType(dtype, util.sizeFromShape(newShape));
    for (let i = 0; i < xSize; ++i) {
        const loc = util.indexToLoc(i, xRank, xStrides);
        // Permute location.
        const newLoc = new Array(loc.length);
        for (let i = 0; i < newLoc.length; i++) {
            newLoc[i] = loc[perm[i]];
        }
        const newIndex = util.locToIndex(newLoc, xRank, newStrides);
        result[newIndex] = xVals[i];
    }
    return result;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function prodImpl(xShape, xDtype, xVals, reductionAxes) {
    const [outShape, reduceShape] = backend_util.computeOutAndReduceShapes(xShape, reductionAxes);
    const outDtype = upcastType(xDtype, 'int32');
    const outVals = util.makeZerosTypedArray(util.sizeFromShape(outShape), outDtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    for (let i = 0; i < outVals.length; ++i) {
        const offset = i * reduceSize;
        let prod = 1;
        for (let j = 0; j < reduceSize; ++j) {
            prod *= xVals[offset + j];
        }
        outVals[i] = prod;
    }
    return { outVals, outShape, outDtype };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function rangeImpl(start, stop, step, dtype) {
    const sameStartStop = start === stop;
    const increasingRangeNegativeStep = start < stop && step < 0;
    const decreasingRangePositiveStep = stop < start && step > 1;
    if (sameStartStop || increasingRangeNegativeStep ||
        decreasingRangePositiveStep) {
        return util.makeZerosTypedArray(0, dtype);
    }
    const numElements = Math.abs(Math.ceil((stop - start) / step));
    const values = util.makeZerosTypedArray(numElements, dtype);
    if (stop < start && step === 1) {
        // Auto adjust the step's sign if it hasn't been set
        // (or was set to 1)
        step = -1;
    }
    values[0] = start;
    for (let i = 1; i < values.length; i++) {
        values[i] = values[i - 1] + step;
    }
    return values;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const rsqrtImpl = createSimpleUnaryImpl((xi) => 1 / Math.sqrt(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sigmoidImpl = createSimpleUnaryImpl((xi) => 1 / (1 + Math.exp(-xi)));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function sliceImpl(vals, begin, size, shape, dtype) {
    const isContinous = slice_util.isSliceContinous(shape, begin, size);
    const length = util.sizeFromShape(size);
    const xStrides = util.computeStrides(shape);
    if (isContinous) {
        const flatOffset = slice_util.computeFlatOffset(begin, xStrides);
        if (dtype === 'string') {
            return vals.slice(flatOffset, flatOffset + length);
        }
        return vals.subarray(flatOffset, flatOffset + length);
    }
    const decodedData = dtype === 'string' ?
        backend_util.fromUint8ToStringArray(vals) :
        vals;
    const inBuf = buffer(shape, dtype, decodedData);
    const outBuf = buffer(size, dtype);
    for (let i = 0; i < outBuf.size; ++i) {
        const outLoc = outBuf.indexToLoc(i);
        const inLoc = outLoc.map((idx, j) => idx + begin[j]);
        outBuf.set(inBuf.get(...inLoc), ...outLoc);
    }
    if (dtype === 'string') {
        return backend_util.fromStringArrayToUint8(outBuf.values);
    }
    return outBuf.values;
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function sparseFillEmptyRowsImpl(indices, indicesShape, indicesDType, values, valuesDType, denseShape, defaultValue) {
    const indicesCount = indicesShape[0];
    const denseRows = denseShape[0];
    const emptyRowIndicator = new Array(denseRows);
    const reverseIndexMap = new Array(indicesCount);
    const rank = indicesShape[1];
    if (denseRows === 0) {
        if (indicesCount !== 0) {
            throw new Error(`Received SparseTensor with denseShape[0] = 0 but
         indices.shape[0] = ${indicesCount}`);
        }
        const outputIndices = util.getArrayFromDType(indicesDType, 0);
        const outputValues = util.getArrayFromDType(valuesDType, 0);
        return [
            outputIndices, [0, rank], outputValues, emptyRowIndicator, reverseIndexMap
        ];
    }
    let rowsAreOrdered = true;
    let lastIndicesRow = 0;
    const csrOffset = new Array(denseRows).fill(0);
    for (let i = 0; i < indicesCount; ++i) {
        // indices is a 2d tensor with shape of [N, rank]
        const row = indices[i * rank];
        if (row < 0) {
            throw new Error(`indices(${i}, 0) is invalid: ${row} < 0`);
        }
        if (row >= denseRows) {
            throw new Error(`indices(${i}, 0) is invalid: ${row} >= ${denseRows}`);
        }
        ++csrOffset[row];
        rowsAreOrdered = rowsAreOrdered && (row >= lastIndicesRow);
        lastIndicesRow = row;
    }
    let allRowsFull = true;
    for (let row = 0; row < denseRows; ++row) {
        // csrOffset here describes the number of elements in this dense row
        const rowEmpty = (csrOffset[row] === 0);
        emptyRowIndicator[row] = rowEmpty;
        allRowsFull = allRowsFull && !rowEmpty;
        // In filled version, each row has at least one element.
        csrOffset[row] = Math.max(csrOffset[row], 1);
        // Update csrOffset to represent the number of elements up to and
        // including denseRows + 1:
        //  csrOffset[0] == #{elements of row 0}
        //  csrOffset[1] == #{elements of row 1} + #{elements of row 0}
        //  ..
        //  csrOffset[i] == starting index for elements in row i + 1.
        if (row > 0) {
            csrOffset[row] += csrOffset[row - 1];
        }
    }
    if (allRowsFull && rowsAreOrdered) {
        const outputIndices = indices;
        const outputValues = values;
        for (let i = 0; i < indicesCount; ++i) {
            reverseIndexMap[i] = i;
        }
        return [
            outputIndices, [indicesCount, rank], outputValues, emptyRowIndicator,
            reverseIndexMap
        ];
    }
    else {
        const fullIndicesCount = csrOffset[denseRows - 1];
        const outputIndices = util.getArrayFromDType(indicesDType, fullIndicesCount * rank);
        const outputValues = util.getArrayFromDType(valuesDType, fullIndicesCount);
        const filledCount = new Array(denseRows).fill(0);
        // Fill in values for rows that are not missing
        for (let i = 0; i < indicesCount; ++i) {
            // indices is a 2d tensor with shape of [N, rank]
            const row = indices[i * rank];
            const offset = filledCount[row];
            const outputI = ((row === 0) ? 0 : csrOffset[row - 1]) + offset;
            filledCount[row]++; // Increment the filled count for this row.
            for (let j = 0; j < rank; ++j) {
                // indices and outputIndices are 2d tensors with shape of [N, rank]
                outputIndices[outputI * rank + j] = indices[i * rank + j];
            }
            outputValues[outputI] = values[i];
            // We'll need this reverse index map to backprop correctly.
            reverseIndexMap[i] = outputI;
        }
        // Fill in values for rows that are missing
        for (let row = 0; row < denseRows; ++row) {
            const rowCount = filledCount[row];
            if (rowCount === 0) { // We haven't filled this row
                const startingIndex = (row === 0) ? 0 : csrOffset[row - 1];
                // Remaining index values were set to zero already.
                // Just need to set the row index in the right location.
                // outputIndices is a 2d tensor with shape of [N, rank]
                outputIndices[startingIndex * rank + 0] = row;
                for (let col = 1; col < rank; ++col) {
                    outputIndices[startingIndex * rank + col] = 0;
                }
                outputValues[startingIndex] = defaultValue;
            }
        }
        return [
            outputIndices, [fullIndicesCount, rank], outputValues, emptyRowIndicator,
            reverseIndexMap
        ];
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function sparseReshapeImpl(inputIndices, inputIndicesShape, inputDType, inputShape, targetShape) {
    const denseSize = util.sizeFromShape(inputShape);
    const nnz = inputIndicesShape[0];
    const outputRank = targetShape.length;
    // Compute the output shape. Determine product of specified dimensions, and
    // find the index of the unspecified one.
    const outputShape = [];
    let product = 1;
    let unknownIndex = -1;
    for (let d = 0; d < outputRank; ++d) {
        const size = targetShape[d];
        if (size === -1) {
            if (unknownIndex !== -1) {
                throw new Error(`only one output dimension may be -1, not both ${unknownIndex} and ${d}`);
            }
            unknownIndex = d;
            outputShape.push(1);
        }
        else {
            if (size < 0) {
                throw new Error(`size ${d} must be non-negative, not ${size}`);
            }
            product *= size;
            outputShape.push(size);
        }
    }
    if (unknownIndex !== -1) {
        if (product <= 0) {
            throw new Error('reshape cannot infer the missing ' +
                'input size for an empty tensor unless all ' +
                'specified input sizes are non-zero');
        }
        const missing = Math.trunc(denseSize / product);
        if (product * missing !== denseSize) {
            throw new Error(`Input to reshape is a SparseTensor with ${denseSize}
          dense values, but the requested shape requires a multiple of ${product}. inputShape=${inputShape} outputShape= ${outputShape}`);
        }
        outputShape[unknownIndex] = missing;
    }
    const outputSize = util.sizeFromShape(outputShape);
    if (outputSize !== denseSize) {
        throw new Error(`Input to reshape is a tensor with ${denseSize} dense values, but the requested shape has ${outputSize}. inputShape=${inputShape} outputShape=${outputShape}`);
    }
    const inputRank = inputShape.length;
    const inputStrides = [];
    if (inputRank > 0) {
        inputStrides[inputRank - 1] = 1;
        for (let d = inputRank - 2; d >= 0; --d) {
            inputStrides[d] = inputStrides[d + 1] * inputShape[d + 1];
        }
    }
    const outputStrides = [];
    if (outputRank > 0) {
        outputStrides[outputRank - 1] = 1;
        for (let d = outputRank - 2; d >= 0; --d) {
            outputStrides[d] = outputStrides[d + 1] * outputShape[d + 1];
        }
    }
    const newIndices = util.getArrayFromDType(inputDType, nnz * outputRank);
    for (let i = 0; i < nnz; ++i) {
        let id = 0;
        for (let j = 0; j < inputRank; ++j) {
            // inputIndices is a 2d tensor with shape of [nnz, inputRank]
            id += inputIndices[i * inputRank + j] * inputStrides[j];
        }
        for (let j = 0; j < outputRank; ++j) {
            // newIndices is a 2d tensor with shape of [nnz, outputRank]
            newIndices[i * outputRank + j] = Math.trunc(id / outputStrides[j]);
            id %= outputStrides[j];
        }
    }
    return [newIndices, [nnz, outputRank], outputShape];
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function sparseSegmentReductionImpl(input, inputShape, inputDType, indices, segmentIds, isMean = false, defaultValue = 0) {
    const numIndices = indices.length;
    if (numIndices !== segmentIds.length) {
        throw new Error(`segmentIds and indices should have same size.`);
    }
    // Flatten the array to two dimensions
    const inputFlat = [inputShape[0], input.length / inputShape[0]];
    const numCol = inputFlat[1];
    // Note that the current implementation assumes that segmentIds values are
    // sorted.
    const lastSegmentIdPlusOne = numIndices > 0 ? segmentIds[numIndices - 1] + 1 : 0;
    const outputRows = lastSegmentIdPlusOne;
    if (outputRows < 0) {
        throw new Error(`segment ids must be >= 0`);
    }
    const outputShape = inputShape.slice();
    outputShape[0] = outputRows;
    const outputLength = outputShape.reduce((product, value) => product * value, 1);
    // Output array is initialized with the value 0 by default.
    const output = util.getArrayFromDType(inputDType, outputLength);
    // Note that we do not initialize the output buffer with a default value, so
    // we need to explicitly set missing indices to the default value.
    if (numIndices === 0) {
        if (outputRows > 0) {
            output.fill(defaultValue);
        }
        return [output, outputShape];
    }
    if (outputRows <= 0) {
        throw new Error(`segment ids must be >= 0`);
    }
    let start = 0, end = 1;
    // Index from which the output is not initialized.
    let uninitializedIndex = 0;
    let outIndex = segmentIds[start];
    while (true) {
        // We initialize nextIndex to 0 to avoid may be uninitialized warning
        let nextIndex = 0;
        if (end < numIndices) {
            nextIndex = segmentIds[end];
            if (outIndex === nextIndex) {
                ++end;
                continue;
            }
            // We have a new segment here.  Verify that the segment ids are growing.
            if (outIndex >= nextIndex) {
                throw new Error(`segment ids are not increasing`);
            }
        }
        if (outIndex < 0 || outIndex >= outputRows) {
            throw new Error(`Segment id ${outIndex} out of range [0, ${outputRows}), possibly because segmentIds input is not sorted.`);
        }
        // If there is a gap between two indices, we need to set that gap to the
        // default value.
        if (outIndex > uninitializedIndex) {
            output.fill(defaultValue, uninitializedIndex * numCol, outIndex * numCol);
        }
        for (let i = start; i < end; ++i) {
            const index = indices[i];
            if (index < 0 || index >= inputFlat[0]) {
                throw new Error(`Bad: indices[${i}] == ${indices[i]} out of range [0, ${inputFlat[0]})`);
            }
            for (let j = 0; j < numCol; j++) {
                output[outIndex * numCol + j] += input[index * numCol + j];
            }
        }
        if (isMean) {
            for (let j = 0; j < numCol; j++) {
                output[outIndex * numCol + j] /= end - start;
            }
        }
        start = end;
        ++end;
        uninitializedIndex = outIndex + 1;
        outIndex = nextIndex;
        if (end > numIndices) {
            break;
        }
    }
    // Fill the gap at the end with the default value.
    if (uninitializedIndex < outputRows) {
        output.fill(defaultValue, uninitializedIndex * numCol, outputRows * numCol);
    }
    return [output, outputShape];
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sqrtImpl = createSimpleUnaryImpl((xi) => Math.sqrt(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const squaredDifferenceImpl = createSimpleBinaryKernelImpl(((a, b) => {
    const diff = a - b;
    return diff * diff;
}));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function stridedSliceImpl(outShape, xBuf, strides, begin) {
    const outBuf = buffer(outShape, xBuf.dtype);
    for (let i = 0; i < outBuf.size; i++) {
        const loc = outBuf.indexToLoc(i);
        const newLoc = new Array(loc.length);
        for (let j = 0; j < newLoc.length; j++) {
            newLoc[j] = loc[j] * strides[j] + begin[j];
        }
        outBuf.set(xBuf.get(...newLoc), ...loc);
    }
    return outBuf;
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * The StringNGramsOp class creates ngrams from ragged string data.
 * The constructor contains all attributes related to the operation such as
 * padding widths and strings, and the compute function can be used to
 * compute the ngrams for different ragged tensor inputs.
 */
class StringNGramsOp {
    constructor(separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences) {
        this.separator = util.encodeString(separator);
        this.nGramWidths = nGramWidths;
        this.leftPad = util.encodeString(leftPad);
        this.rightPad = util.encodeString(rightPad);
        this.padWidth = padWidth;
        this.preserveShort = preserveShortSequences;
    }
    getPadWidth(nGramWidth) {
        // Ngrams can be padded with either a fixed pad width or a dynamic pad
        // width depending on the 'padWidth' arg, but in no case should the padding
        // ever be wider than 'nGramWidth' - 1.
        return Math.min(this.padWidth < 0 ? nGramWidth - 1 : this.padWidth, nGramWidth - 1);
    }
    getNumNGrams(length, nGramWidth) {
        const padWidth = this.getPadWidth(nGramWidth);
        return Math.max(0, ((length + 2 * padWidth) - nGramWidth) + 1);
    }
    createNGrams(data, splitIndex, output, outputStartIndex, numNGrams, nGramWidth) {
        for (let nGramIndex = 0; nGramIndex < numNGrams; ++nGramIndex) {
            const padWidth = this.getPadWidth(nGramWidth);
            const leftPadding = Math.max(0, padWidth - nGramIndex);
            const rightPadding = Math.max(0, padWidth - (numNGrams - (nGramIndex + 1)));
            const numTokens = nGramWidth - (leftPadding + rightPadding);
            const dataStartIndex = splitIndex + (leftPadding > 0 ? 0 : nGramIndex - padWidth);
            // Calculate the total expected size of the nGram so we can reserve the
            // correct amount of space in the string.
            let nGramSize = 0;
            // Size of the left padding.
            nGramSize += leftPadding * this.leftPad.length;
            // Size of the tokens.
            for (let n = 0; n < numTokens; ++n) {
                nGramSize += data[dataStartIndex + n].length;
            }
            // Size of the right padding.
            nGramSize += rightPadding * this.rightPad.length;
            // Size of the separators.
            const numSeparators = leftPadding + rightPadding + numTokens - 1;
            nGramSize += numSeparators * this.separator.length;
            // Build the nGram.
            output[outputStartIndex + nGramIndex] = new Uint8Array(nGramSize);
            const nGram = output[outputStartIndex + nGramIndex];
            let nextNGramIndex = 0;
            const appendToNGram = (str) => str.forEach((value) => nGram[nextNGramIndex++] = value);
            for (let n = 0; n < leftPadding; ++n) {
                appendToNGram(this.leftPad);
                appendToNGram(this.separator);
            }
            // Only output first numTokens - 1 pairs of data and separator
            for (let n = 0; n < numTokens - 1; ++n) {
                appendToNGram(data[dataStartIndex + n]);
                appendToNGram(this.separator);
            }
            // Handle case when there are no tokens or no right padding as these
            // can result in consecutive separators.
            if (numTokens > 0) {
                // If we have tokens, then output last and then pair each separator
                // with the right padding that follows, to ensure nGram ends either with
                // the token or with the right pad.
                appendToNGram(data[dataStartIndex + numTokens - 1]);
                for (let n = 0; n < rightPadding; ++n) {
                    appendToNGram(this.separator);
                    appendToNGram(this.rightPad);
                }
            }
            else {
                // If we don't have tokens, then the last item inserted into the nGram
                // has been the separator from the left padding loop above. Hence,
                // output right pad and separator and make sure to finish with a
                // padding, not a separator.
                for (let n = 0; n < rightPadding - 1; ++n) {
                    appendToNGram(this.rightPad);
                    appendToNGram(this.separator);
                }
                appendToNGram(this.rightPad);
            }
        }
    }
    // Data and splits together form the definition of the ragged tensor,
    // where data is 1 dimensional and contains the values of the tensor
    // and splits denotes the indices at which each row starts.
    compute(data, splits) {
        // Validate that the splits are valid indices into data, only if there are
        // splits specified.
        const inputDataSize = data.length;
        const splitsSize = splits.length;
        if (splitsSize > 0) {
            let prevSplit = splits[0];
            if (prevSplit !== 0) {
                throw new Error(`First split value must be 0, got ${prevSplit}`);
            }
            for (let i = 1; i < splitsSize; ++i) {
                let validSplits = splits[i] >= prevSplit;
                validSplits = validSplits && (splits[i] <= inputDataSize);
                if (!validSplits) {
                    throw new Error(`Invalid split value ${splits[i]}, must be in [${prevSplit}, ${inputDataSize}]`);
                }
                prevSplit = splits[i];
            }
            if (prevSplit !== inputDataSize) {
                throw new Error(`Last split value must be data size. Expected ${inputDataSize}, got ${prevSplit}`);
            }
        }
        const numBatchItems = splitsSize - 1;
        const nGramsSplits = util.getArrayFromDType('int32', splitsSize);
        // If there is no data or size, return an empty ragged tensor.
        if (inputDataSize === 0 || splitsSize === 0) {
            const empty = new Array(inputDataSize);
            for (let i = 0; i <= numBatchItems; ++i) {
                nGramsSplits[i] = 0;
            }
            return [empty, nGramsSplits];
        }
        nGramsSplits[0] = 0;
        for (let i = 1; i <= numBatchItems; ++i) {
            const length = splits[i] - splits[i - 1];
            let numNGrams = 0;
            this.nGramWidths.forEach((nGramWidth) => {
                numNGrams += this.getNumNGrams(length, nGramWidth);
            });
            if (this.preserveShort && length > 0 && numNGrams === 0) {
                numNGrams = 1;
            }
            nGramsSplits[i] = nGramsSplits[i - 1] + numNGrams;
        }
        const nGrams = new Array(nGramsSplits[numBatchItems]);
        for (let i = 0; i < numBatchItems; ++i) {
            const splitIndex = splits[i];
            let outputStartIdx = nGramsSplits[i];
            this.nGramWidths.forEach((nGramWidth) => {
                const length = splits[i + 1] - splits[i];
                const numNGrams = this.getNumNGrams(length, nGramWidth);
                this.createNGrams(data, splitIndex, nGrams, outputStartIdx, numNGrams, nGramWidth);
                outputStartIdx += numNGrams;
            });
            // If we're preserving short sequences, check to see if no sequence was
            // generated by comparing the current output start idx to the original
            // one (nGramSplitsdata). If no ngrams were generated, then they will
            // be equal (since we increment outputStartIdx by numNGrams every
            // time we create a set of ngrams.)
            if (this.preserveShort && outputStartIdx === nGramsSplits[i]) {
                const dataLength = splits[i + 1] - splits[i];
                // One legitimate reason to not have any ngrams when this.preserveShort
                // is true is if the sequence itself is empty. In that case, move on.
                if (dataLength === 0) {
                    continue;
                }
                // We don't have to worry about dynamic padding sizes here: if padding
                // was dynamic, every sequence would have had sufficient padding to
                // generate at least one nGram.
                const nGramWidth = dataLength + 2 * this.padWidth;
                const numNGrams = 1;
                this.createNGrams(data, splitIndex, nGrams, outputStartIdx, numNGrams, nGramWidth);
            }
        }
        return [nGrams, nGramsSplits];
    }
}
function stringNGramsImpl(data, dataSplits, separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences) {
    return new StringNGramsOp(separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences)
        .compute(data, dataSplits);
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function split(str, delimiters, skipEmpty, result) {
    if (!str.length) {
        return;
    }
    // When the delimiter is empty, the input is split into individual characters.
    if (delimiters.length === 0) {
        for (let i = 0; i < str.length; ++i) {
            result.push(str.subarray(i, i + 1));
        }
        return;
    }
    // When there is one delimiter, the input is split only at that delimiter.
    if (delimiters.length === 1) {
        const delimiter = delimiters[0];
        let f = str.indexOf(delimiter);
        while (f !== -1) {
            const token = str.subarray(0, f);
            if (!skipEmpty || token.length !== 0) {
                result.push(token);
            }
            str = str.subarray(f + 1);
            f = str.indexOf(delimiter);
        }
        if (!skipEmpty || str.length !== 0) {
            result.push(str);
        }
        return;
    }
    // When there are multiple delimiters, the input is split at every instance
    // one of the delimiters appears.
    let tokenStart = 0;
    for (let i = 0; i < str.length + 1; i++) {
        if ((i === str.length) || (delimiters.indexOf(str[i]) !== -1)) {
            const token = str.subarray(tokenStart, i);
            if (!skipEmpty || token.length !== 0) {
                result.push(token);
            }
            tokenStart = i + 1;
        }
    }
}
function stringSplitImpl(input, delimiter, skipEmpty) {
    const batchSize = input.length;
    // Empty delimiter means split the input character by character.
    const tokens = [];
    let outputSize = 0;
    let maxNumEntries = 0;
    const numIndices = new Array(batchSize);
    for (let i = 0; i < batchSize; ++i) {
        const prevTokensLength = tokens.length;
        split(input[i], delimiter, skipEmpty, tokens);
        const nEntries = tokens.length - prevTokensLength;
        numIndices[i] = nEntries;
        outputSize += nEntries;
        maxNumEntries = Math.max(maxNumEntries, nEntries);
    }
    const indices = util.getArrayFromDType('int32', outputSize * 2);
    const values = new Array(outputSize);
    const shape = [batchSize, maxNumEntries];
    let c = 0;
    for (let i = 0; i < batchSize; ++i) {
        for (let j = 0; j < numIndices[i]; ++j) {
            // indices is a 2d tensor with shape of [outputSize, 2]
            indices[c * 2] = i;
            indices[c * 2 + 1] = j;
            values[c] = tokens[c];
            ++c;
        }
    }
    return [indices, values, shape];
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function stringToHashBucketFastImpl(input, numBuckets) {
    const output = util.getArrayFromDType('int32', input.length);
    for (let i = 0; i < input.length; ++i) {
        output[i] =
            util.fingerPrint64(input[i]).modulo(numBuckets).getLowBitsUnsigned();
    }
    return output;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const subImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => aValue - bValue));

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * An implementation of the tile kernel shared between webgl and cpu for string
 * tensors only.
 */
function tileImpl(xBuf, reps) {
    const newShape = new Array(xBuf.rank);
    for (let i = 0; i < newShape.length; i++) {
        newShape[i] = xBuf.shape[i] * reps[i];
    }
    const result = buffer(newShape, xBuf.dtype);
    for (let i = 0; i < result.values.length; ++i) {
        const newLoc = result.indexToLoc(i);
        const originalLoc = new Array(xBuf.rank);
        for (let j = 0; j < originalLoc.length; j++) {
            originalLoc[j] = newLoc[j] % xBuf.shape[j];
        }
        const originalIndex = xBuf.locToIndex(originalLoc);
        result.values[i] = xBuf.values[originalIndex];
    }
    return result;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const comparePair = (a, b) => {
    const valueDiff = b.value - a.value;
    return valueDiff === 0 ? a.index - b.index : valueDiff;
};
/**
 * Partitions array where all elements smaller than the (k+1) smallest element
 * are found to the left of it, and all larger to the right of it.
 * Based on the Floyd-Rivest Algorithm, ref:
 * https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm
 * @param array: Array to partition
 * @param left: Left index for the interval
 * @param right: Right index for the interval
 * @param k: Desired index value, where array[k] is the (k+1)th smallest element
 *           when left = 0
 */
function select(array, k, left = 0, right = array.length - 1) {
    while (right > left) {
        // Use select recursively to sample a smaller set of size s
        // the arbitrary constants 600 and 0.5 are used in the original
        // version to minimize execution time.
        if (right - left > 600) {
            const n = right - left + 1;
            const i = k - left + 1;
            const z = Math.log(n);
            const s = 0.5 * Math.exp(2 * z / 3);
            const sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Math.sign(i - n / 2);
            const newLeft = Math.max(left, Math.floor(k - i * s / n + sd));
            const newRight = Math.min(right, Math.floor(k + (n - i) * s / n + sd));
            select(array, k, newLeft, newRight);
        }
        // partition the elements between left and right around t
        const t = array[k];
        let i = left;
        let j = right;
        util.swap(array, left, k);
        if (comparePair(array[right], t) > 0) {
            util.swap(array, left, right);
        }
        while (i < j) {
            util.swap(array, i, j);
            i++;
            j--;
            while (comparePair(array[i], t) < 0) {
                i = i + 1;
            }
            while (comparePair(array[j], t) > 0) {
                j = j - 1;
            }
        }
        if (comparePair(array[left], t) === 0) {
            util.swap(array, left, j);
        }
        else {
            j = j + 1;
            util.swap(array, j, right);
        }
        // Adjust left and right towards the boundaries of the subset
        // containing the (k - left + 1)th smallest element.
        if (j <= k) {
            left = j + 1;
        }
        if (k <= j) {
            right = j - 1;
        }
    }
}
function topKImpl(x, xShape, xDtype, k, sorted) {
    // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
    const lastDim = xShape[xShape.length - 1];
    const [batch, size] = [x.length / lastDim, lastDim];
    const allTopKVals = util.getTypedArrayFromDType(xDtype, batch * k);
    const allTopKIndices = util.getTypedArrayFromDType('int32', batch * k);
    for (let b = 0; b < batch; b++) {
        const offset = b * size;
        const vals = x.subarray(offset, offset + size);
        let valAndInd = new Array(vals.length);
        vals.forEach((value, index) => valAndInd[index] = { value, index });
        if (k < valAndInd.length) {
            select(valAndInd, k);
            valAndInd = valAndInd.slice(0, k);
        }
        if (sorted) {
            valAndInd.sort(comparePair);
        }
        const outOffset = b * k;
        const topKVals = allTopKVals.subarray(outOffset, outOffset + k);
        const topKIndices = allTopKIndices.subarray(outOffset, outOffset + k);
        for (let i = 0; i < k; i++) {
            topKVals[i] = valAndInd[i].value;
            topKIndices[i] = valAndInd[i].index;
        }
    }
    // Reshape back to the original input shape, except that the last
    // dimension is k.
    const outputShape = xShape.slice();
    outputShape[outputShape.length - 1] = k;
    return [
        buffer(outputShape, xDtype, allTopKVals),
        buffer(outputShape, 'int32', allTopKIndices)
    ];
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function uniqueImpl(values, axis, shape, dtype) {
    // Normalize and validate axis.
    const $axis = util.parseAxisParam(axis, shape)[0];
    // Calculate the new shape that is suitable for extracting data along the
    // given axis.
    //
    // The rank is 3.
    // The size of the 1st dimension is the size of all the axes < the given axis.
    // The size of the 2nd dimension is the same as the size of the given axis.
    // The size of the 3rd dimension is the size of all the axes > the given axis.
    //
    // For example, for a 4D tensor with shape=[2, 3, 5, 4] and axis=2, the
    // newShape would be: [2*3, 5, 4].
    //
    // Note that this is not the final output shape. This will be the shape for an
    // intermediate TensorBuffer (see inputBuffer below) to allow us to extract
    // values along the given axis. To demonstrate how it works, consider the
    // following example:
    //
    // Input: a 3D tensor, with shape [1, 2, 3]
    // [
    //   [
    //      [1,2,3],
    //      [4,5,6]
    //   ]
    // ]
    // Axis: 2 (the last axis).
    // Along axis 2, we expect to extract 3 tensors: [1,4], [2,5], [3,6].
    //
    // For this example, newShape would be: [2, 3, 1], where 2 is calculated from
    // 1*2. The re-shaped data would look like:
    //
    // [
    //   [
    //     [1], [2], [3]
    //   ],
    //   [
    //     [4], [5], [6]
    //   ]
    // ]
    //
    // Then, we can construct a 3-level nested loop by the following dimension
    // order to extract the values along the axis (dimension1):
    // i: dimension1       // 0,1,2 (newShape[1])
    //   m: dimension0     // 0,1   (newShape[0])
    //     n: dimension2   // 0     (newShape[2])
    //
    //                       m, i, n
    //                      ---------
    // Iteration 0: data at [0, 0, 0] => "1"
    // Iteration 1: data at [1, 0, 0] => "4"
    // We got [1,4].
    // Iteration 2: data at [0, 1, 0] => "2"
    // Iteration 3: data at [1, 1, 0] => "5"
    // We got [2,5].
    // Iteration 4: data at [0, 2, 0] => "3"
    // Iteration 5: data at [1, 2, 0] => "6"
    // We got [3,6].
    const newShape = [1, shape[0], 1];
    for (let i = 0; i < $axis; i++) {
        newShape[0] *= shape[i];
    }
    newShape[1] = shape[$axis];
    for (let i = $axis + 1; i < shape.length; i++) {
        newShape[2] *= shape[i];
    }
    // A map from unique elements (their string representations) to their values
    // in "indices" (below).
    const uniqueElements = {};
    // The indices of each unique element in the original tensor along the given
    // axis. It is 1D and has the same size as the given axis.
    const indices = new Int32Array(shape[$axis]);
    // Create a buffer so we can easily extract value at a given location.
    const inputBuffer = new TensorBuffer(newShape, dtype, values);
    // The indices along the given axis that have unique elements. This is a
    // de-duped version of "indices" above.
    const uniqueIndices = [];
    const is1DTensor = newShape[0] === 1 && newShape[2] === 1;
    for (let i = 0; i < shape[$axis]; i++) {
        // Extract values along the axis.
        let element;
        if (is1DTensor) {
            // Fast path for 1D tensor input.
            element = values[i].toString();
        }
        else {
            const axisValues = [];
            for (let m = 0; m < newShape[0]; m++) {
                for (let n = 0; n < newShape[2]; n++) {
                    axisValues.push(inputBuffer.get(m, i, n));
                }
            }
            element = axisValues.join(',');
        }
        // Dedup and update various indices.
        if (uniqueElements[element] !== undefined) {
            indices[i] = uniqueElements[element];
        }
        else {
            const uniqueIndex = Object.keys(uniqueElements).length;
            uniqueElements[element] = uniqueIndex;
            indices[i] = uniqueIndex;
            uniqueIndices.push(i);
        }
    }
    // Now we know where each of the unique elements are located along the axis
    // (uniqueIndices). Extract them from input buffer and store them in the
    // output buffer.
    const outputTmpShape = newShape.slice();
    outputTmpShape[1] = Object.keys(uniqueElements).length;
    const outputBuffer = new TensorBuffer(outputTmpShape, dtype);
    uniqueIndices.forEach((uniqueElementIndex, i) => {
        for (let m = 0; m < newShape[0]; m++) {
            for (let n = 0; n < newShape[2]; n++) {
                outputBuffer.set(inputBuffer.get(m, uniqueElementIndex, n), m, i, n);
            }
        }
    });
    // The output shape can be calculated from the input shape with the size of
    // the given axis replaced by the number of unique elements along that axis.
    const outputShape = shape.slice();
    outputShape[$axis] = outputTmpShape[1];
    return {
        outputValues: outputBuffer.values,
        outputShape,
        indices,
    };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

var shared = /*#__PURE__*/Object.freeze({
  __proto__: null,
  simpleAbsImpl: simpleAbsImpl,
  addImpl: addImpl,
  bincountImpl: bincountImpl,
  bincountReduceImpl: bincountReduceImpl,
  ceilImpl: ceilImpl,
  concatImpl: concatImpl,
  equalImpl: equalImpl,
  expImpl: expImpl,
  expm1Impl: expm1Impl,
  floorImpl: floorImpl,
  gatherNdImpl: gatherNdImpl,
  gatherV2Impl: gatherV2Impl,
  greaterImpl: greaterImpl,
  greaterEqualImpl: greaterEqualImpl,
  lessImpl: lessImpl,
  lessEqualImpl: lessEqualImpl,
  linSpaceImpl: linSpaceImpl,
  logImpl: logImpl,
  maxImpl: maxImpl,
  maximumImpl: maximumImpl,
  minimumImpl: minimumImpl,
  multiplyImpl: multiplyImpl,
  negImpl: negImpl,
  notEqualImpl: notEqualImpl,
  prodImpl: prodImpl,
  rangeImpl: rangeImpl,
  rsqrtImpl: rsqrtImpl,
  sigmoidImpl: sigmoidImpl,
  sliceImpl: sliceImpl,
  sparseFillEmptyRowsImpl: sparseFillEmptyRowsImpl,
  sparseReshapeImpl: sparseReshapeImpl,
  sparseSegmentReductionImpl: sparseSegmentReductionImpl,
  sqrtImpl: sqrtImpl,
  squaredDifferenceImpl: squaredDifferenceImpl,
  stridedSliceImpl: stridedSliceImpl,
  stringNGramsImpl: stringNGramsImpl,
  stringSplitImpl: stringSplitImpl,
  stringToHashBucketFastImpl: stringToHashBucketFastImpl,
  subImpl: subImpl,
  tileImpl: tileImpl,
  topKImpl: topKImpl,
  transposeImpl: transposeImpl,
  uniqueImpl: uniqueImpl
});

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const { addImpl: addImplCPU, ceilImpl: ceilImplCPU, concatImpl: concatImplCPU, equalImpl: equalImplCPU, expImpl: expImplCPU, expm1Impl: expm1ImplCPU, floorImpl: floorImplCPU, gatherNdImpl: gatherNdImplCPU, gatherV2Impl: gatherV2ImplCPU, greaterEqualImpl: greaterEqualImplCPU, greaterImpl: greaterImplCPU, lessEqualImpl: lessEqualImplCPU, lessImpl: lessImplCPU, logImpl: logImplCPU, maxImpl: maxImplCPU, maximumImpl: maximumImplCPU, minimumImpl: minimumImplCPU, multiplyImpl: multiplyImplCPU, negImpl: negImplCPU, notEqualImpl: notEqualImplCPU, prodImpl: prodImplCPU, rangeImpl: rangeImplCPU, rsqrtImpl: rsqrtImplCPU, simpleAbsImpl: simpleAbsImplCPU, sliceImpl: sliceImplCPU, stridedSliceImpl: stridedSliceImplCPU, stringNGramsImpl: stringNGramsImplCPU, subImpl: subImplCPU, tileImpl: tileImplCPU, transposeImpl: transposeImplCPU, uniqueImpl: uniqueImplCPU, } = shared;

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const abs = unaryKernelFunc({ opType: UnaryOpType.ABS, cpuKernelImpl: simpleAbsImplCPU });
const absConfig = {
    kernelName: Abs,
    backendName: 'webgpu',
    kernelFunc: abs
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const addKernelFunc = binaryKernelFunc({
    opSnippet: BinaryOpType.ADD,
    cpuKernelImpl: addImplCPU,
    supportsComplex: true
});
const addConfig = {
    kernelName: Add,
    backendName: 'webgpu',
    kernelFunc: addKernelFunc
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class AddNPackedProgram {
    constructor(shapes) {
        this.workPerThread = 4;
        this.workGroupSize = [64, 1, 1];
        this.outputShape = shapes[0];
        this.variableNames = shapes.map((_, i) => `T${i}`);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.shaderKey = 'addN';
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const snippets = [];
        // Get target elements from every input tensor.
        this.variableNames.forEach(variable => {
            snippets.push(`float v${variable} = get${variable}AtOutCoords(coords);`);
        });
        // Calculate the sum of all elements.
        const operation = this.variableNames
            .map(variable => {
            return `v${variable}`;
        })
            .join(' + ');
        const type = getCoordsDataType(this.outputShape.length);
        const userCode = `
      void main() {
        int index = getGlobalIndex();
        for (int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;
          if (flatIndex < size) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);
            ${snippets.join('\n        ')}
            setOutput(flatIndex, ${operation});
          }
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const snippets = [];
        // Get target elements from every input tensor.
        this.variableNames.forEach(variable => {
            snippets.push(`let v${variable} = get${variable}AtOutCoordsByCoords(coords);`);
        });
        // Calculate the sum of all elements.
        const operation = this.variableNames
            .map(variable => {
            return `v${variable}`;
        })
            .join(' + ');
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        for (var i = 0u; i < ${this.workPerThread}u; i = i + 1u) {
          let flatIndex = index * ${this.workPerThread}u + i;
          if (flatIndex < uniforms.size) {
            let coords = getCoordsFromFlatIndex(flatIndex);
            ${snippets.join('\n        ')}
            setOutputFlat(flatIndex, ${operation});
          }
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function addN(args) {
    const { inputs, backend } = args;
    const tensors = inputs;
    if (tensors.length === 1) {
        return identity({ inputs: { x: tensors[0] }, backend });
    }
    const dtype = tensors.map(t => t.dtype).reduce((d1, d2) => upcastType(d1, d2));
    const shapes = tensors.map(t => t.shape);
    const program = new AddNPackedProgram(shapes);
    return backend.runWebGPUProgram(program, tensors, dtype);
}
const addNConfig = {
    kernelName: AddN,
    backendName: 'webgpu',
    kernelFunc: addN
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class ArgMinMaxProgram {
    constructor(inputShape, axis, reduceType) {
        this.variableNames = ['x'];
        this.uniforms = 'int axis;';
        this.uniformsWgsl = 'axis : u32;';
        const axes = [axis];
        backend_util.assertAxesAreInnerMostDims('arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes, inputShape.length);
        this.op = reduceType === 'min' ? '<' : '>';
        // |outShape| is the shape with the removed axis
        // |reduceShape| is the shape we are reducing. i.e. [ inputShape[axis] ]
        const [outputShape, reduceShape] = backend_util.computeOutAndReduceShapes(inputShape, axes);
        this.outputShape = outputShape.length === 0 ? [1] : outputShape;
        // Length of the axis we're reducing on.
        const reduceSize = util.sizeFromShape(reduceShape);
        // The number of comparisons each thread will do
        this.reductionFactor = 2;
        // Note that the maximum of workgroup X dimension is 256.
        const xMaxThreads = 256; // gl_MaxComputeWorkGroupSize.
        const xThreads = Math.min(Math.ceil(reduceSize / this.reductionFactor), xMaxThreads);
        this.workGroupSize = [xThreads, 1, 1];
        this.dispatchLayout = { x: [], y: this.outputShape.map((d, i) => i) };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.inputShape = inputShape;
        this.shaderKey = `argMinMax${this.op}`;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        // When this.workGroupSize[0] > 1, each thread reduces Length /
        // this.workGroupSize[0] values. Thes results are stored in shared memory
        // and iteratively reduced.
        const reduceInSharedMemory = this.workGroupSize[0] > 1;
        const sharedMemorySnippet = `
      shared int xBestIndices[WorkGroupSize];
      shared float xBestValues[WorkGroupSize];
    `;
        const sharedMemoryReduceSnippet = `
      xBestIndices[gl_LocalInvocationID.x] = bestIndex;
      xBestValues[gl_LocalInvocationID.x] = bestValue;

      int currentSize = WorkGroupSize;
      while (currentSize > 1) {
        barrier();

        for (int w = 0; w < ${this.reductionFactor}; ++w) {
          int i = int(gl_LocalInvocationID.x) * ${this.reductionFactor} + w;
          if (i < currentSize) {
            int candidateIndex = xBestIndices[i];
            float candidate = xBestValues[i];
            if (candidate ${this.op} bestValue && !isnan(candidate)) {
              bestValue = candidate;
              bestIndex = candidateIndex;
            }
          }
        }

        xBestIndices[gl_LocalInvocationID.x] = bestIndex;
        xBestValues[gl_LocalInvocationID.x] = bestValue;

        currentSize = DIV_CEIL(currentSize, ${this.reductionFactor});
      }

      if (gl_LocalInvocationID.x == 0) {
        setOutput(flatOutputIndex, int(bestIndex));
      }
    `;
        const outputCoordsType = getCoordsDataType(this.outputShape.length);
        const indexOutputCoords = (outputCoords, index) => {
            if (this.outputShape.length === 1) {
                return outputCoords;
            }
            else {
                return `${outputCoords}[${index}]`;
            }
        };
        const indexInputShape = (index) => {
            if (this.inputShape.length === 1) {
                return 'xShape';
            }
            else {
                return `xShape[${index}]`;
            }
        };
        const userCode = `
      #define DIV_CEIL(x, y) (((x) - 1) / (y) + 1)

      const int WorkGroupSize = int(gl_WorkGroupSize.x);

      ${reduceInSharedMemory ? sharedMemorySnippet : ''}

      // In order to get a flattened index into the input tensor, we need to
      // add back the index along the reduced dimension to |outputCoords|.
      // This function outputs the offset to the first value along
      // |axis| and the stride to get the next value of the input along |axis|.
      ivec2 getInputCoordInfo() {
        const ${outputCoordsType} outputCoords = getOutputCoords();
        int i = ${this.outputShape.length - 1};

        int stride = 1;
        int inputStride = 1;
        int offset = 0;

        for (int r = 1; r <= ${this.inputShape.length}; ++r) {
          int length = ${indexInputShape(`${this.inputShape.length} - r`)};
          if (${this.inputShape.length} - r == axis) {
            inputStride = stride;
          } else {
            offset += ${indexOutputCoords('outputCoords', 'i--')} * stride;
          }
          stride *= length;
        }

        return ivec2(offset, inputStride);
      }

      int getInputIndex(ivec2 coordInfo, int index) {
        return coordInfo[0] + coordInfo[1] * index;
      }

      void main() {
        const ivec2 coordInfo = getInputCoordInfo();

        int bestIndex = 0;
        float bestValue = float(x[getInputIndex(coordInfo, bestIndex)]);

        const int Length = ${indexInputShape('axis')};
        const int WorkPerThread = DIV_CEIL(Length, WorkGroupSize);

        for (int w = 0; w < WorkPerThread; ++w) {
          int i = int(gl_GlobalInvocationID.x) * WorkPerThread + w;
          if (i < Length) {
            float candidate = float(x[getInputIndex(coordInfo, i)]);
            if (candidate ${this.op} bestValue && !isnan(candidate)) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
        }

        const int flatOutputIndex = int(gl_GlobalInvocationID.y);
        ${reduceInSharedMemory ? sharedMemoryReduceSnippet :
            'setOutput(flatOutputIndex, int(bestIndex));'}
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        // When this.workGroupSize[0] > 1, each thread reduces Length /
        // this.workGroupSize[0] values. Thes results are stored in shared memory
        // and iteratively reduced.
        const reduceInSharedMemory = this.workGroupSize[0] > 1;
        const sharedMemorySnippet = `
      var<workgroup> xBestIndices : array<u32, ${this.workGroupSize[0]}>;
      var<workgroup> xBestValues : array<f32, ${this.workGroupSize[0]}>;
    `;
        const sharedMemoryReduceSnippet = `
      xBestIndices[localId.x] = bestIndex;
      xBestValues[localId.x] = bestValue;

      for(var currentSize = WorkGroupSize; currentSize > 1u; currentSize = DIV_CEIL(currentSize, ${this.reductionFactor}u)) {
        workgroupBarrier();

        for (var w = 0u; w < ${this.reductionFactor}u; w = w + 1u) {
          let i = localId.x * ${this.reductionFactor}u + w;
          if (i < currentSize) {
            let candidateIndex = xBestIndices[i];
            let candidate = xBestValues[i];
            if(candidate ${this.op} bestValue && !isNanCustom(candidate)) {
              bestValue = candidate;
              bestIndex = candidateIndex;
            }
          }
        }

        xBestIndices[localId.x] = bestIndex;
        xBestValues[localId.x] = bestValue;
      }

      if (localId.x == 0u) {
        setOutputFlatI32(flatOutputIndex, i32(bestIndex));
      }
    `;
        const outputCoordsType = getCoordsDataTypeWgsl(this.outputShape.length);
        const indexOutputCoords = (outputCoords, index) => {
            if (this.outputShape.length === 1) {
                return outputCoords;
            }
            else {
                return `${outputCoords}[${index}]`;
            }
        };
        const indexInputShape = (index) => {
            if (this.inputShape.length === 1) {
                return 'uniforms.xShape';
            }
            else {
                return `uniforms.xShape[${index}]`;
            }
        };
        const userCode = `
      fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
      }

      let WorkGroupSize = ${this.workGroupSize[0]}u;

      ${reduceInSharedMemory ? sharedMemorySnippet : ''}

      // In order to get a flattened index into the input tensor, we need to
      // add back the index along the reduced dimension to |outputCoords|.
      // This function outputs the offset to the first value along
      // |axis| and the stride to get the next value of the input along |axis|.
      fn getInputCoordInfo(globalId : vec3<u32>, globalIndex : u32) -> vec2<u32>{
        let outputCoords : ${outputCoordsType} = getOutputCoords(globalId, globalIndex);
        var i = ${this.outputShape.length - 1}u;

        var stride = 1u;
        var inputStride = 1u;
        var offset = 0u;

        for (var r = 1u; r <= ${this.inputShape.length}u; r = r + 1u) {
          let length = ${indexInputShape(`${this.inputShape.length}u - r`)};
          if (${this.inputShape.length}u - r == uniforms.axis) {
            inputStride = stride;
          } else {
            offset = offset + ${indexOutputCoords('outputCoords', 'i')} * stride;
            i = i - 1u;
          }
          stride = stride * length;
        }

        return vec2<u32>(offset, inputStride);
      }

      fn getInputIndex(coordInfo : vec2<u32>, index : u32) -> u32{
        return coordInfo[0] + coordInfo[1] * index;
      }

      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coordInfo = getInputCoordInfo(globalId, index);

        var bestIndex = 0u;
        var bestValue = x.numbers[getInputIndex(coordInfo, bestIndex)];

        let Length = ${indexInputShape('uniforms.axis')};
        let WorkPerThread = DIV_CEIL(Length, WorkGroupSize);

        for (var w = 0u; w < WorkPerThread; w = w + 1u) {
          let i = globalId.x * WorkPerThread + w;
          if (i < Length) {
            let candidate = x.numbers[getInputIndex(coordInfo, i)];
            if (candidate ${this.op} bestValue && !isNanCustom(f32(candidate))) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
        }

        let flatOutputIndex = globalId.y;
        ${reduceInSharedMemory ?
            sharedMemoryReduceSnippet :
            'setOutputFlatI32(flatOutputIndex, i32(bestIndex));'}
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class TransposeSharedProgram {
    constructor(aShape, newDim) {
        this.variableNames = ['A'];
        // Note that the maximum number of workgroup invocations by webgpu is 256.
        this.workGroupSize = [16, 16, 1];
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [0], y: [1] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [1, 1, 1]);
        this.shaderKey = 'transposeShared';
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const userCode = `
    const int TILE_DIM = ${this.workGroupSize[0]};
    shared float tile[TILE_DIM][TILE_DIM + 1];
    void main() {
        int index = int(gl_GlobalInvocationID.x);
        int x = int(gl_WorkGroupID.x) * TILE_DIM + int(gl_LocalInvocationID.x);
        int y = int(gl_WorkGroupID.y) * TILE_DIM + int(gl_LocalInvocationID.y);
        int width = outShape[0];
        int height = outShape[1];
        if (x < width && y < height) {
          tile[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
              A[y * width + x];
        }
        barrier();

        x = int(gl_WorkGroupID.y) * TILE_DIM + int(gl_LocalInvocationID.x);
        y = int(gl_WorkGroupID.x) * TILE_DIM + int(gl_LocalInvocationID.y);
        if (x < height && y < width) {
          setOutput((y * height + x), tile[gl_LocalInvocationID.x]
            [gl_LocalInvocationID.y]);
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const userCode = `
      let TILE_DIM = ${this.workGroupSize[0]}u;
      var<workgroup> tile : array<array<f32, ${this.workGroupSize[0] + 1}>, ${this.workGroupSize[0]}>;
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let workGroupID = (globalId - localId)/vec3<u32>(${this.workGroupSize[0]}u, ${this.workGroupSize[1]}u, ${this.workGroupSize[2]}u);
        var x = workGroupID.x * TILE_DIM + localId.x;
        var y = workGroupID.y * TILE_DIM + localId.y;
        let width = uniforms.outShape[0];
        let height = uniforms.outShape[1];
        if (x < width && y < height) {
          tile[localId.y][localId.x] =
              A.numbers[y * width + x];
        }
        workgroupBarrier();

        x = workGroupID.y * TILE_DIM + localId.x;
        y = workGroupID.x * TILE_DIM + localId.y;
        if (x < height && y < width) {
          setOutputFlat((y * height + x), tile[localId.x]
            [localId.y]);
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class TransposeProgram {
    constructor(aShape, newDim) {
        this.variableNames = ['A'];
        this.workPerThread = 4;
        this.workGroupSize = [64, 1, 1];
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.newDim = newDim;
        this.shaderKey = `transpose_${newDim}`;
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.outputShape.length);
        const switched = getSwitchedCoords(this.newDim);
        const userCode = `
      void main() {
        int index = getGlobalIndex();

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < size) {
            ${dtype} resRC = getCoordsFromFlatIndex(flatIndex);
            setOutput(flatIndex, A[getFlatIndex(
              ${dtype}(${switched}), aShape)]);
          }
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const dtype = getCoordsDataTypeWgsl(this.outputShape.length);
        const switched = getSwitchedCoords(this.newDim);
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}

        for(var i = 0u; i < ${this.workPerThread}u; i = i + 1u) {
          let flatIndex = index * ${this.workPerThread}u + i;
          if(flatIndex < uniforms.size) {
            let resRC = getCoordsFromFlatIndex(flatIndex);
            setOutputFlat(flatIndex, A.numbers[getFlatIndex${this.outputShape.length}D(
              ${dtype}(${switched}), uniforms.aShape)]);
          }
        }
      }
    `;
        return userCode;
    }
}
function getSwitchedCoords(newDim) {
    const rank = newDim.length;
    if (rank > 4) {
        throw Error(`Transpose for rank ${rank} is not yet supported`);
    }
    const switchedCoords = new Array(rank);
    for (let i = 0; i < newDim.length; i++) {
        switchedCoords[newDim[i]] = `resRC[${i}]`;
    }
    return switchedCoords.join();
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function transpose(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { perm } = attrs;
    const webgpuBackend = backend;
    const xRank = x.shape.length;
    const newShape = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
        newShape[i] = x.shape[perm[i]];
    }
    if (backend.shouldExecuteOnCPU([x])) {
        const xData = webgpuBackend.tensorMap.get(x.dataId);
        const values = xData.values;
        const outValues = transposeImplCPU(values, x.shape, x.dtype, perm, newShape);
        return backend.makeTensorInfo(newShape, x.dtype, outValues);
    }
    if (x.shape.length === 2 && util.arraysEqual(perm, [1, 0])) {
        const program = new TransposeSharedProgram(x.shape, perm);
        return webgpuBackend.runWebGPUProgram(program, [x], x.dtype);
    }
    const program = new TransposeProgram(x.shape, perm);
    return webgpuBackend.runWebGPUProgram(program, [x], x.dtype);
}
const transposeConfig = {
    kernelName: Transpose,
    backendName: 'webgpu',
    kernelFunc: transpose
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function argMax(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis } = attrs;
    let axes = util.parseAxisParam(axis, x.shape);
    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
    let $x = x;
    const intermediateTensorInfos = [];
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        intermediateTensorInfos.push($x);
        axes = backend_util.getInnerMostAxes(axes.length, $x.shape.length);
    }
    backend_util.assertAxesAreInnerMostDims('argMax', [axes[0]], $x.shape.length);
    const program = new ArgMinMaxProgram($x.shape, axes[0], 'max');
    const uniformData = [{ type: 'int32', data: [axes[0]] }];
    const out = backend.runWebGPUProgram(program, [$x], 'int32', uniformData);
    intermediateTensorInfos.forEach(t => backend.disposeData(t.dataId));
    return out;
}
const argMaxConfig = {
    kernelName: ArgMax,
    backendName: 'webgpu',
    kernelFunc: argMax
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function argMin(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis } = attrs;
    let axes = util.parseAxisParam(axis, x.shape);
    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
    let $x = x;
    const intermediateTensorInfos = [];
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        intermediateTensorInfos.push($x);
        axes = backend_util.getInnerMostAxes(axes.length, $x.shape.length);
    }
    backend_util.assertAxesAreInnerMostDims('argMin', [axes[0]], $x.shape.length);
    const program = new ArgMinMaxProgram($x.shape, axes[0], 'min');
    const uniformData = [{ type: 'int32', data: [axes[0]] }];
    const out = backend.runWebGPUProgram(program, [$x], 'int32', uniformData);
    intermediateTensorInfos.forEach(t => backend.disposeData(t.dataId));
    return out;
}
const argMinConfig = {
    kernelName: ArgMin,
    backendName: 'webgpu',
    kernelFunc: argMin
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Pool2DProgram {
    constructor(convInfo, poolType) {
        this.variableNames = ['x'];
        this.uniforms = 'ivec2 pad, stride, dilation, convDims, filterDims;';
        this.uniformsWgsl = `pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>; convDims : vec2<u32>; filterDims : vec2<u32>;`;
        // TODO(jiajia.qin@intel.com): Dynamically choose different workGroupSize for
        // different output shapes.
        this.workGroupSize = [128, 1, 1];
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.shaderKey = `pool2D_${poolType}`;
        this.poolType = poolType;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        let updateSnippet = `resultValue = max(value, resultValue);`;
        if (this.poolType === 'avg') {
            updateSnippet = `resultValue += value; count += 1.0;`;
        }
        let returnValue = `resultValue`;
        if (this.poolType === 'avg') {
            returnValue = `resultValue / count`;
        }
        const userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        if (coordsInBounds(coords, outShape)) {
          int batch = coords[0];
          ivec2 xRCCorner = coords.yz * stride - pad;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          float resultValue = ${this.poolType === 'avg' ? '0.0' : '-1.0 / 1e-20'};
          float count = 0.0;

          for (int wR = 0; wR < filterDims.x; wR += dilation.x) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= convDims.x) {
              continue;
            }

            for (int wC = 0; wC < filterDims.y; wC += dilation.y) {
              int xC = xCCorner + wC;
              if (xC < 0 || xC >= convDims.y) {
                continue;
              }

              float value = getX(batch, xR, xC, coords[3]);
              ${updateSnippet}
            }
          }

          setOutput(batch, coords[1], coords[2], coords[3], ${returnValue});
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        let updateSnippet = `resultValue = max(value, resultValue);`;
        if (this.poolType === 'avg') {
            updateSnippet = `resultValue = resultValue + value; count = count + 1.0;`;
        }
        let returnValue = `resultValue`;
        if (this.poolType === 'avg') {
            returnValue = `resultValue / count`;
        }
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coords = getOutputCoords(globalId, index);
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          let batch = coords[0];
          let xRCCorner = vec2<i32>(coords.yz * uniforms.stride - uniforms.pad);
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          var resultValue = ${this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};
          var count = 0.0;

          for (var wR = 0u; wR < uniforms.filterDims.x; wR = wR + uniforms.dilation.x) {
            let xR = xRCorner + i32(wR);

            if (xR < 0 || xR >= i32(uniforms.convDims.x)) {
              continue;
            }

            for (var wC = 0u; wC < uniforms.filterDims.y; wC = wC + uniforms.dilation.y) {
              let xC = xCCorner + i32(wC);
              if (xC < 0 || xC >= i32(uniforms.convDims.y)) {
                continue;
              }

              let value = getX(batch, u32(xR), u32(xC), coords[3]);
              ${updateSnippet}
            }
          }

          setOutput(batch, coords[1], coords[2], coords[3], ${returnValue});
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class PoolWithFilterSizeEqualsOneProgram {
    constructor(convInfo) {
        this.variableNames = ['x'];
        this.uniforms = 'ivec2 pad, stride, dilation, convDims, filterDims;';
        this.uniformsWgsl = `pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>; convDims : vec2<u32>; filterDims : vec2<u32>;`;
        this.workGroupSize = [256, 1, 1];
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.shaderKey = 'poolWithFilterSizeEqualsOne';
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        if (all(lessThan(coords, outShape))) {
          ivec2 xRCCorner = coords.yz * stride;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          float value = getX(batch, xRCorner, xCCorner, d);
          setOutput(batch, coords[1], coords[2], d, value);
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coords = getOutputCoords(globalId, index);
        let batch = coords[0];
        let d = coords[3];

        if (all(coords < uniforms.outShape)) {
          let xRCCorner = coords.yz * uniforms.stride;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          let value = getX(batch, xRCorner, xCCorner, d);
          setOutput(batch, coords[1], coords[2], d, value);
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function avgPool(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const dilations = 1;
    const convInfo = backend_util.computePool2DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
        util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
        return identity({ inputs: { x }, backend });
    }
    let program;
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1) {
        program = new PoolWithFilterSizeEqualsOneProgram(convInfo);
    }
    else {
        program = new Pool2DProgram(convInfo, 'avg');
    }
    const dimensions = [
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] }, {
            type: 'int32',
            data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        }
    ];
    return backend.runWebGPUProgram(program, [x], x.dtype, dimensions);
}
const avgPoolConfig = {
    kernelName: AvgPool,
    backendName: 'webgpu',
    kernelFunc: avgPool
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function batchMatMul(args) {
    const { inputs, backend, attrs } = args;
    const { a, b } = inputs;
    const { transposeA, transposeB } = attrs;
    return batchMatMulImpl({ a, b, transposeA, transposeB, backend });
}
const batchMatMulConfig = {
    kernelName: BatchMatMul,
    backendName: 'webgpu',
    kernelFunc: batchMatMul,
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class SliceProgram {
    constructor(start, destSize) {
        this.variableNames = ['source'];
        this.workPerThread = 1;
        this.workGroupSize = [64, 1, 1];
        this.outputShape = destSize;
        this.rank = destSize.length;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.start = start;
        this.uniforms = `${getCoordsDataType(start.length)} start; `;
        this.uniformsWgsl = `start : ${getCoordsDataTypeWgsl(start.length)}; `;
        this.shaderKey = 'slice';
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.rank);
        const sourceCoords = getCoords(this.rank);
        let coordSum;
        if (this.start.length === 1) {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc.${coords[i]} = start + coords.${coords[i]};`;
            });
        }
        else {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc.${coords[i]} = start[${i}] + coords.${coords[i]};`;
            });
        }
        const userCode = `
      void main() {
        int index = getGlobalIndex();
        if (index < size)
        {
          ${dtype} sourceLoc;
          ${dtype} coords = getOutputCoords();
          ${coordSum.join('\n')}
          setOutput(index, getSource(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const dtype = getCoordsDataTypeWgsl(this.rank);
        const sourceCoords = getCoords(this.rank);
        let coordSum;
        if (this.start.length === 1) {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc = uniforms.start + coords;`;
            });
        }
        else {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc.${coords[i]} = uniforms.start[${i}] + coords.${coords[i]};`;
            });
        }
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if (index < uniforms.size)
        {
          var sourceLoc : ${dtype};
          let coords = getOutputCoords(globalId, index);
          ${coordSum.join('\n')}
          setOutputFlat(index, getSource(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
const coords = ['x', 'y', 'z', 'w', 'u', 'v'];
function getCoords(rank) {
    if (rank === 1) {
        return 'sourceLoc';
    }
    else if (rank <= 6) {
        return coords.slice(0, rank).map(coord => `sourceLoc.${coord}`).join(',');
    }
    else {
        throw Error(`Slicing for rank ${rank} is not yet supported`);
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function slice(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { begin, size } = attrs;
    const [$begin, $size] = slice_util.parseSliceParams(x, begin, size);
    slice_util.assertParamsValid(x, $begin, $size);
    if (backend.shouldExecuteOnCPU([x]) || x.dtype === 'string') {
        const xBufferInfo = backend.tensorMap.get(x.dataId);
        const outValues = sliceImplCPU(xBufferInfo.values, $begin, $size, x.shape, x.dtype);
        return backend.makeTensorInfo($size, x.dtype, outValues);
    }
    if (util.sizeFromShape($size) === 0) {
        return backend.makeTensorInfo($size, x.dtype, []);
    }
    // TODO(xing.xu): Add shadow slice support.
    const program = new SliceProgram($begin, $size);
    const uniformData = [{ type: 'int32', data: $begin }];
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}
const sliceConfig = {
    kernelName: Slice,
    backendName: 'webgpu',
    kernelFunc: slice
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const batchToSpaceND = (args) => {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockShape, crops } = attrs;
    util.assert(x.shape.length <= 4, () => 'batchToSpaceND for rank > 4 with a WebGPU backend not ' +
        'implemented yet');
    const prod = blockShape.reduce((a, b) => a * b);
    const reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
    const permuted = backend_util.getPermuted(reshaped.length, blockShape.length);
    const reshapedPermuted = backend_util.getReshapedPermuted(x.shape, blockShape, prod);
    const sliceBeginCoords = backend_util.getSliceBeginCoords(crops, blockShape.length);
    const sliceSize = backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);
    const toDispose = [];
    const reshapedIntermediate = reshape({ inputs: { x }, backend, attrs: { shape: reshaped } });
    const transposedIntermediate = transpose({ inputs: { x: reshapedIntermediate }, backend, attrs: { perm: permuted } });
    const reshapedIntermediate2 = reshape({
        inputs: { x: transposedIntermediate },
        backend,
        attrs: { shape: reshapedPermuted }
    });
    const sliced = slice({
        inputs: { x: reshapedIntermediate2 },
        backend,
        attrs: { begin: sliceBeginCoords, size: sliceSize }
    });
    toDispose.push(reshapedIntermediate);
    toDispose.push(transposedIntermediate);
    toDispose.push(reshapedIntermediate2);
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return sliced;
};
const batchToSpaceNDConfig = {
    kernelName: BatchToSpaceND,
    backendName: 'webgpu',
    kernelFunc: batchToSpaceND
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const notEqual = binaryKernelFunc({
    opSnippet: BinaryOpType.NOT_EQUAL,
    dtype: 'bool',
    cpuKernelImpl: notEqualImplCPU
});
const notEqualConfig = {
    kernelName: NotEqual,
    backendName: 'webgpu',
    kernelFunc: notEqual
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function real(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const inputData = backend.tensorMap.get(input.dataId);
    return identity({ inputs: { x: inputData.complexTensorInfos.real }, backend });
}
const realConfig = {
    kernelName: Real,
    backendName: 'webgpu',
    kernelFunc: real
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function int(input, backend) {
    const program = new UnaryOpProgram(input.shape, UnaryOpType.TO_INT);
    const output = backend.runWebGPUProgram(program, [input], 'int32');
    return { dataId: output.dataId, shape: output.shape, dtype: output.dtype };
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function cast(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { dtype } = attrs;
    // Casting to complex64.
    if (dtype === 'complex64') {
        if (x.dtype === 'complex64') {
            return identity({ inputs: { x }, backend });
        }
        // TODO: Import kernel function once zeros is modularized.
        const zerosTensor = zeros(x.shape);
        const floatX = cast({ inputs: { x }, backend, attrs: { dtype: 'float32' } });
        const result = complex({ inputs: { real: floatX, imag: zerosTensor }, backend });
        zerosTensor.dispose();
        backend.disposeData(floatX.dataId);
        return result;
    }
    // Casting from complex64
    if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const result = cast({ inputs: { x: realPart }, backend, attrs: { dtype } });
        backend.disposeData(realPart.dataId);
        return result;
    }
    if (!util.hasEncodingLoss(x.dtype, dtype)) {
        // We don't change the underlying data, since we cast to higher
        // precision.
        const result = identity({ inputs: { x }, backend });
        return { dataId: result.dataId, shape: result.shape, dtype };
    }
    if (dtype === 'int32') {
        return int(x, backend);
    }
    if (dtype === 'bool') {
        const zerosTensorInfo = backend.makeTensorInfo([], 'bool', util.getTypedArrayFromDType('bool', 1));
        const binaryInputs = { a: x, b: zerosTensorInfo };
        const result = notEqual({ inputs: binaryInputs, backend });
        backend.disposeData(zerosTensorInfo.dataId);
        return result;
    }
    throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
}
const castConfig = {
    kernelName: Cast,
    backendName: 'webgpu',
    kernelFunc: cast
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const ceil = unaryKernelFunc({ opType: UnaryOpType.CEIL, cpuKernelImpl: ceilImplCPU });
const ceilConfig = {
    kernelName: Ceil,
    backendName: 'webgpu',
    kernelFunc: ceil
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class ClipVec4Program {
    constructor(outputShape) {
        this.variableNames = ['A'];
        this.uniforms = 'float minVal; float maxVal;';
        this.uniformsWgsl = 'minVal : f32; maxVal : f32;';
        this.workPerThread = 4;
        this.workGroupSize = [64, 1, 1];
        this.isVec4 = true;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.shaderKey = 'clipVec4';
        this.size = util.sizeFromShape(this.outputShape) / 4;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const userCode = `
      void main() {
        int index = getGlobalIndex();
          if(index < size) {
            vec4 value = getAAtOutCoords();
            vec4 clampedValue;
            for (int i = 0; i < 4; ++i) {
              if (isnan(value[i])) {
                clampedValue[i] = value[i];
              } else {
                clampedValue[i] = clamp(value[i], minVal, maxVal);
              }
            }

            setOutput(index, clampedValue);
          }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if(index < uniforms.size) {
          let value = getAAtOutCoordsByGlobalId(globalId, index);
          var clampedValue : vec4<f32>;
          for (var i = 0u; i < 4u; i = i + 1u) {
            if (isNanCustom(value[i])) {
              clampedValue[i] = value[i];
            } else {
              clampedValue[i] = clamp(value[i], uniforms.minVal, uniforms.maxVal);
            }
          }

          setOutputFlat(index, clampedValue);
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class ClipProgram {
    constructor(outputShape) {
        this.variableNames = ['A'];
        this.uniforms = 'float minVal; float maxVal;';
        this.uniformsWgsl = 'minVal : f32; maxVal : f32;';
        this.workGroupSize = [64, 1, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.shaderKey = 'clip';
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const userCode = `
      void main() {
        int index = getGlobalIndex();
        if(index < size) {
          float value = getAAtOutCoords();
          if (isnan(value)) {
            setOutput(index, value);
            return;
          }
          setOutput(index, clamp(value, minVal, maxVal));
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if(index < uniforms.size) {
          let value = getAAtOutCoordsByGlobalId(globalId, index);
          if (isNanCustom(value)) {
            setOutputFlat(index, value);
            return;
          }
          setOutputFlat(index, clamp(value, uniforms.minVal, uniforms.maxVal));
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function clipByValue(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { clipValueMin, clipValueMax } = attrs;
    let program;
    const uniformData = [
        { type: 'float32', data: [clipValueMin] },
        { type: 'float32', data: [clipValueMax] }
    ];
    if (util.sizeFromShape(x.shape) % 4 === 0) {
        program = new ClipVec4Program(x.shape);
    }
    else {
        program = new ClipProgram(x.shape);
    }
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}
const clipByValueConfig = {
    kernelName: ClipByValue,
    backendName: 'webgpu',
    kernelFunc: clipByValue
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class ConcatProgram {
    constructor(shapes) {
        this.workPerThread = 4;
        this.workGroupSize = [64, 1, 1];
        this.outputShape =
            backend_util.computeOutShape(shapes, 1 /* axis */);
        this.variableNames = shapes.map((_, i) => `T${i}`);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.shapes = shapes;
        // shapes is used by const snippets.
        this.shaderKey = `concat${shapes}`;
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const offsets = new Array(this.shapes.length - 1);
        const snippets = [];
        if (offsets.length > 0) {
            offsets[0] = this.shapes[0][1];
            for (let i = 1; i < offsets.length; i++) {
                offsets[i] = offsets[i - 1] + this.shapes[i][1];
            }
            snippets.push(`if (yC < ${offsets[0]}) setOutput(coords.x, coords.y, getT0(yR, yC));`);
            for (let i = 1; i < offsets.length; i++) {
                const shift = offsets[i - 1];
                snippets.push(`else if (yC < ${offsets[i]}) ` +
                    `setOutput(coords.x, coords.y, getT${i}(yR, yC-${shift}));`);
            }
            const lastIndex = offsets.length;
            const lastShift = offsets[offsets.length - 1];
            snippets.push(`else setOutput(coords.x, coords.y, getT${lastIndex}(yR, yC-${lastShift}));`);
        }
        else {
            snippets.push(`setOutput(coords.x, coords.y, getT0(yR, yC));`);
        }
        const userCode = `
      void main() {
        int index = getGlobalIndex();

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < size) {
            ivec2 coords = getCoordsFromFlatIndex(flatIndex);
            int yR = coords.x;
            int yC = coords.y;

            ${snippets.join('\n        ')}
          }
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const offsets = new Array(this.shapes.length - 1);
        const snippets = [];
        if (offsets.length > 0) {
            offsets[0] = this.shapes[0][1];
            for (let i = 1; i < offsets.length; i++) {
                offsets[i] = offsets[i - 1] + this.shapes[i][1];
            }
            snippets.push(`if (yC < ${offsets[0]}u){ setOutput(coords.x, coords.y, getT0(yR, yC)); }`);
            for (let i = 1; i < offsets.length; i++) {
                const shift = offsets[i - 1];
                snippets.push(`elseif (yC < ${offsets[i]}u){ ` +
                    `setOutput(coords.x, coords.y, getT${i}(yR, yC - ${shift}u)); }`);
            }
            const lastIndex = offsets.length;
            const lastShift = offsets[offsets.length - 1];
            snippets.push(`else { setOutput(coords.x, coords.y, getT${lastIndex}(yR, yC - ${lastShift}u)); }`);
        }
        else {
            snippets.push(`setOutput(coords.x, coords.y, getT0(yR, yC));`);
        }
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        for(var i = 0u; i < ${this.workPerThread}u; i = i + 1u) {
          let flatIndex = index * ${this.workPerThread}u + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromFlatIndex(flatIndex);
            let yR = coords.x;
            let yC = coords.y;

            ${snippets.join('\n        ')}
          }
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function imag(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const inputData = backend.tensorMap.get(input.dataId);
    return identity({ inputs: { x: inputData.complexTensorInfos.imag }, backend });
}
const imagConfig = {
    kernelName: Imag,
    backendName: 'webgpu',
    kernelFunc: imag
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function concatImpl$1(inputs, axis, backend) {
    const dtype = inputs[0].dtype;
    if (dtype === 'complex64') {
        const reals = inputs.map((t) => real({ inputs: { input: t }, backend }));
        const imags = inputs.map((t) => imag({ inputs: { input: t }, backend }));
        const realConcated = concatImpl$1(reals, axis, backend);
        const imagConcated = concatImpl$1(imags, axis, backend);
        const result = complex({ inputs: { real: realConcated, imag: imagConcated }, backend });
        reals.forEach(r => backend.disposeData(r.dataId));
        imags.forEach(i => backend.disposeData(i.dataId));
        backend.disposeData(realConcated.dataId);
        backend.disposeData(imagConcated.dataId);
        return result;
    }
    let runOnCpu = backend.shouldExecuteOnCPU(inputs);
    // Run on cpu if dtype is string. For string, the backend represents it
    // as Uint8Array[], where each Uint8Array is a character. Given that the
    // computation is only on the outer array, uploading the whole data onto
    // gpu is wasteful. Also, currently webgpu doesn't have a design to
    // upload and retrieve Uint8Array[] between cpu and gpu. Therefore, we
    // just run the kernel on cpu if dtype is string.
    if (dtype === 'string') {
        runOnCpu = true;
    }
    if (runOnCpu) {
        // Any concat of n-dimensional tensors across any axis can be reduced to
        // a concatenation of two-dimensional tensors across the axis 1 by first
        // partitioning the axes of the original tensors into those less than the
        // axis to be concatenated and the rest. Then reshape the tensors
        // into a two-dimensional tensor by collapsing these two sets of axes and
        // concatenate the resulting matrices across the axis 1, finally reshaping
        // the result to have the proper shape.
        const tensors2D = inputs.map(t => {
            const innerSize = util.sizeFromShape(t.shape.slice(axis));
            const shape = [-1, innerSize];
            return reshape({ inputs: { x: t }, backend, attrs: { shape } });
        });
        const inputsValShapes = tensors2D.map(t => {
            return { vals: backend.readSync(t.dataId), shape: t.shape };
        });
        // Concats 2d tensors along axis=1.
        const outShape = backend_util.computeOutShape(tensors2D.map(t => t.shape), 1 /* axis */);
        const simplyConcat = tensors2D[0].shape[0] === 1;
        const outVals = concatImplCPU(inputsValShapes, outShape, dtype, simplyConcat);
        const finalOutShape = backend_util.computeOutShape(inputs.map(t => t.shape), axis);
        const outInfo = backend.makeTensorInfo(finalOutShape, dtype, outVals);
        tensors2D.forEach(t => backend.disposeData(t.dataId));
        return outInfo;
    }
    const { tensors2D, outShape } = computeTensors2D(inputs, axis, backend);
    const program = new ConcatProgram((tensors2D).map(t => t.shape));
    const res = backend.runWebGPUProgram(program, tensors2D, tensors2D[0].dtype);
    tensors2D.forEach(r => backend.disposeData(r.dataId));
    const reshapedResult = reshape({ inputs: { x: res }, backend, attrs: { shape: outShape } });
    backend.disposeData(res.dataId);
    return reshapedResult;
}
function computeTensors2D(inputs, axis, backend) {
    const outShape = backend_util.computeOutShape(inputs.map(t => t.shape), axis);
    const tensors2D = inputs.map(t => reshape({
        inputs: { x: t },
        backend,
        attrs: {
            shape: [
                util.sizeFromShape(t.shape.slice(0, axis)),
                util.sizeFromShape(t.shape.slice(axis))
            ]
        }
    }));
    return { tensors2D, outShape };
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function concat(args) {
    const { inputs, backend, attrs } = args;
    const { axis } = attrs;
    const $axis = util.parseAxisParam(axis, inputs[0].shape)[0];
    const outShape = backend_util.computeOutShape(inputs.map(t => t.shape), $axis);
    if (util.sizeFromShape(outShape) === 0) {
        return backend.makeTensorInfo(outShape, inputs[0].dtype, []);
    }
    // Keep only non-empty tensors (ignore tensors with 0 in their shape).
    const $inputs = inputs.filter(t => util.sizeFromShape(t.shape) > 0);
    if ($inputs.length === 1) {
        return identity({ inputs: { x: $inputs[0] }, backend });
    }
    const shapes = $inputs.map(t => t.shape);
    backend_util.assertParamsConsistent(shapes, $axis);
    return concatImpl$1($inputs, $axis, backend);
}
const concatConfig = {
    kernelName: Concat,
    backendName: 'webgpu',
    kernelFunc: concat
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Im2ColProgram {
    constructor(outputShape, isChannelsLast) {
        this.variableNames = ['A'];
        this.uniforms = `ivec2 pad, stride, dilation; int outWidth, itemsPerBlockRow,
      inChannels;`;
        this.workPerThread = 4;
        this.workGroupSize = [64, 1, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.isChannelsLast = isChannelsLast;
        this.shaderKey = `im2col_${this.isChannelsLast}`;
        this.size = util.sizeFromShape(this.outputShape);
    }
    getUserCode() {
        const rowDim = this.isChannelsLast ? 0 : 1;
        const colDim = this.isChannelsLast ? 1 : 2;
        const userCode = `
      void main() {
        int index = getGlobalIndex();

        for(int i=0; i<${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          ivec2 rc = getCoordsFromFlatIndex(flatIndex);

          if(flatIndex < size) {
            int blockIndex = rc[0];
            int pos = rc[1];

            int offsetY = int(blockIndex / outWidth) * stride[1] - pad[1];
            int d0 = offsetY + dilation[1] * (pos / itemsPerBlockRow);
            float value = 0.0;
            if(d0 < aShape[${rowDim}] && d0 >= 0) {
              int offsetX = int(mod(blockIndex, outWidth) * stride[0] -
                pad[0]);
              int d1 = offsetX + dilation[0] * (int(mod(pos,
                itemsPerBlockRow) / inChannels));
              int ch = int(mod(pos, inChannels));
              if(d1 < aShape[${colDim}] && d1 >= 0) {
                value = getA(d0, d1, ch);
              }
            }
            setOutput(flatIndex, value);
          }
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// For 1x1 kernels that iterate through every point in the input, convolution
// can be expressed as matrix multiplication (without need for memory
// remapping).
function conv2dByMatMul({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    const xShape = x.shape;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const transposeA = false;
    const transposeB = false;
    const targetShape = isChannelsLast ? xShape[0] * xShape[1] * xShape[2] :
        xShape[0] * xShape[2] * xShape[3];
    const xReshaped = reshape({
        inputs: { x },
        backend,
        attrs: { shape: [1, targetShape, convInfo.inChannels] }
    });
    const filterReshaped = reshape({
        inputs: { x: filter },
        backend,
        attrs: { shape: [1, convInfo.inChannels, convInfo.outChannels] }
    });
    const result = batchMatMulImpl({
        a: xReshaped,
        b: filterReshaped,
        transposeA,
        transposeB,
        backend,
        bias,
        activation,
        preluActivationWeights,
        leakyreluAlpha
    });
    const out = reshape({ inputs: { x: result }, backend, attrs: { shape: convInfo.outShape } });
    backend.disposeData(xReshaped.dataId);
    backend.disposeData(filterReshaped.dataId);
    backend.disposeData(result.dataId);
    return out;
}
// Implements the im2row algorithm as outlined in "High Performance
// Convolutional Neural Networks for Document Processing" (Suvisoft, 2006)
function conv2dWithIm2Col({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    // Rearranges conv2d input so each block to be convolved over forms the
    // column of a new matrix with shape [filterWidth * filterHeight *
    // inChannels, outHeight * outWidth]. The filter is also rearranged so each
    // output channel forms a row of a new matrix with shape [outChannels,
    // filterWidth * filterHeight * inChannels]. The convolution is then
    // computed by multiplying these matrices and reshaping the result.
    const { filterWidth, filterHeight, inChannels, strideWidth, strideHeight, padInfo, outWidth, outHeight, dilationWidth, dilationHeight, dataFormat } = convInfo;
    const isChannelsLast = dataFormat === 'channelsLast';
    const sharedDim = filterWidth * filterHeight * inChannels;
    const numCols = outHeight * outWidth;
    const x2ColShape = [numCols, sharedDim];
    const transposeA = false;
    const transposeB = false;
    const intermediates = [];
    const xSqueezed = reshape({ inputs: { x }, backend, attrs: { shape: x.shape.slice(1) } });
    const w2Row = reshape({ inputs: { x: filter }, backend, attrs: { shape: [1, sharedDim, -1] } });
    intermediates.push(xSqueezed);
    intermediates.push(w2Row);
    const im2ColProgram = new Im2ColProgram(x2ColShape, isChannelsLast);
    const dimensions = [
        { type: 'int32', data: [padInfo.left, padInfo.top] },
        { type: 'int32', data: [strideWidth, strideHeight] },
        { type: 'int32', data: [dilationWidth, dilationHeight] },
        { type: 'int32', data: [outWidth] },
        { type: 'int32', data: [inChannels * filterWidth] },
        { type: 'int32', data: [inChannels] }
    ];
    const im2Col = backend.runWebGPUProgram(im2ColProgram, [xSqueezed], xSqueezed.dtype, dimensions);
    const im2Col3D = reshape({
        inputs: { x: im2Col },
        backend,
        attrs: { shape: [1, x2ColShape[0], x2ColShape[1]] }
    });
    intermediates.push(im2Col);
    intermediates.push(im2Col3D);
    const matMulProgram = new MatMulPackedProgram([1, x2ColShape[0], x2ColShape[1]], [1, numCols, convInfo.outChannels], env().get('WEBGPU_MATMUL_WORK_PER_THREAD'), transposeA, transposeB);
    const result = backend.runWebGPUProgram(matMulProgram, [im2Col3D, w2Row], im2Col3D.dtype);
    const outShape = isChannelsLast ?
        [1, outHeight, outWidth, convInfo.outChannels] :
        [1, convInfo.outChannels, outHeight, outWidth];
    const out = reshape({ inputs: { x: result }, backend, attrs: { shape: outShape } });
    intermediates.push(result);
    for (const i of intermediates) {
        backend.disposeData(i.dataId);
    }
    return out;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Conv2DMMVec4Program {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivationWeights = false, hasLeakyreluAlpha = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'ivec2 filterDims, pad, stride, dilation;';
        this.uniformsWgsl = `filterDims : vec2<u32>; pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>;
      dimAOuter : u32; dimBOuter : u32; dimInner : u32;`;
        this.isVec4 = true;
        this.outputShape = convInfo.outShape;
        util.assert(convInfo.dataFormat === 'channelsLast', () => 'TODO: NCHW is unimplemented');
        this.dispatchLayout = { x: [3], y: [1, 2], z: [0] };
        this.workGroupSize = [8, 8, 1];
        const elementsPerThread = [4, 4, 1];
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, elementsPerThread);
        this.convInfo = convInfo;
        this.addBias = addBias;
        this.useWgsl = getUseWgsl();
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        this.hasLeakyreluAlpha = hasLeakyreluAlpha;
        if (this.addBias) {
            this.variableNames.push('bias');
        }
        if (this.hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        if (this.hasLeakyreluAlpha) {
            this.variableNames.push('leakyreluAlpha');
        }
        [this.fitA, this.fitB] = this.getShapeFit(elementsPerThread);
        this.shaderKey =
            `conv2DMMVec4_${this.activation}_${this.fitA}_${this.fitB}`;
    }
    getShapeFit(elementsPerThread) {
        const tileAOuter = this.workGroupSize[1] * elementsPerThread[1];
        const tileBOuter = this.workGroupSize[0] * elementsPerThread[0];
        const tileInner = tileBOuter;
        const tileSizeA = [tileAOuter, tileInner];
        const tileSizeB = [tileInner, tileBOuter];
        const dimAOuter = this.outputShape[1] * this.outputShape[2];
        const dimBOuter = this.outputShape[3];
        const dimInner = this.convInfo.filterHeight * this.convInfo.filterWidth *
            this.convInfo.inChannels;
        return [
            tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]),
            tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter])
        ];
    }
    getUserCode() {
        const elementsPerThread = [4, 4, 1];
        const matMulSource = makeMatMulPackedVec4Source(elementsPerThread);
        // Below code only applys to valid padding type.
        const sampleAWithRemainder = `int flatIndex = getFlatIndex(coord, xShape);
        int divBy4Remainder = flatIndex % 4;
        int divBy4Index = flatIndex / 4;
        vec4 curData = x[divBy4Index];
        if (divBy4Remainder == 0) {
          temp = curData;
        } else {
          // TODO: This could end up being a redundant load with another one in
          // the same shader invocation. Perhaps there's an opportunity for
          // optimization
          vec4 nextData = x[divBy4Index + 1];
          if (divBy4Remainder == 1) {
            temp = vec4(curData.yzw, nextData.x);
          } else if (divBy4Remainder == 2) {
            temp = vec4(curData.zw, nextData.xy);
          } else if (divBy4Remainder == 3) {
            temp = vec4(curData.w, nextData.xyz);
          }
        }
        `;
        const remainder = this.convInfo.inChannels % 4;
        const remainderSnippet = remainder === 0 ?
            `// The bounds checking is always needed since we use it to pad zero for
        // the 'same' padding type.
        resData = coordsInBounds(coord, xShape) ?
        x[getFlatIndex(coord, xShape) / 4] : vec4(0.0);` :
            `vec4 temp = vec4(0.0);
        ${sampleAWithRemainder}
        resData = temp;
        if (WCol == (filterDims[1] - 1)) {
          coord = ivec4(
            coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
          ${sampleAWithRemainder}
          if (inChCoord == 0) {
            resData = vec4(resData.xyz, temp.x);
          } else if (inChCoord == 1) {
            resData = vec4(resData.xy, temp.xy);
          } else {
            resData = vec4(resData.x, temp.xyz);
          }
        }
        `;
        const readASnippet = `int outRow = r / outShape[2];
        int outCol = r % outShape[2];
        int WRow = c / (filterDims[1] * xShape[3]);
        int WCol = (c / xShape[3]) % filterDims[1];
        int inChCoord = c % xShape[3];
        ivec4 coord = ivec4(
            batch,
            outRow * stride[0] + dilation[0] * WRow - pad[0],
            outCol * stride[1] + dilation[1] * WCol - pad[1],
            inChCoord);
        vec4 resData = vec4(0.0);
        ${remainderSnippet}
        return resData;`;
        const sampleA = this.fitA ? `${readASnippet}` : `if (r < dimAOuter && c < dimInner) {
          ${readASnippet}
        } else {
          return vec4(0.0);
        }`;
        const sampleB = this.fitB ?
            `W[row * dimBOuter / 4 + col]` :
            `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
            W[row * dimBOuter / 4 + col] : vec4(0.0)`;
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation, this.isVec4, this.useWgsl);
            if (this.hasPreluActivationWeights) {
                activationSnippet = `vec4 activation(vec4 a, ivec4 outCoord) {
          vec4 b = getPreluActivationWeightsAtOutCoords(outCoord);
          ${activationOp}
        }`;
            }
            else if (this.hasLeakyreluAlpha) {
                activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${activationOp}
        }`;
                throw new Error('Leakyrelu is not supported.');
            }
            else {
                activationSnippet = `
        vec4 activation(vec4 a, ivec4 outCoord) {
          ${activationOp}
        }`;
            }
            applyActivationSnippet = `value = activation(value, outCoord);`;
        }
        const addBiasSnippet = this.addBias ? 'ivec4 coords = getOutputCoords(); ' +
            'value += getBiasAtOutCoords(outCoord);' :
            '';
        const userCode = `
        ${activationSnippet}
        ${matMulSource}

        int batch;
        int dimAOuter = outShape[1] * outShape[2];
        int dimBOuter = outShape[3];
        int dimInner = filterDims[0] * filterDims[1] * xShape[3];
        vec4 mm_readA(int row, int col) {
          int r = int(row), c = int(col * 4);
          ${sampleA};
        }

        vec4 mm_readB(int row, int col) {
          return ${sampleB};
        }

        void mm_write(int row, int col, vec4 value) {
          if (row < dimAOuter && col * 4 < dimBOuter)
          {
            ivec4 outCoord = ivec4(
              batch,
              row / outShape[2],
              row % outShape[2],
              col * 4);
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutput(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
              value);
          }
        }

        void main() {
          batch = int(gl_GlobalInvocationID.z);

          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
        return userCode;
    }
    // index is used to avoid repeated definition error.
    getSampleAWithRemainderWgsl(index) {
        return `let flatIndex${index} = getFlatIndex4D(coord, uniforms.xShape);
    let divBy4Remainder${index} = flatIndex${index} % 4u;
    let divBy4Index${index} = flatIndex${index} / 4u;
    let curData${index} = x.numbers[divBy4Index${index}];
    if (divBy4Remainder${index} == 0u) {
      temp = curData${index};
    } else {
      // TODO: This could end up being a redundant load with another one in
      // the same shader invocation. Perhaps there's an opportunity for
      // optimization
      let nextData${index} = x.numbers[divBy4Index${index} + 1u];
      if (divBy4Remainder${index} == 1u) {
        temp = vec4<f32>(curData${index}.yzw, nextData${index}.x);
      } elseif (divBy4Remainder${index} == 2u) {
        temp = vec4<f32>(curData${index}.zw, nextData${index}.xy);
      } elseif (divBy4Remainder${index} == 3u) {
        temp = vec4<f32>(curData${index}.w, nextData${index}.xyz);
      }
    }
    `;
    }
    getUserCodeWgsl() {
        const elementsPerThread = [4, 4, 1];
        const matMulSource = makeMatMulPackedVec4SourceWgsl(elementsPerThread, this.workGroupSize);
        const remainder = this.convInfo.inChannels % 4;
        // Below code only applys to valid padding type.
        const remainderSnippet = remainder === 0 ?
            `// The bounds checking is always needed since we use it to pad zero for
          // the 'same' padding type.
          if (coordsInBounds4D(coord, uniforms.xShape)) {
            resData = x.numbers[getFlatIndex4D(coord, uniforms.xShape) / 4u];
          } else {
            resData = vec4<f32>(0.0); }` :
            `var temp = vec4<f32>(0.0);
          ${this.getSampleAWithRemainderWgsl(1)}
          resData = temp;
          if (WCol == (uniforms.filterDims[1] - 1u)) {
            coord = vec4<u32>(
              coord.x, coord.y + 1u, coord.z + 1u - uniforms.filterDims[1], 0u);
              ${this.getSampleAWithRemainderWgsl(2)}
            if (inChCoord == 0u) {
              resData = vec4<f32>(resData.xyz, temp.x);
            } elseif (inChCoord == 1u) {
              resData = vec4<f32>(resData.xy, temp.xy);
            } else {
              resData = vec4<f32>(resData.x, temp.xyz);
            }
          }
          `;
        const readASnippet = `let outRow = r / uniforms.outShape[2];
        let outCol = r % uniforms.outShape[2];
        let WRow = c / (uniforms.filterDims[1] * uniforms.xShape[3]);
        let WCol = (c / uniforms.xShape[3]) % uniforms.filterDims[1];
        let inChCoord = c % uniforms.xShape[3];
        var coord = vec4<u32>(
            batch,
            outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0],
            outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1],
            inChCoord);
        var resData = vec4<f32>(0.0);
        ${remainderSnippet}
        return resData;`;
        const sampleA = this.fitA ?
            `${readASnippet}` :
            `if (r < uniforms.dimAOuter && c < uniforms.dimInner) {
          ${readASnippet}
         }
         return vec4<f32>(0.0);
        `;
        const sampleB = this.fitB ?
            `return W.numbers[row * uniforms.dimBOuter / 4u + col];` :
            `if(coordsInBounds2D(vec2<u32>(row, col * 4u), vec2<u32>(uniforms.dimInner, uniforms.dimBOuter))) {
           return W.numbers[row * uniforms.dimBOuter / 4u + col];
         }
         return vec4<f32>(0.0);
        `;
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation, this.isVec4, this.useWgsl);
            if (this.hasPreluActivationWeights) {
                activationSnippet =
                    `fn activation(a : vec4<f32>, outCoord : vec4<u32>) -> vec4<f32> {
          let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
          ${activationOp}
        }`;
            }
            else if (this.hasLeakyreluAlpha) {
                activationSnippet = `fn activation(a: vec4<f32>) -> vec4<f32> {
          let b = getLeakyreluAlphaAtOutCoords();
          ${activationOp}
        }`;
                throw new Error('Leakyrelu is not supported.');
            }
            else {
                activationSnippet = `
        fn activation(a : vec4<f32>, outCoord : vec4<u32>) -> vec4<f32> {
          ${activationOp}
        }`;
            }
            applyActivationSnippet = `value = activation(value, outCoord);`;
        }
        const addBiasSnippet = this.addBias ?
            'value = value + getBiasAtOutCoordsByCoords(outCoord);' :
            '';
        const userCode = `
        ${activationSnippet}
        fn mm_readA(row : u32, col : u32, globalId : vec3<u32>) -> vec4<f32> {
          let r = row;
          let c = col * 4u;
          var batch = globalId.z;
          ${sampleA}
        }

        fn mm_readB(row : u32, col : u32, globalId : vec3<u32>) -> vec4<f32> {
          ${sampleB}
        }

        fn mm_write(row : u32, col : u32, valueInput : vec4<f32>, globalId : vec3<u32>) {
          var batch = globalId.z;
          var value = valueInput;
          if (row < uniforms.dimAOuter && col * 4u < uniforms.dimBOuter)
          {
            let outCoord = vec4<u32>(
              batch,
              row / uniforms.outShape[2],
              row % uniforms.outShape[2],
              col * 4u);
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutput(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
              value);
          }
        }
        ${matMulSource}
      `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Conv2DMMProgram {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivationWeights = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'ivec2 filterDims, pad, stride, dilation;';
        this.outputShape = convInfo.outShape;
        util.assert(convInfo.dataFormat === 'channelsLast', () => 'TODO: NCHW is unimplemented');
        this.dispatchLayout = { x: [3], y: [1, 2], z: [0] };
        this.workGroupSize =
            computeWorkGroupSizeForConv2d(this.dispatchLayout, this.outputShape);
        this.elementsPerThread =
            computeWorkPerThreadForConv2d(this.dispatchLayout, this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, this.elementsPerThread);
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.convInfo = convInfo;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        [this.fitA, this.fitB] = this.getShapeFit();
        this.shaderKey = `conv2DMM_${this.elementsPerThread}_${this.activation}_${this.fitA}_${this.fitB}`;
    }
    getShapeFit() {
        const tileAOuter = this.workGroupSize[1] * this.elementsPerThread[1];
        const tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
        const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
        util.assert(tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0, () => 
        // tslint:disable-next-line: max-line-length
        'tileInner must be multiple of workgroupsize.x and workgroupsize.y');
        const tileSizeA = [tileAOuter, tileInner];
        const tileSizeB = [tileInner, tileBOuter];
        const dimAOuter = this.outputShape[1] * this.outputShape[2];
        const dimBOuter = this.outputShape[3];
        const dimInner = this.convInfo.filterHeight * this.convInfo.filterWidth *
            this.convInfo.inChannels;
        return [
            tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]),
            tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter])
        ];
    }
    getUserCode() {
        const matMulSource = makeMatMulPackedSource(this.elementsPerThread);
        const readASnippet = `
    int outRow = row / outShape[2];
    int outCol = row % outShape[2];

    int WRow = col / (filterDims[1] * xShape[3]);
    int WCol = (col / xShape[3]) % filterDims[1];

    ivec4 coord = ivec4(
        batch,
        outRow * stride[0] + dilation[0] * WRow - pad[0],
        outCol * stride[1] + dilation[1] * WCol - pad[1],
        col % xShape[3]);
    // The bounds checking is always needed since we use it to pad zero for the
    // 'same' padding type.
    return coordsInBounds(coord, xShape) ? x[getFlatIndex(coord, xShape)] : 0;`;
        const sampleA = this.fitA ? `${readASnippet}` :
            `if (row < dimAOuter && col < dimInner) {
      ${readASnippet}
    } else {
      return 0;
    }`;
        const sampleB = this.fitB ?
            `W[row * dimBOuter + col]` :
            `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
        W[row * dimBOuter + col] : 0`;
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation);
            if (this.hasPreluActivationWeights) {
                activationSnippet = `float activation(float a, ivec4 outCoord) {
                  float b = getPreluActivationWeightsAtOutCoords(outCoord);
                  ${activationOp}
                }`;
            }
            else {
                activationSnippet = `
                  float activation(float a, ivec4 outCoord) {
                    ${activationOp}
                  }
                `;
            }
            applyActivationSnippet = `value = activation(value, outCoord);`;
        }
        const addBiasSnippet = this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';
        const userCode = `
    ${activationSnippet}
    ${matMulSource}

    int batch;
    int dimAOuter = outShape[1] * outShape[2];
    int dimBOuter = outShape[3];
    int dimInner = filterDims[0] * filterDims[1] * xShape[3];
    float mm_readA(int row, int col) {
      ${sampleA}
    }

    float mm_readB(int row, int col) {
      return ${sampleB};
    }

    void mm_write(int row, int col, float value) {
      ivec4 outCoord = ivec4(
          batch,
          row / outShape[2],
          row % outShape[2],
          col);
      ${addBiasSnippet}
      ${applyActivationSnippet}
      result[getFlatIndex(outCoord, outShape)] = value;
    }

    void main() {
      batch = int(gl_GlobalInvocationID.z);

      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
  `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Conv2DNaiveProgram {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivationWeights = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'ivec2 filterDims, pad, stride, dilation;';
        this.workGroupSize = [128, 1, 1];
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        util.assert(convInfo.dataFormat === 'channelsLast', () => 'TODO: NCHW is unimplemented');
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.convInfo = convInfo;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        this.shaderKey = `conv2DNaive_${this.activation}`;
    }
    getUserCode() {
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation);
            if (this.hasPreluActivationWeights) {
                activationSnippet = `float activation(float a) {
                  float b = getPreluActivationWeightsAtOutCoords();
                  ${activationOp}
                }`;
            }
            else {
                activationSnippet = `
                  float activation(float a) {
                    ${activationOp}
                  }
                `;
            }
            applyActivationSnippet = `value = activation(value);`;
        }
        const addBiasSnippet = this.addBias ? 'value += getBiasAtOutCoords();' : '';
        const userCode = `
      ${activationSnippet}
      float readInp(int batch, int row, int col, int chan) {
        ivec4 coord = ivec4(batch, row, col, chan);
        return coordsInBounds(coord, xShape) ?
          getX(batch, row, col, chan) : 0;
      }

      float readFilt(int row, int col, int xChannel, int outChannel) {
        ivec4 coord = ivec4(row, col, xChannel, outChannel);
        return coordsInBounds(coord, wShape) ?
          getW(row, col, xChannel, outChannel) : 0;
      }

      void writeResult(int batch, int row, int col, int chan, float value) {
        ivec4 coord = ivec4(batch, row, col, chan);
        if (coordsInBounds(coord, outShape)) {
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(batch, row, col, chan, value);
        }
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int outChannel = coords[3];

        float acc = 0.0;

        for (int row = 0; row < filterDims[0]; ++row) {
          for (int col = 0; col < filterDims[1]; ++col) {
            for (int xChannel = 0; xChannel < xShape[3]; ++xChannel) {
              float v = readInp(batch,
                  coords[1] * stride[0] + dilation[0] * row - pad[0],
                  coords[2] * stride[1] + dilation[1] * col - pad[1],
                  xChannel);
              float f = readFilt(row, col, xChannel, outChannel);
              acc += v * f;
            }
          }
        }

        writeResult(batch, coords[1], coords[2], outChannel, acc);
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function conv2d(args) {
    const { inputs, attrs, backend } = args;
    const { x, filter } = inputs;
    const { strides, pad, dataFormat, dilations, dimRoundingMode } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, dilations, pad, dimRoundingMode, false /* depthwise */, $dataFormat);
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
        (convInfo.padInfo.type === 'SAME' || convInfo.padInfo.type === 'VALID')) {
        return conv2dByMatMul({ x, filter, convInfo, backend });
    }
    if (env().getBool('WEBGPU_CONV_SEPARATE_IM2COL_SHADER') && x.shape[0] === 1) {
        return conv2dWithIm2Col({ x, filter, convInfo, backend });
    }
    let program;
    const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];
    const dimensions = [
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        { type: 'int32', data: [...padInfo] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] }
    ];
    if (env().getBool('WEBGPU_USE_NAIVE_CONV2D')) {
        // TODO(kainino0x): This may be obsolete, but is kept for reference.
        program = new Conv2DNaiveProgram(convInfo);
    }
    else if (
    // TODO(jiajia.qin@intel.com): It seems that the vec4 version is not
    // good if convInfo.outChannels is too small. For example, input = [1,
    // 128, 128, 4], filter = [25, 25, 4, 4]. In this case, lots of theads
    // will run idle. So temporarily, use 64 as the threshold.
    (convInfo.inChannels % 4 === 0 ||
        (convInfo.inChannels === 3 && convInfo.padInfo.type === 'VALID')) &&
        convInfo.outChannels % 4 === 0 && convInfo.outChannels >= 64) {
        program = new Conv2DMMVec4Program(convInfo);
        if (program.useWgsl === true) {
            const dimAOuter = convInfo.outShape[1] * convInfo.outShape[2];
            const dimBOuter = convInfo.outShape[3];
            const dimInner = convInfo.filterHeight * convInfo.filterWidth * convInfo.inShape[3];
            dimensions.push({ type: 'uint32', data: [dimAOuter] }, { type: 'uint32', data: [dimBOuter] }, { type: 'uint32', data: [dimInner] });
        }
    }
    else {
        program = new Conv2DMMProgram(convInfo);
    }
    return backend.runWebGPUProgram(program, [x, filter], x.dtype, dimensions);
}
const conv2DConfig = {
    kernelName: Conv2D,
    backendName: 'webgpu',
    kernelFunc: conv2d
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Conv2DDerInputMMProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'ivec2 filterDims, pads, stride; ivec4 outBackprop;';
        this.outputShape = convInfo.inShape;
        util.assert(convInfo.dataFormat === 'channelsLast', () => 'TODO: NCHW is unimplemented');
        this.dispatchLayout = { x: [3], y: [1, 2], z: [0] };
        this.workGroupSize =
            computeWorkGroupSizeForConv2d(this.dispatchLayout, this.outputShape);
        this.elementsPerThread =
            computeWorkPerThreadForConv2d(this.dispatchLayout, this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, this.elementsPerThread);
        this.shaderKey = `conv2DDerInputMM_${this.elementsPerThread}`;
    }
    getUserCode() {
        const matMulSource = makeMatMulPackedSource(this.elementsPerThread);
        const readASnippet = `
    int outRow = row / outShape[2];
    int outCol = row % outShape[2];

    int WRow = col / (filterDims[1] * outBackprop[3]);
    int WCol = (col / outBackprop[3]) % filterDims[1];
    float xR = float(outRow - pads[0] + WRow) / float(stride[0]);
    float xC = float(outCol - pads[1] + WCol) / float(stride[1]);
    if (xR < 0.0 || xR >= float(outBackprop[1]) || fract(xR) > 0.0) {
      return 0;
    }
    if (xC < 0.0 || xC >= float(outBackprop[2]) || fract(xC) > 0.0) {
      return 0;
    }
    ivec4 coord = ivec4(
        batch,
        int(xR),
        int(xC),
        col % outBackprop[3]);
    return x[getFlatIndex(coord, xShape)];`;
        const sampleA = `if (row < dimAOuter && col < dimInner) {
      ${readASnippet}
    } else {
      return 0;
    }`;
        const userCode = `
    ${matMulSource}

    int batch;
    int dimAOuter = outShape[1] * outShape[2];
    int dimBOuter = outShape[3];
    int dimInner = filterDims[0] * filterDims[1] * outBackprop[3];

    float mm_readA(int row, int col) {
      ${sampleA}
    }

    float mm_readB(int row, int col) {
      if (row < dimInner && col < dimBOuter)
      {
        int WRow = row / (filterDims[1] * outBackprop[3]);
        int WCol = (row / outBackprop[3]) % filterDims[1];
        ivec4 coord = ivec4(
            filterDims.x - 1 - WRow,
            filterDims.y - 1 - WCol,
            col,
            row % outBackprop[3]);
        return W[getFlatIndex(coord, wShape)];
      } else
      {
        return 0;
      }
    }

    void mm_write(int row, int col, float value) {
      ivec4 outCoord = ivec4(
          batch,
          row / outShape[2],
          row % outShape[2],
          col);
      result[getFlatIndex(outCoord, outShape)] = value;
    }

    void main() {
      batch = int(gl_GlobalInvocationID.z);

      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
  `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Conv2DDerInputProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'W'];
        this.uniforms = 'ivec2 filterDims, pads, stride; ivec4 outBackprop;';
        this.workGroupSize = [64, 1, 1];
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
        this.shaderKey = `conv2DDerInput_${this.isChannelsLast}`;
    }
    getUserCode() {
        const rowDim = this.isChannelsLast ? 1 : 2;
        const colDim = this.isChannelsLast ? 2 : 3;
        const channelDim = this.isChannelsLast ? 3 : 1;
        return `
    void main() {
      ivec4 coords = getOutputCoords();
      if (coordsInBounds(coords, outShape)) {
        int batch = coords[0];
        int d1 = coords[${channelDim}];

        ivec2 dyCorner = ivec2(coords[${rowDim}], coords[${colDim}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < filterDims.x; wR++) {
          float dyR = float(dyRCorner + wR) / float(stride.x);

          if (dyR < 0.0 || dyR >= float(outBackprop[1]) || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = filterDims.x - 1 - wR;

          for (int wC = 0; wC < filterDims.y; wC++) {
            float dyC = float(dyCCorner + wC) / float(stride.y);

            if (dyC < 0.0 || dyC >= float(outBackprop[2]) ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = filterDims.y - 1 - wC;

            for (int d2 = 0; d2 < outBackprop[3]; d2++) {

              if (${this.isChannelsLast}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(coords[0], coords[1], coords[2], coords[3], dotProd);
      }
    }
  `;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function conv2DBackpropInput(args) {
    const { inputs, backend, attrs } = args;
    const { dy, filter } = inputs;
    const { inputShape, strides, pad, dataFormat, dimRoundingMode } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(inputShape, filter.shape, strides, 1 /* dilations */, pad, dimRoundingMode, false, $dataFormat);
    let program;
    if (env().getBool('WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE')) {
        // Keep Conv2DDerInputProgram for reference.
        program = new Conv2DDerInputProgram(convInfo);
    }
    else {
        program = new Conv2DDerInputMMProgram(convInfo);
    }
    const dimensions = [
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        {
            type: 'int32',
            data: [
                convInfo.filterHeight - 1 - convInfo.padInfo.top,
                convInfo.filterWidth - 1 - convInfo.padInfo.left
            ]
        },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        {
            type: 'int32',
            data: [
                convInfo.batchSize, convInfo.outHeight, convInfo.outWidth,
                convInfo.outChannels
            ]
        },
    ];
    return backend.runWebGPUProgram(program, [dy, filter], 'float32', dimensions);
}
const conv2DBackpropInputConfig = {
    kernelName: Conv2DBackpropInput,
    backendName: 'webgpu',
    kernelFunc: conv2DBackpropInput,
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class CropAndResizeProgram {
    constructor(channnel, boxShape, cropSize, method) {
        this.variableNames = ['Image', 'Boxes', 'BoxInd'];
        this.uniforms = 'float extrapolationValue;';
        this.uniformsWgsl = 'extrapolationValue : f32;';
        this.workGroupSize = [64, 1, 1];
        const [numBoxes,] = boxShape;
        this.outputShape = [numBoxes, cropSize[0], cropSize[1], channnel];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.methodId = method === 'bilinear' ? 1 : 0;
        this.cropHeightBiggerThan1 = this.outputShape[1] > 1;
        this.cropWidthBiggerThan1 = this.outputShape[2] > 1;
        this.shaderKey = `cropAndResize_${this.methodId}_${this.cropHeightBiggerThan1}_${this.cropWidthBiggerThan1}`;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const [inputHeightFloat, inputWidthFloat] = [`float(imageShape[1] - 1)`, `float(imageShape[2] - 1)`];
        const [heightRatio, heightScale, inY] = this.cropHeightBiggerThan1 ?
            [
                `(${inputHeightFloat} / float(outShape[1] - 1))`,
                '(y2-y1) * height_ratio',
                `y1*${inputHeightFloat} + float(y)*(height_scale)`,
            ] :
            [
                '0.0',
                '0.0',
                `0.5 * (y1+y2) * ${inputHeightFloat}`,
            ];
        const [widthRatio, widthScale, inX] = this.cropWidthBiggerThan1 ?
            [
                `(${inputWidthFloat} / float(outShape[2] - 1))`,
                '(x2-x1) * width_ratio',
                `x1*${inputWidthFloat} + float(x)*(width_scale)`,
            ] :
            [
                '0.0',
                '0.0',
                `0.5 * (x1+x2) * ${inputWidthFloat}`,
            ];
        // Reference implementation
        // tslint:disable-next-line:max-line-length
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op_gpu.cu.cc
        const userCode = `
      void writeResult(ivec4 coords,float value) {
        if (coordsInBounds(coords, outShape)) {
          setOutput(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
      void main() {
        const float height_ratio = float(${heightRatio});
        const float width_ratio = float(${widthRatio});
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];
        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);
        // get image in batch index
        int bInd = int(round(getBoxInd(b)));
        if(bInd < 0 || bInd >= outShape[0]) {
          return;
        }
        float height_scale = ${heightScale};
        float width_scale = ${widthScale};
        float in_y = ${inY};
        if( in_y < 0.0 || in_y > ${inputHeightFloat} ) {
          writeResult(coords,extrapolationValue);
          return;
        }
        float in_x = ${inX};
        if( in_x < 0.0 || in_x > ${inputWidthFloat} ) {
          writeResult(coords,extrapolationValue);
          return;
        }
        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${this.methodId} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));
          float topLeft = getImage(bInd, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(bInd, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(bInd, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(bInd, sourceCeilCR.y, sourceCeilCR.x, d);
          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);
          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          writeResult(coords,newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(
            bInd, sourceNearestCR.y, sourceNearestCR.x, d);
          writeResult(coords,newValue);
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const [inputHeightFloat, inputWidthFloat] = [
            `f32(uniforms.imageShape[1] - 1u)`, `f32(uniforms.imageShape[2] - 1u)`
        ];
        const [heightRatio, heightScale, inY] = this.cropHeightBiggerThan1 ?
            [
                `(${inputHeightFloat} / f32(uniforms.outShape[1] - 1u))`,
                '(y2-y1) * height_ratio',
                `y1*${inputHeightFloat} + f32(y)*(height_scale)`,
            ] :
            [
                '0.0',
                '0.0',
                `0.5 * (y1+y2) * ${inputHeightFloat}`,
            ];
        const [widthRatio, widthScale, inX] = this.cropWidthBiggerThan1 ?
            [
                `(${inputWidthFloat} / f32(uniforms.outShape[2] - 1u))`,
                '(x2-x1) * width_ratio',
                `x1*${inputWidthFloat} + f32(x)*(width_scale)`,
            ] :
            [
                '0.0',
                '0.0',
                `0.5 * (x1+x2) * ${inputWidthFloat}`,
            ];
        // Reference implementation
        // tslint:disable-next-line:max-line-length
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op_gpu.cu.cc
        const userCode = `
      fn writeResult(coords : vec4<u32>, value : f32) {
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          setOutput(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let height_ratio = f32(${heightRatio});
        let width_ratio = f32(${widthRatio});
        let coords = getOutputCoords(globalId, index);
        let b = coords[0];
        let y = coords[1];
        let x = coords[2];
        let d = coords[3];
        // get box vals
        let y1 = getBoxes(b, 0u);
        let x1 = getBoxes(b, 1u);
        let y2 = getBoxes(b, 2u);
        let x2 = getBoxes(b, 3u);
        // get image in batch index
        let bInd = i32(round(getBoxInd(b)));
        if(bInd < 0 || bInd >= i32(uniforms.outShape[0])) {
          return;
        }
        let height_scale = ${heightScale};
        let width_scale = ${widthScale};
        let in_y = ${inY};
        if( in_y < 0.0 || in_y > ${inputHeightFloat} ) {
          writeResult(coords, uniforms.extrapolationValue);
          return;
        }
        let in_x = ${inX};
        if( in_x < 0.0 || in_x > ${inputWidthFloat} ) {
          writeResult(coords, uniforms.extrapolationValue);
          return;
        }
        let sourceFracIndexCR = vec2<f32>(in_x,in_y);
        if(${this.methodId} == 1) {
          // Compute the four integer indices.
          let sourceFloorCR = vec2<i32>(sourceFracIndexCR);
          let sourceCeilCR = vec2<i32>(ceil(sourceFracIndexCR));
          let topLeft = getImage(u32(bInd), u32(sourceFloorCR.y), u32(sourceFloorCR.x), d);
          let bottomLeft = getImage(u32(bInd), u32(sourceCeilCR.y), u32(sourceFloorCR.x), d);
          let topRight = getImage(u32(bInd), u32(sourceFloorCR.y), u32(sourceCeilCR.x), d);
          let bottomRight = getImage(u32(bInd), u32(sourceCeilCR.y), u32(sourceCeilCR.x), d);
          let fracCR = sourceFracIndexCR - vec2<f32>(sourceFloorCR);
          let top = topLeft + (topRight - topLeft) * fracCR.x;
          let bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          let newValue = top + (bottom - top) * fracCR.y;
          writeResult(coords, newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          let sourceNearestCR = vec2<i32>(floor(
            sourceFracIndexCR + vec2<f32>(0.5,0.5)));
          let newValue = getImage(
            u32(bInd), u32(sourceNearestCR.y), u32(sourceNearestCR.x), d);
          writeResult(coords,newValue);
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const cropAndResize = (args) => {
    const { inputs, backend, attrs } = args;
    const { image, boxes, boxInd } = inputs;
    const { cropSize, method, extrapolationValue } = attrs;
    const program = new CropAndResizeProgram(image.shape[3], boxes.shape, cropSize, method);
    const uniformData = [{ type: 'float32', data: [extrapolationValue] }];
    return backend.runWebGPUProgram(program, [image, boxes, boxInd], 'float32', uniformData);
};
const cropAndResizeConfig = {
    kernelName: CropAndResize,
    backendName: 'webgpu',
    kernelFunc: cropAndResize
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class DepthwiseConv2D3x3Program {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivation = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'ivec2 pad, stride, dilation, inDims;';
        this.uniformsWgsl = 'pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>; inDims : vec2<u32>;';
        this.workGroupSize = [4, 4, 4];
        this.isVec4 = true;
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = { x: [0, 1], y: [2], z: [3] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [1, 4, 4]);
        util.assert(convInfo.dataFormat === 'channelsLast', () => 'TODO: NCHW is unimplemented');
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivation) {
            this.variableNames.push('preluActivationWeights');
        }
        this.convInfo = convInfo;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivation = hasPreluActivation;
        this.shaderKey = `depthwise3x3_${activation}`;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation, this.isVec4);
            if (this.hasPreluActivation) {
                activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords(coords);
          ${activationOp}
        }`;
            }
            else {
                activationSnippet = `
        vec4 activation(vec4 a) {
            ${activationOp}
          }
        `;
            }
            applyActivationSnippet = `dotProd[i] = activation(dotProd[i]);`;
        }
        const addBiasSnippet = this.addBias ? 'dotProd[i] += getBiasAtOutCoords(coords);' : '';
        const userCode = `
      ${activationSnippet}

      void main() {
        int batch = 0;
        int r = int(gl_GlobalInvocationID.x);
        int c = int(gl_GlobalInvocationID.y) * 4;
        int d2= int(gl_GlobalInvocationID.z) * 4;
        ivec2 xRCCorner = ivec2(r, c) * stride - pad;
        int d1 = d2;
        int q = 0;

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        vec4 wVals[9];
        wVals[0] = getW(0, 0, d1, q);
        wVals[1] = getW(0, 1, d1, q);
        wVals[2] = getW(0, 2, d1, q);
        wVals[3] = getW(1, 0, d1, q);
        wVals[4] = getW(1, 1, d1, q);
        wVals[5] = getW(1, 2, d1, q);
        wVals[6] = getW(2, 0, d1, q);
        wVals[7] = getW(2, 1, d1, q);
        wVals[8] = getW(2, 2, d1, q);

        vec4 xVals[3][6];
        for (int wR = 0; wR < 3; wR++) {
          int xR = xRCorner + wR * dilation[0];
          for (int wC = 0; wC < 6; wC++) {
            int xC = xCCorner + wC * dilation[1];
            if (xR < 0 || xR >= inDims[0] || xC < 0 || xC >= inDims[1]) {
              xVals[wR][wC] = vec4(0.0, 0.0, 0.0, 0.0);
            } else {
              xVals[wR][wC] = getX(batch, xR, xC, d1);
            }
          }
        }

        vec4 dotProd[4];
        dotProd[0] = vec4(0.0, 0.0, 0.0, 0.0);
        dotProd[1] = vec4(0.0, 0.0, 0.0, 0.0);
        dotProd[2] = vec4(0.0, 0.0, 0.0, 0.0);
        dotProd[3] = vec4(0.0, 0.0, 0.0, 0.0);

        for (int wR = 0; wR < 3; wR++) {
          for (int wC = 0; wC < 3; wC++) {
            int indexW = wR * 3 + wC;
            dotProd[0] += xVals[wR][0 + wC] * wVals[indexW];
            dotProd[1] += xVals[wR][1 + wC] * wVals[indexW];
            dotProd[2] += xVals[wR][2 + wC] * wVals[indexW];
            dotProd[3] += xVals[wR][3 + wC] * wVals[indexW];
          }
        }

        for (int i = 0; i < 4; i++)
        {
          ivec4 coords = ivec4(batch, r, c + i, d2);
          if (coordsInBounds(coords, outShape)) {
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutput(coords[0], coords[1], coords[2], coords[3], dotProd[i]);
          }
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation, this.isVec4, this.useWgsl);
            if (this.hasPreluActivation) {
                activationSnippet =
                    `fn activation(a : vec4<f32>, globalId : vec3<u32>, globalIndex : u32) -> vec4<f32> {
          let b = getPreluActivationWeightsAtOutCoordsByGlobalId(globalId, globalIndex);
          ${activationOp}
        }`;
            }
            else {
                activationSnippet = `
        fn activation(a : vec4<f32>, globalId : vec3<u32>, globalIndex : u32) -> vec4<f32> {
            ${activationOp}
          }
        `;
            }
            applyActivationSnippet =
                `dotProd[i] = activation(dotProd[i], globalId, index);`;
        }
        const addBiasSnippet = this.addBias ?
            'dotProd[i] = dotProd[i] + getBiasAtOutCoordsByCoords(coords);' :
            '';
        const userCode = `
      ${activationSnippet}

      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let batch = 0u;
        let r = globalId.x;
        let c = globalId.y * 4u;
        let d2 = globalId.z * 4u;
        let xRCCorner = vec2<i32>(vec2<u32>(r, c) * uniforms.stride - uniforms.pad);
        let d1 = d2;
        let q = 0u;

        let xRCorner = xRCCorner.x;
        let xCCorner = xRCCorner.y;

        var wVals : array<vec4<f32>, 9>;
        wVals[0] = getW(0u, 0u, d1, q);
        wVals[1] = getW(0u, 1u, d1, q);
        wVals[2] = getW(0u, 2u, d1, q);
        wVals[3] = getW(1u, 0u, d1, q);
        wVals[4] = getW(1u, 1u, d1, q);
        wVals[5] = getW(1u, 2u, d1, q);
        wVals[6] = getW(2u, 0u, d1, q);
        wVals[7] = getW(2u, 1u, d1, q);
        wVals[8] = getW(2u, 2u, d1, q);

        var xVals : array<array<vec4<f32>, 6>, 3>;
        for (var wR = 0u; wR < 3u; wR = wR + 1u) {
          let xR = xRCorner + i32(wR * uniforms.dilation[0]);
          for (var wC = 0u; wC < 6u; wC = wC + 1u) {
            let xC = xCCorner + i32(wC * uniforms.dilation[1]);
            if (xR < 0 || xR >= i32(uniforms.inDims[0]) || xC < 0 || xC >= i32(uniforms.inDims[1])) {
              xVals[wR][wC] = vec4<f32>(0.0);
            } else {
              xVals[wR][wC] = getX(batch, u32(xR), u32(xC), d1);
            }
          }
        }

        var dotProd : array<vec4<f32>, 4>;
        dotProd[0] = vec4<f32>(0.0);
        dotProd[1] = vec4<f32>(0.0);
        dotProd[2] = vec4<f32>(0.0);
        dotProd[3] = vec4<f32>(0.0);

        for (var wR = 0u; wR < 3u; wR = wR + 1u) {
          for (var wC = 0u; wC < 3u; wC = wC + 1u) {
            let indexW = wR * 3u + wC;
            dotProd[0] = dotProd[0] + xVals[wR][0u + wC] * wVals[indexW];
            dotProd[1] = dotProd[1] + xVals[wR][1u + wC] * wVals[indexW];
            dotProd[2] = dotProd[2] + xVals[wR][2u + wC] * wVals[indexW];
            dotProd[3] = dotProd[3] + xVals[wR][3u + wC] * wVals[indexW];
          }
        }

        for (var i = 0u; i < 4u; i = i + 1u) {
          let coords = vec4<u32>(batch, r, c + i, d2);
          if (coordsInBounds4D(coords, uniforms.outShape)) {
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutput(coords[0], coords[1], coords[2], coords[3], dotProd[i]);
          }
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class DepthwiseConv2DProgram {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivation = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'ivec2 pad, stride, dilation, inDims;';
        this.uniformsWgsl = `pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>; inDims : vec2<u32>;`;
        // This is an experimental value.
        this.workGroupSize = [256, 1, 1];
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        util.assert(convInfo.dataFormat === 'channelsLast', () => 'TODO: NCHW is unimplemented');
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivation) {
            this.variableNames.push('preluActivationWeights');
        }
        this.convInfo = convInfo;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivation = hasPreluActivation;
        this.useWgsl = getUseWgsl();
        this.shaderKey = `depthwise_${this.convInfo.filterHeight}_${this.convInfo.filterWidth}_${this.activation}_${this.convInfo.outChannels / this.convInfo.inChannels}`;
    }
    getUserCode() {
        const channelMul = this.convInfo.outChannels / this.convInfo.inChannels;
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation);
            if (this.hasPreluActivation) {
                activationSnippet = `float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${activationOp}
        }`;
            }
            else {
                activationSnippet = `
          float activation(float a) {
            ${activationOp}
          }
        `;
            }
            applyActivationSnippet = `dotProd = activation(dotProd);`;
        }
        const addBiasSnippet = this.addBias ? 'dotProd += getBiasAtOutCoords();' : '';
        const userCode = `
      ${activationSnippet}

      void writeResult(int batch, int row, int col, int chan, float value) {
        ivec4 coord = ivec4(batch, row, col, chan);
        if (coordsInBounds(coord, outShape)) {
          setOutput(batch, row, col, chan, value);
        }
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        ivec2 xRCCorner = coords.yz * stride - pad;
        int d2 = coords[3];
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        int inputRowStart = xRCCorner.x;
        int inputColStart = xRCCorner.y;
        int inputRowEnd = inputRowStart + ${this.convInfo.filterHeight} * dilation[0];
        int inputColEnd = inputColStart + ${this.convInfo.filterWidth} * dilation[1];

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        // Extract if checking out of for loop for performance.
        if (inputRowStart >= 0 && inputColStart >= 0 &&
          inputRowEnd < inDims[0] && inputColEnd < inDims[1])
          {
            // Here using a constant value |this.convInfo.filterHeight| instead
            // of uniform value is in order to loop unrolling.
            for (int wR = 0; wR < ${this.convInfo.filterHeight}; wR++) {
              int xR = inputRowStart + wR * dilation[0];

              for (int wC = 0; wC < ${this.convInfo.filterWidth}; wC++) {
                int xC = inputColStart + wC * dilation[1];

                float xVal = getX(batch, xR, xC, d1);
                float wVal = getW(wR, wC, d1, q);
                dotProd += xVal * wVal;
              }
            }
          } else {
            for (int wR = 0; wR < ${this.convInfo.filterHeight}; wR++) {
              int xR = inputRowStart + wR * dilation[0];

              if (xR < 0 || xR >= inDims[0]) {
                continue;
              }

              for (int wC = 0; wC < ${this.convInfo.filterWidth}; wC++) {
                int xC = inputColStart + wC * dilation[1];

                if (xC < 0 || xC >= inDims[1]) {
                  continue;
                }

                float xVal = getX(batch, xR, xC, d1);
                float wVal = getW(wR, wC, d1, q);
                dotProd += xVal * wVal;
              }
            }
          }

        ${addBiasSnippet}
        ${applyActivationSnippet}
        writeResult(batch, coords[1], coords[2], d2, dotProd);
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const channelMul = this.convInfo.outChannels / this.convInfo.inChannels;
        let activationSnippet = '', applyActivationSnippet = '';
        if (this.activation) {
            const activationOp = mapActivationToShaderProgram(this.activation, false, this.useWgsl);
            if (this.hasPreluActivation) {
                activationSnippet =
                    `fn activation(a : f32, globalId : vec3<u32>, index : u32) -> f32 {
          let b = getPreluActivationWeightsAtOutCoordsByGlobalId(globalId, index);
          ${activationOp}
        }`;
            }
            else {
                activationSnippet = `
          fn activation(a : f32, globalId : vec3<u32>, index : u32) -> f32 {
            ${activationOp}
          }
        `;
            }
            applyActivationSnippet =
                `dotProd = activation(dotProd, globalId, index);`;
        }
        const addBiasSnippet = this.addBias ?
            'dotProd = dotProd + getBiasAtOutCoordsByGlobalId(globalId, index);' :
            '';
        const userCode = `
      ${activationSnippet}

      fn writeResult(batch : u32, row : u32, col : u32, chan : u32, value : f32) {
        let coord = vec4<u32>(batch, row, col, chan);
        if (coordsInBounds4D(coord, uniforms.outShape)) {
          setOutput(batch, row, col, chan, value);
        }
      }

      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coords = getOutputCoords(globalId, index);
        let batch = coords[0];
        let xRCCorner = vec2<i32>(coords.yz * uniforms.stride - uniforms.pad);
        let d2 = coords[3];
        let d1 = d2 / ${channelMul}u;
        let q = d2 - d1 * ${channelMul}u;

        let inputRowStart = xRCCorner.x;
        let inputColStart = xRCCorner.y;
        let inputRowEnd = inputRowStart + i32(${this.convInfo.filterHeight}u * uniforms.dilation[0]);
        let inputColEnd = inputColStart + i32(${this.convInfo.filterWidth}u * uniforms.dilation[1]);

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;

        // Extract if checking out of for loop for performance.
        if (inputRowStart >= 0 && inputColStart >= 0 &&
          inputRowEnd < i32(uniforms.inDims[0]) && inputColEnd < i32(uniforms.inDims[1])) {
            // Here using a constant value |this.convInfo.filterHeight| instead
            // of uniform value is in order to loop unrolling.
            for (var wR = 0u; wR < ${this.convInfo.filterHeight}u; wR = wR + 1u) {
              let xR = inputRowStart + i32(wR * uniforms.dilation[0]);

              for (var wC = 0u; wC < ${this.convInfo.filterWidth}u; wC = wC + 1u) {
                let xC = inputColStart + i32(wC * uniforms.dilation[1]);

                let xVal = getX(batch, u32(xR), u32(xC), d1);
                let wVal = getW(wR, u32(wC), d1, q);
                dotProd = dotProd + xVal * wVal;
              }
            }
          } else {
            for (var wR = 0u; wR < ${this.convInfo.filterHeight}u; wR = wR + 1u) {
              let xR = inputRowStart + i32(wR * uniforms.dilation[0]);

              if (xR < 0 || xR >= i32(uniforms.inDims[0])) {
                continue;
              }

              for (var wC = 0u; wC < ${this.convInfo.filterWidth}u; wC = wC + 1u) {
                let xC = inputColStart + i32(wC * uniforms.dilation[1]);

                if (xC < 0 || xC >= i32(uniforms.inDims[1])) {
                  continue;
                }

                let xVal = getX(batch, u32(xR), u32(xC), d1);
                let wVal = getW(wR, wC, d1, q);
                dotProd = dotProd + xVal * wVal;
              }
            }
          }

        ${addBiasSnippet}
        ${applyActivationSnippet}
        writeResult(batch, coords[1], coords[2], d2, dotProd);
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function depthwiseConv2dNative(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter } = inputs;
    const { strides, pad, dilations, dimRoundingMode } = attrs;
    let $dilations = dilations;
    if ($dilations == null) {
        $dilations = [1, 1];
    }
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, $dilations, pad, dimRoundingMode, true /* depthwise */);
    let program;
    // TODO: To see if we need to relax the limitation. Currently, it's only for
    // filter size 3x3.
    if (convInfo.batchSize === 1 && convInfo.inHeight === convInfo.outHeight &&
        convInfo.inWidth === convInfo.outWidth && convInfo.strideHeight === 1 &&
        convInfo.strideWidth === 1 &&
        convInfo.filterHeight === convInfo.filterWidth &&
        convInfo.inChannels === convInfo.outChannels &&
        convInfo.filterHeight === 3 && convInfo.inChannels % 4 === 0) {
        program = new DepthwiseConv2D3x3Program(convInfo);
    }
    else {
        program = new DepthwiseConv2DProgram(convInfo);
    }
    const dimensions = [
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] }
    ];
    return backend.runWebGPUProgram(program, [x, filter], x.dtype, dimensions);
}
const depthwiseConv2dNativeConfig = {
    kernelName: DepthwiseConv2dNative,
    backendName: 'webgpu',
    kernelFunc: depthwiseConv2dNative,
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const multiplyKernelFunc = binaryKernelFunc({
    opSnippet: BinaryOpType.MUL,
    cpuKernelImpl: multiplyImplCPU,
    supportsComplex: true
});
const multiplyConfig = {
    kernelName: Multiply,
    backendName: 'webgpu',
    kernelFunc: multiplyKernelFunc
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class ReduceProgram {
    constructor(reduceInfo, reduceType, outputDtype) {
        this.variableNames = ['x'];
        this.uniforms = 'int reduceSize;';
        this.uniformsWgsl = 'reduceSize : u32;';
        this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
        const [outputShape,] = backend_util.computeOutAndReduceShapes(this.inputShape, [1]);
        this.outputShape = outputShape.length === 0 ? [1] : outputShape;
        this.reductionFactor = 2;
        // Note that the maximum of workgroup X dimension is 256.
        const xMaxThreads = 256;
        const xThreads = Math.min(Math.ceil(reduceInfo.inSize / this.reductionFactor), xMaxThreads);
        this.workGroupSize = [xThreads, 1, 1];
        this.dispatchLayout = { x: [], y: this.outputShape.map((d, i) => i) };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.reduceType = reduceType;
        this.shaderKey = `reduce_${reduceType}_${outputDtype}`;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const reduceInSharedMemory = this.workGroupSize[0] > 1;
        let reduceOp = ``;
        let initValue = '0.0';
        if (this.reduceType === 'min' || this.reduceType === 'max') {
            reduceOp = `
         if (isnan(candidate)) {
          bestValue = float(NAN);
         } else if (candidate ${this.reduceType === 'min' ? '<' : '>'}
           bestValue)
           {  bestValue = candidate; }`;
            initValue = 'float(x[offset])';
        }
        else if (this.reduceType === 'sum' || this.reduceType === 'mean') {
            reduceOp = ' bestValue += candidate; ';
        }
        else if (this.reduceType === 'prod') {
            reduceOp = ' bestValue *= candidate; ';
            initValue = '1.0';
        }
        const outputSnippet = this.reduceType === 'mean' ?
            `setOutput(flatOutputIndex, bestValue / float(reduceSize));` :
            `setOutput(flatOutputIndex, bestValue);`;
        const sharedMemorySnippet = `
         shared float xBestValues[WorkGroupSize];
       `;
        const sharedMemoryReduceSnippet = `
       xBestValues[gl_LocalInvocationID.x] = bestValue;
       ${this.reduceType === 'sum' || this.reduceType === 'mean' ||
            this.reduceType === 'prod' ?
            `bestValue=${initValue};` :
            ' '}
       int currentSize = WorkGroupSize;
       while (currentSize > 1) {
         barrier();
         for (int w = 0; w < ${this.reductionFactor}; ++w) {
           int i = int(gl_LocalInvocationID.x) * ${this.reductionFactor} + w;
           if (i < currentSize) {
             float candidate = xBestValues[i];
             ${reduceOp}
           }
         }
         barrier();
         xBestValues[gl_LocalInvocationID.x] = bestValue;
         currentSize = DIV_CEIL(currentSize, ${this.reductionFactor});
         ${this.reduceType === 'sum' || this.reduceType === 'mean' ||
            this.reduceType === 'prod' ?
            `if(currentSize > 1) bestValue=${initValue};` :
            ''}
       }
       if (gl_LocalInvocationID.x == 0) {
         ${outputSnippet}
       }
     `;
        const outputCoordsType = getCoordsDataType(this.outputShape.length);
        const userCode = `
       #define DIV_CEIL(x, y) (((x) - 1) / (y) + 1)
       const int WorkGroupSize = int(gl_WorkGroupSize.x);
       ${reduceInSharedMemory ? sharedMemorySnippet : ''}
       int getOffset() {
         const ${outputCoordsType} outputCoords = getOutputCoords();
         int offset = ${this.outputShape.length === 1 ? 'outputCoords' :
            'outputCoords[0]'} * reduceSize;
         return offset;
       }
       void main() {
         const int offset= getOffset();
         float bestValue = ${initValue};
         const int Length = reduceSize;
         const int WorkPerThread = DIV_CEIL(Length, WorkGroupSize);
         for (int w = 0; w < WorkPerThread; ++w) {
           int i = int(gl_GlobalInvocationID.x) * WorkPerThread + w;
           if (i < Length) {
             float candidate = float(x[offset + i]);
             ${reduceOp}
           }
         }
         const int flatOutputIndex = int(gl_GlobalInvocationID.y);
         ${reduceInSharedMemory ? sharedMemoryReduceSnippet : outputSnippet}
       }
     `;
        return userCode;
    }
    getUserCodeWgsl() {
        const reduceInSharedMemory = this.workGroupSize[0] > 1;
        let reduceOp = ``;
        let initValue = '0.0';
        if (this.reduceType === 'min' || this.reduceType === 'max') {
            reduceOp = `
         if (isNanCustom(candidate)) {
          bestValue = uniforms.NAN;
         } elseif (candidate ${this.reduceType === 'min' ? '<' : '>'}
           bestValue)
           {  bestValue = candidate; }`;
            initValue = 'f32(x.numbers[offset])';
        }
        else if (this.reduceType === 'sum' || this.reduceType === 'mean') {
            reduceOp = ' bestValue = bestValue + candidate; ';
        }
        else if (this.reduceType === 'prod') {
            reduceOp = ' bestValue = bestValue * candidate; ';
            initValue = '1.0';
        }
        const outputSnippet = this.reduceType === 'mean' ?
            // tslint:disable-next-line:max-line-length
            `setOutputFlat(flatOutputIndex, bestValue / f32(uniforms.reduceSize));` :
            `setOutputFlat(flatOutputIndex, bestValue);`;
        const sharedMemorySnippet = `
         var<workgroup> xBestValues : array<f32, ${this.workGroupSize[0]}>;
       `;
        const sharedMemoryReduceSnippet = `
       xBestValues[localId.x] = bestValue;
       ${this.reduceType === 'sum' || this.reduceType === 'mean' ||
            this.reduceType === 'prod' ?
            `bestValue = ${initValue};` :
            ' '}
       var currentSize = WorkGroupSize;
       for(; currentSize > 1u;) {
         workgroupBarrier();
         for (var w = 0u; w < ${this.reductionFactor}u; w = w + 1u) {
           let i = localId.x * ${this.reductionFactor}u + w;
           if (i < currentSize) {
             let candidate = xBestValues[i];
             ${reduceOp}
           }
         }
         workgroupBarrier();
         xBestValues[localId.x] = bestValue;
         currentSize = DIV_CEIL(currentSize, ${this.reductionFactor}u);
         ${this.reduceType === 'sum' || this.reduceType === 'mean' ||
            this.reduceType === 'prod' ?
            `if(currentSize > 1u) { bestValue = ${initValue}; }` :
            ''}
       }
       if (localId.x == 0u) {
         ${outputSnippet}
       }
     `;
        const userCode = `
       fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
       }
       let WorkGroupSize = ${this.workGroupSize[0]}u;
       ${reduceInSharedMemory ? sharedMemorySnippet : ''}
       fn getOffset(globalId : vec3<u32>, index : u32) -> u32 {
         let outputCoords = getOutputCoords(globalId, index);
         let offset = ${this.outputShape.length === 1 ?
            'outputCoords' :
            'outputCoords[0]'} * uniforms.reduceSize;
         return offset;
       }
       ${getMainHeaderStringWgsl(this.workGroupSize)} {
         ${getGlobalIndexStringWgsl(this.workGroupSize)}
         let offset= getOffset(globalId, index);
         var bestValue = ${initValue};
         let Length = uniforms.reduceSize;
         let WorkPerThread = DIV_CEIL(Length, WorkGroupSize);
         for (var w = 0u; w < WorkPerThread; w = w + 1u) {
           let i = globalId.x * WorkPerThread + w;
           if (i < Length) {
             let candidate = f32(x.numbers[offset + i]);
             ${reduceOp}
           }
         }
         let flatOutputIndex = globalId.y;
         ${reduceInSharedMemory ? sharedMemoryReduceSnippet : outputSnippet}
       }
     `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function reduce(x, axis, keepDims, reduceType, backend) {
    const xRank = x.shape.length;
    const toDispose = [];
    const origAxes = util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
    let input = x;
    if (permutedAxes != null) {
        input = transpose({ inputs: { x }, attrs: { perm: permutedAxes }, backend });
        axes = backend_util.getInnerMostAxes(axes.length, xRank);
        toDispose.push(input);
    }
    backend_util.assertAxesAreInnerMostDims(reduceType, axes, xRank);
    const [reduceOutShape, reduceShape] = backend_util.computeOutAndReduceShapes(input.shape, axes);
    let resOutShape = reduceOutShape;
    if (keepDims) {
        // rather than reshape at the end, set the target shape here.
        resOutShape = backend_util.expandShapeToKeepDim(reduceOutShape, origAxes);
    }
    let res;
    if ((reduceType === 'max' || reduceType === 'prod') &&
        backend.shouldExecuteOnCPU([input])) {
        const xVals = backend.tensorMap.get(input.dataId).values;
        switch (reduceType) {
            case 'max':
                const outValues = maxImplCPU(xVals, util.sizeFromShape(reduceShape), resOutShape, x.dtype);
                res = backend.makeTensorInfo(resOutShape, x.dtype, outValues);
                break;
            case 'prod':
                const { outVals, outShape, outDtype } = prodImplCPU(input.shape, input.dtype, xVals, axes);
                res = backend.makeTensorInfo(outShape, outDtype, outVals);
                break;
            default:
                throw new Error(`${reduceType} CPU implementation is not yet supported.`);
        }
    }
    else {
        const inSize = util.sizeFromShape(reduceShape);
        const xSize = util.sizeFromShape(input.shape);
        const batchSize = xSize / inSize;
        const reduceInfo = { windowSize: inSize, inSize, batchSize, outSize: 1 };
        const dtype = reduceType === 'mean' ? 'float32' : sumOutType(x.dtype);
        const uniformData = [
            { type: 'int32', data: [inSize] },
        ];
        const program = new ReduceProgram(reduceInfo, reduceType, dtype);
        const reduced = backend.runWebGPUProgram(program, [input], dtype, uniformData);
        toDispose.push(reduced);
        res = reshape({ inputs: { x: reduced }, attrs: { shape: resOutShape }, backend });
    }
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return res;
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function sum(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    return reduce(x, axis, keepDims, 'sum', backend);
}
const sumConfig = {
    kernelName: Sum,
    backendName: 'webgpu',
    kernelFunc: sum
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function einsum(args) {
    const { inputs, backend, attrs } = args;
    const { equation } = attrs;
    const tensors = inputs;
    const { allDims, summedDims, idDims } = backend_util.decodeEinsumEquation(equation, tensors.length);
    backend_util.checkEinsumDimSizes(allDims.length, idDims, tensors);
    const { path, steps } = backend_util.getEinsumComputePath(summedDims, idDims);
    const nSteps = steps.length;
    let out = null;
    let numDimsRemaining = allDims.length;
    const tensorsToDispose = [];
    for (let i = 0; i < nSteps; ++i) {
        for (const idTerm of steps[i]) {
            const { permutationIndices: perm, expandDims: dimsToExpand } = backend_util.getEinsumPermutation(numDimsRemaining, idDims[idTerm]);
            let x;
            if (backend_util.isIdentityPermutation(perm)) {
                x = tensors[idTerm];
            }
            else {
                x = transpose({ inputs: { x: tensors[idTerm] }, backend, attrs: { perm } });
                tensorsToDispose.push(x);
            }
            const targetShape = x.shape.slice();
            for (let k = 0; k < dimsToExpand.length; ++k) {
                targetShape.splice(dimsToExpand[k], 0, 1);
            }
            if (!util.arraysEqual(x.shape, targetShape)) {
                x = reshape({ inputs: { x }, backend, attrs: { shape: targetShape } });
                tensorsToDispose.push(x);
            }
            if (out === null) {
                out = x;
            }
            else {
                // tslint:disable-next-line: no-unnecessary-type-assertion
                out =
                    multiplyKernelFunc({ inputs: { a: x, b: out }, backend });
                tensorsToDispose.push(out);
            }
        }
        if (i < nSteps - 1) {
            if (path[i] >= 0) {
                out = sum({
                    inputs: { x: out },
                    backend,
                    attrs: {
                        axis: path[i] - (allDims.length - numDimsRemaining),
                        keepDims: false
                    }
                });
                tensorsToDispose.push(out);
            }
            numDimsRemaining--;
        }
    }
    // Clean up intermediate tensors.
    for (const tensorInfo of tensorsToDispose) {
        if (tensorInfo === out) {
            continue;
        }
        backend.disposeData(tensorInfo.dataId);
    }
    return out;
}
const einsumConfig = {
    kernelName: Einsum,
    backendName: 'webgpu',
    kernelFunc: einsum
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const elu = unaryKernelFunc({ opType: UnaryOpType.ELU });
const eluConfig = {
    kernelName: Elu,
    backendName: 'webgpu',
    kernelFunc: elu
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const equal = binaryKernelFunc({ opSnippet: BinaryOpType.EQUAL, dtype: 'bool', cpuKernelImpl: equalImplCPU });
const equalConfig = {
    kernelName: Equal,
    backendName: 'webgpu',
    kernelFunc: equal
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const exp = unaryKernelFunc({ opType: UnaryOpType.EXP, cpuKernelImpl: expImplCPU });
const expConfig = {
    kernelName: Exp,
    backendName: 'webgpu',
    kernelFunc: exp
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function expandDims(args) {
    const { inputs, attrs, backend } = args;
    const { dim } = attrs;
    const { input } = inputs;
    const inputRank = input.shape.length;
    const newShape = input.shape.slice();
    let $dim = dim;
    if (dim < 0) {
        // Negative value is counted from the tail of rank.
        util.assert(-(inputRank + 1) <= dim, () => `Axis must be in the interval [${-(inputRank + 1)}, ${inputRank}]`);
        $dim = inputRank + dim + 1;
    }
    newShape.splice($dim, 0, 1);
    return reshape({ inputs: { x: input }, backend, attrs: { shape: newShape } });
}
const expandDimsConfig = {
    kernelName: ExpandDims,
    backendName: 'webgpu',
    kernelFunc: expandDims,
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const expm1 = unaryKernelFunc({ opType: UnaryOpType.EXPM1, cpuKernelImpl: expm1ImplCPU });
const expm1Config = {
    kernelName: Expm1,
    backendName: 'webgpu',
    kernelFunc: expm1
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class FillProgram {
    constructor(shape) {
        this.variableNames = [];
        this.outputShape = [];
        this.uniforms = 'float value;';
        this.uniformsWgsl = 'value : f32;';
        this.workPerThread = 4;
        this.workGroupSize = [16, 1, 1];
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.shaderKey = 'fill';
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const userCode = `
    void main() {
      int index = getGlobalIndex();
      for (int i = 0; i < ${this.workPerThread}; i++) {
        int flatIndex = index * ${this.workPerThread} + i;
        if (flatIndex < size) {
          setOutput(flatIndex, float(value));
        }
      }
    }
  `;
        return userCode;
    }
    getUserCodeWgsl() {
        const userCode = `
    ${getMainHeaderStringWgsl(this.workGroupSize)} {
      ${getGlobalIndexStringWgsl(this.workGroupSize)}
      for (var i = 0u; i < ${this.workPerThread}u; i = i + 1u) {
        let flatIndex = index * ${this.workPerThread}u + i;
        if (flatIndex < uniforms.size) {
          setOutputFlat(flatIndex, uniforms.value);
        }
      }
    }
  `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function fill(args) {
    const { backend, attrs } = args;
    const { shape, value } = attrs;
    let { dtype } = attrs;
    dtype = dtype || util.inferDtype(value);
    if (dtype === 'string') {
        // String type should be handled in CPU memory.
        const values = util.getArrayFromDType(dtype, util.sizeFromShape(shape));
        values.fill(value);
        return backend.makeTensorInfo(shape, dtype, values);
    }
    else {
        const program = new FillProgram(shape);
        const uniformData = [{ type: 'float32', data: [value] }];
        return backend.runWebGPUProgram(program, [], dtype, uniformData);
    }
}
const fillConfig = {
    kernelName: Fill,
    backendName: 'webgpu',
    kernelFunc: fill
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const floor = unaryKernelFunc({ opType: UnaryOpType.FLOOR, cpuKernelImpl: floorImplCPU });
const floorConfig = {
    kernelName: Floor,
    backendName: 'webgpu',
    kernelFunc: floor
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const floorDiv = binaryKernelFunc({ opSnippet: BinaryOpType.INT_DIV, dtype: 'int32' });
const floorDivConfig = {
    kernelName: FloorDiv,
    backendName: 'webgpu',
    kernelFunc: floorDiv
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use backend file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function fromPixelsExternalImage(args) {
    const { externalImage, backend, attrs, outShape, useImport } = args;
    const { numChannels } = attrs;
    const size = util.sizeFromShape(outShape);
    const strides = util.computeStrides(outShape);
    const uniformData = [size, numChannels, ...strides];
    const output = backend.makeTensorInfo(outShape, 'int32');
    const program = backend.getFromPixelsProgram(useImport ? 'import' : 'copyExternal');
    program.updateOutputShape(outShape);
    // Different outShape will affect preprocessor result,
    // e.g. getCoordsFromFlatIndex. FromPixelsImageExternalImage needs
    // to recompile the pipeline to get the correct result.
    // FromPixelsExternalImage leverages webgpu backend pipeline
    // cache system to avoid useless recompile.
    const outputShapes = [output.shape];
    const outputTypes = [output.dtype];
    const key = makeShaderKey(program, outputShapes, outputTypes);
    const layout = program.getLayout(backend.device);
    const pipeline = backend.getAndSavePipeline(key, () => {
        return compileProgram(backend.glslang, backend.device, program, layout.pipelineLayout, [], output, true);
    });
    program.setPipeline(pipeline);
    if (!useImport) {
        backend.queue.copyExternalImageToTexture({ source: externalImage, origin: { x: 0, y: 0 } }, {
            texture: program.makeInputTexture(backend.device, outShape[1], outShape[0])
        }, [outShape[1], outShape[0]]);
    }
    const info = backend.tensorMap.get(output.dataId);
    info.bufferInfo.buffer = backend.acquireBuffer(info.bufferInfo.byteSize);
    program.setUniform(backend.device, uniformData);
    let externalResource;
    if (useImport) {
        const externalTextureDescriptor = {
            source: externalImage
        };
        externalResource =
            backend.device.importExternalTexture(externalTextureDescriptor);
    }
    else {
        externalResource = program.inputTexture.createView();
    }
    backend.runFromPixelsProgram(program, info.bufferInfo.buffer, layout, externalResource, output.dataId);
    return output;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use backend file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const fromPixelsConfig = {
    kernelName: FromPixels,
    backendName: 'webgpu',
    kernelFunc: fromPixels,
};
let fromPixels2DContext;
function fromPixels(args) {
    const { inputs, backend, attrs } = args;
    let { pixels } = inputs;
    const { numChannels } = attrs;
    if (pixels == null) {
        throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
    }
    const isVideo = typeof (HTMLVideoElement) !== 'undefined' &&
        pixels instanceof HTMLVideoElement;
    const isImage = typeof (HTMLImageElement) !== 'undefined' &&
        pixels instanceof HTMLImageElement;
    const isCanvas = typeof (HTMLCanvasElement) !== 'undefined' &&
        pixels instanceof HTMLCanvasElement;
    const isImageBitmap = typeof (ImageBitmap) !== 'undefined' && pixels instanceof ImageBitmap;
    const [width, height] = isVideo ?
        [
            pixels.videoWidth,
            pixels.videoHeight
        ] :
        [pixels.width, pixels.height];
    const outShape = [height, width, numChannels];
    if (env().getBool('WEBGPU_USE_IMPORT')) {
        if (isVideo) {
            return fromPixelsExternalImage({
                externalImage: pixels,
                backend,
                attrs,
                outShape,
                useImport: true
            });
        }
    }
    if (isVideo || isImage) {
        if (fromPixels2DContext == null) {
            fromPixels2DContext = document.createElement('canvas').getContext('2d');
        }
        fromPixels2DContext.canvas.width = width;
        fromPixels2DContext.canvas.height = height;
        fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
        pixels = fromPixels2DContext.canvas;
    }
    if (isImageBitmap || isCanvas || isVideo || isImage) {
        return fromPixelsExternalImage({
            externalImage: pixels,
            backend,
            attrs,
            outShape,
            useImport: false
        });
    }
    // TODO: Encoding should happen on GPU once we no longer have to download
    // image data to the CPU.
    const imageData = pixels.data;
    let pixelArray = imageData;
    if (numChannels != null && numChannels !== 4) {
        pixelArray = new Uint8Array(pixels.width * pixels.height * numChannels);
        const dataLength = imageData.length;
        let j = 0;
        for (let i = 0; i < dataLength; i++) {
            if (i % 4 < numChannels) {
                pixelArray[j++] = imageData[i];
            }
        }
    }
    const output = backend.makeTensorInfo(outShape, 'int32');
    const info = backend.tensorMap.get(output.dataId);
    info.values = new Int32Array(pixelArray);
    backend.maybeReleaseBuffer(output.dataId);
    backend.uploadToGPU(output.dataId);
    return output;
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class BatchNormProgram {
    constructor(xShape, meanShape, varianceShape, offsetShape, scaleShape) {
        this.uniforms = 'float varianceEpsilon;';
        this.uniformsWgsl = 'varianceEpsilon : f32;';
        // This is an experimental value.
        this.workGroupSize = [128, 1, 1];
        this.variableNames = ['x', 'mean', 'variance'];
        backend_util.assertAndGetBroadcastShape(xShape, meanShape);
        backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
        this.outputShape = xShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        if (offsetShape != null) {
            backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
            this.variableNames.push('offset');
        }
        if (scaleShape != null) {
            backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
            this.variableNames.push('scale');
        }
        this.offsetShape = offsetShape;
        this.scaleShape = scaleShape;
        this.shaderKey = 'batchNorm';
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        let offsetSnippet = '0.0';
        if (this.offsetShape != null) {
            offsetSnippet = 'getOffsetAtOutCoords()';
        }
        let scaleSnippet = '1.0';
        if (this.scaleShape != null) {
            scaleSnippet = 'getScaleAtOutCoords()';
        }
        const dim = this.outputShape.length;
        const coordsDataType = getCoordsDataType(dim);
        let setOutput = 'setOutput(coords[0], coords[1], coords[2], coords[3], value);';
        if (dim === 2) {
            setOutput = 'setOutput(coords[0], coords[1], value);';
        }
        if (dim === 3) {
            setOutput = 'setOutput(coords[0], coords[1], coords[2], value);';
        }
        const userCode = `
      void writeResult(${coordsDataType} coords,float value) {
        if (coordsInBounds(coords, outShape)) {
          ${setOutput}
        }
      }
      void main() {
        ${coordsDataType} coords = getOutputCoords();
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${offsetSnippet};
        float scale = ${scaleSnippet};
        float inv = scale * inversesqrt(variance + float(varianceEpsilon));
        writeResult(coords,dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
  `;
        return userCode;
    }
    getUserCodeWgsl() {
        let offsetSnippet = '0.0';
        if (this.offsetShape != null) {
            offsetSnippet = 'getOffsetAtOutCoordsByGlobalId(globalId, index)';
        }
        let scaleSnippet = '1.0';
        if (this.scaleShape != null) {
            scaleSnippet = 'getScaleAtOutCoordsByGlobalId(globalId, index)';
        }
        const dim = this.outputShape.length;
        const coordsDataType = getCoordsDataTypeWgsl(dim);
        let setOutput = 'setOutput(coords[0], coords[1], coords[2], coords[3], value);';
        if (dim === 2) {
            setOutput = 'setOutput(coords[0], coords[1], value);';
        }
        if (dim === 3) {
            setOutput = 'setOutput(coords[0], coords[1], coords[2], value);';
        }
        const userCode = `
      fn writeResult(coords : ${coordsDataType}, value : f32) {
        if (coordsInBounds${dim}D(coords, uniforms.outShape)) {
          ${setOutput}
        }
      }
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coords = getOutputCoords(globalId, index);
        let xValue = getXAtOutCoordsByGlobalId(globalId, index);
        let meanValue = getMeanAtOutCoordsByGlobalId(globalId, index);
        let varianValue = getVarianceAtOutCoordsByGlobalId(globalId, index);
        let offsetValue = ${offsetSnippet};
        let scaleValue = ${scaleSnippet};
        let inv = scaleValue * inverseSqrt(varianValue + f32(uniforms.varianceEpsilon));
        writeResult(coords,dot(vec3<f32>(xValue, -meanValue, offsetValue), vec3<f32>(inv, inv, 1.0)));
      }
  `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const fusedBatchNormConfig = {
    kernelName: FusedBatchNorm,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { x, scale, offset, mean, variance } = inputs;
        const { varianceEpsilon } = attrs;
        const webGPUBackend = backend;
        const batchNormInputs = [x, mean, variance];
        let offsetShape = null;
        if (offset != null) {
            offsetShape = offset.shape;
            batchNormInputs.push(offset);
        }
        let scaleShape = null;
        if (scale != null) {
            scaleShape = scale.shape;
            batchNormInputs.push(scale);
        }
        const program = new BatchNormProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape);
        const uniformData = [{ type: 'float32', data: [varianceEpsilon] }];
        return webGPUBackend.runWebGPUProgram(program, batchNormInputs, x.dtype, uniformData);
    }
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function fusedConv2d(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, bias, preluActivationWeights } = inputs;
    const { strides, pad, dataFormat, dilations, dimRoundingMode, activation, leakyreluAlpha } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, dilations, pad, dimRoundingMode, false /* depthwise */, $dataFormat);
    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    let program;
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
        (convInfo.padInfo.type === 'SAME' || convInfo.padInfo.type === 'VALID')) {
        return conv2dByMatMul({
            x,
            filter,
            convInfo,
            backend,
            bias,
            activation,
            preluActivationWeights,
            leakyreluAlpha
        });
    }
    const useNaive = env().getBool('WEBGPU_USE_NAIVE_CONV2D');
    const useVec4 = convInfo.inChannels % 4 === 0 && convInfo.outChannels % 4 === 0;
    const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];
    const dimensions = [
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        { type: 'int32', data: [...padInfo] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] }
    ];
    if (useNaive) {
        // TODO(kainino0x): This may be obsolete, but is kept for reference.
        program = new Conv2DNaiveProgram(convInfo, hasBias, activation, hasPreluActivationWeights);
    }
    else if (useVec4) {
        program = new Conv2DMMVec4Program(convInfo, hasBias, activation, hasPreluActivationWeights);
        const dimAOuter = convInfo.outShape[1] * convInfo.outShape[2];
        const dimBOuter = convInfo.outShape[3];
        const dimInner = convInfo.filterHeight * convInfo.filterWidth * convInfo.inShape[3];
        dimensions.push({ type: 'int32', data: [dimAOuter] }, { type: 'int32', data: [dimBOuter] }, { type: 'int32', data: [dimInner] });
    }
    else {
        program = new Conv2DMMProgram(convInfo, hasBias, activation, hasPreluActivationWeights);
    }
    const inputVar = [x, filter];
    if (hasBias) {
        inputVar.push(bias);
    }
    if (hasPreluActivationWeights) {
        inputVar.push(preluActivationWeights);
    }
    return backend.runWebGPUProgram(program, inputVar, x.dtype, dimensions);
}
const fusedConv2DConfig = {
    kernelName: FusedConv2D,
    backendName: 'webgpu',
    kernelFunc: fusedConv2d,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function fusedDepthwiseConv2D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, bias, preluActivationWeights } = inputs;
    const { strides, pad, dilations, dimRoundingMode, activation } = attrs;
    let $dilations = dilations;
    if ($dilations == null) {
        $dilations = [1, 1];
    }
    util.assert(backend_util.eitherStridesOrDilationsAreOne(strides, $dilations), () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
        `1. Got strides ${strides} and dilations '${$dilations}'`);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, $dilations, pad, dimRoundingMode, true /* depthwise */);
    const programInputs = [x, filter];
    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (hasBias) {
        programInputs.push(bias);
    }
    if (hasPreluActivationWeights) {
        programInputs.push(preluActivationWeights);
    }
    let program;
    // TODO: To see if we need to relax the limitation. Currently, it's only for
    // filter size 3x3.
    if (convInfo.batchSize === 1 && convInfo.inHeight === convInfo.outHeight &&
        convInfo.inWidth === convInfo.outWidth && convInfo.strideHeight === 1 &&
        convInfo.strideWidth === 1 &&
        convInfo.filterHeight === convInfo.filterWidth &&
        convInfo.inChannels === convInfo.outChannels &&
        convInfo.filterHeight === 3 && convInfo.inChannels % 4 === 0) {
        program = new DepthwiseConv2D3x3Program(convInfo, hasBias, activation, hasPreluActivationWeights);
    }
    else {
        program = new DepthwiseConv2DProgram(convInfo, hasBias, activation, hasPreluActivationWeights);
    }
    const dimensions = [
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] }
    ];
    const result = backend.runWebGPUProgram(program, programInputs, 'float32', dimensions);
    return result;
}
const fusedDepthwiseConv2DConfig = {
    kernelName: FusedDepthwiseConv2D,
    backendName: 'webgpu',
    kernelFunc: fusedDepthwiseConv2D,
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class GatherNDProgram {
    constructor(sliceDim, shape) {
        this.variableNames = ['A', 'indices'];
        this.workGroupSize = [64, 1, 1];
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.shaderKey = `gathernd_${sliceDim}`;
        this.size = util.sizeFromShape(this.outputShape);
        this.sliceDim = sliceDim;
        this.uniforms = `int sliceDim; ${getCoordsDataType(sliceDim)} strides;`;
        this.uniformsWgsl =
            `sliceDim : u32; strides : ${getCoordsDataTypeWgsl(sliceDim)};`;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.outputShape.length);
        let strideString;
        if (this.sliceDim > 1) {
            strideString = 'strides[j]';
        }
        else {
            strideString = 'strides';
        }
        const userCode = `
         void main() {
          int currentIndex = getGlobalIndex();
          ${dtype} coords = getOutputCoords();
          int flattenIndex = 0;
          for (int j = 0; j < sliceDim; j++) {
            int index = int(round(getIndices(coords[0], j)));
            int strideNum = ${strideString};
            flattenIndex += index * strideNum;
          }
          if (currentIndex < size) {
            setOutput(currentIndex, getA(flattenIndex, coords[1]));
          }
        }
      `;
        return userCode;
    }
    getUserCodeWgsl() {
        let strideString;
        if (this.sliceDim > 1) {
            strideString = 'uniforms.strides[j]';
        }
        else {
            strideString = 'uniforms.strides';
        }
        const userCode = `
        ${getMainHeaderStringWgsl(this.workGroupSize)} {
          ${getGlobalIndexStringWgsl(this.workGroupSize)}
          let coords = getOutputCoords(globalId, index);
          var flattenIndex = 0u;
          for (var j = 0u; j < uniforms.sliceDim; j = j + 1u) {
            let indexTemp = u32(round(getIndices(coords[0], j)));
            let strideNum = ${strideString};
            flattenIndex = flattenIndex + indexTemp * strideNum;
          }
          if (index < uniforms.size) {
            setOutputFlat(index, getA(flattenIndex, coords[1]));
          }
        }
      `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function gatherNd(args) {
    const { inputs, backend } = args;
    const { params, indices } = inputs;
    const indicesShape = indices.shape;
    const sliceRank = indicesShape[indicesShape.length - 1];
    const paramsSize = util.sizeFromShape(params.shape);
    const [resultShape, numSlices, sliceSize, strides] = backend_util.prepareAndValidate(params, indices);
    const flattenIndices = reshape({ inputs: { x: indices }, backend, attrs: { shape: [numSlices, sliceRank] } });
    const flattenX = reshape({
        inputs: { x: params },
        backend,
        attrs: { shape: [(util.sizeFromShape(params.shape) / sliceSize), sliceSize] }
    });
    if (backend.shouldExecuteOnCPU([params, indices]) ||
        params.dtype === 'string') {
        const indicesData = backend.readSync(indices.dataId);
        const paramsBuf = backend.bufferSync(params);
        const outValue = gatherNdImplCPU(indicesData, paramsBuf, params.dtype, numSlices, sliceRank, sliceSize, strides, params.shape, paramsSize);
        return backend.makeTensorInfo(resultShape, params.dtype, outValue.values);
    }
    const program = new GatherNDProgram(sliceRank, [numSlices, sliceSize]);
    const uniformData = [{ type: 'int32', data: [sliceRank] }, { type: 'int32', data: strides }];
    const res = backend.runWebGPUProgram(program, [flattenX, flattenIndices], flattenX.dtype, uniformData);
    const reshaped = reshape({ inputs: { x: res }, backend, attrs: { shape: resultShape } });
    backend.disposeData(flattenIndices.dataId);
    backend.disposeData(flattenX.dataId);
    backend.disposeData(res.dataId);
    return reshaped;
}
const gatherNdConfig = {
    kernelName: GatherNd,
    backendName: 'webgpu',
    kernelFunc: gatherNd
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class GatherProgram {
    constructor(aShape, outputShape) {
        this.variableNames = ['A', 'indices'];
        this.workGroupSize = [64, 1, 1];
        this.outputShape = aShape.slice();
        this.aShape = aShape;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.shaderKey = `gather`;
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const sourceCoords = getSourceCoords(this.aShape);
        const userCode = `
      void main() {
        int index = getGlobalIndex();
        ivec4 resRC = getOutputCoords();
        if (index < size) {
          setOutput(index, getA(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const sourceCoords = getSourceCoords(this.aShape, 'u32');
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let resRC = getOutputCoords(globalId, index);
        if (index < uniforms.size) {
          setOutputFlat(index, getA(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
// The input and output are always flattened into rank 4 tensors.
function getSourceCoords(aShape, typePrefix = 'int') {
    const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
    const sourceCoords = [];
    for (let i = 0; i < aShape.length; i++) {
        if (i === 2) {
            sourceCoords.push(`${typePrefix}(getIndices(resRC.x, resRC.z))`);
        }
        else {
            sourceCoords.push(`${currentCoords[i]}`);
        }
    }
    return sourceCoords.join();
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function gatherV2(args) {
    const { inputs, backend, attrs } = args;
    const { x, indices } = inputs;
    const { axis, batchDims } = attrs;
    const parsedAxis = util.parseAxisParam(axis, x.shape)[0];
    const shapeInfo = backend_util.segment_util.collectGatherOpShapeInfo(x, indices, parsedAxis, batchDims);
    const indicesSize = util.sizeFromShape(indices.shape);
    const toDispose = [];
    const flattenX = reshape({
        inputs: { x },
        backend,
        attrs: {
            shape: [
                shapeInfo.batchSize, shapeInfo.outerSize, shapeInfo.dimSize,
                shapeInfo.sliceSize
            ]
        }
    });
    const flattenIndex = reshape({
        inputs: { x: indices },
        backend,
        attrs: { shape: [shapeInfo.batchSize, indicesSize / shapeInfo.batchSize] }
    });
    toDispose.push(flattenX);
    toDispose.push(flattenIndex);
    const flattenOutputShape = [
        shapeInfo.batchSize, shapeInfo.outerSize, indicesSize / shapeInfo.batchSize,
        shapeInfo.sliceSize
    ];
    if (backend.shouldExecuteOnCPU([x, indices])) {
        const indicesBufferInfo = backend.tensorMap.get(flattenIndex.dataId);
        const indicesValues = indicesBufferInfo.values;
        const indicesBuf = buffer(flattenIndex.shape, flattenIndex.dtype, indicesValues);
        const xBufferInfo = backend.tensorMap.get(flattenX.dataId);
        const xValues = xBufferInfo.values;
        const xBuf = buffer(flattenX.shape, flattenX.dtype, xValues);
        const outBuf = gatherV2ImplCPU(xBuf, indicesBuf, flattenOutputShape);
        toDispose.forEach(t => backend.disposeData(t.dataId));
        return backend.makeTensorInfo(shapeInfo.outputShape, outBuf.dtype, outBuf.values);
    }
    const program = new GatherProgram(flattenX.shape, flattenOutputShape);
    const res = backend.runWebGPUProgram(program, [flattenX, flattenIndex], flattenX.dtype);
    toDispose.push(res);
    const reshaped = reshape({ inputs: { x: res }, backend, attrs: { shape: shapeInfo.outputShape } });
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return reshaped;
}
const gatherV2Config = {
    kernelName: GatherV2,
    backendName: 'webgpu',
    kernelFunc: gatherV2
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const greater = binaryKernelFunc({
    opSnippet: BinaryOpType.GREATER,
    cpuKernelImpl: greaterImplCPU,
    dtype: 'bool',
});
const greaterConfig = {
    kernelName: Greater,
    backendName: 'webgpu',
    kernelFunc: greater
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const greaterEqual = binaryKernelFunc({
    opSnippet: BinaryOpType.GREATER_EQUAL,
    dtype: 'bool',
    cpuKernelImpl: greaterEqualImplCPU
});
const greaterEqualConfig = {
    kernelName: GreaterEqual,
    backendName: 'webgpu',
    kernelFunc: greaterEqual
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const less = binaryKernelFunc({ opSnippet: BinaryOpType.LESS, dtype: 'bool', cpuKernelImpl: lessImplCPU });
const lessConfig = {
    kernelName: Less,
    backendName: 'webgpu',
    kernelFunc: less
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const lessEqual = binaryKernelFunc({
    opSnippet: BinaryOpType.LESS_EQUAL,
    dtype: 'bool',
    cpuKernelImpl: lessEqualImplCPU
});
const lessEqualConfig = {
    kernelName: LessEqual,
    backendName: 'webgpu',
    kernelFunc: lessEqual
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const log = unaryKernelFunc({ opType: UnaryOpType.LOG, cpuKernelImpl: logImplCPU });
const logConfig = {
    kernelName: Log,
    backendName: 'webgpu',
    kernelFunc: log
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const logicalAnd = binaryKernelFunc({
    opSnippet: BinaryOpType.LOGICAL_AND,
    dtype: 'bool'
});
const logicalAndConfig = {
    kernelName: LogicalAnd,
    backendName: 'webgpu',
    kernelFunc: logicalAnd
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function max(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { reductionIndices, keepDims } = attrs;
    return reduce(x, reductionIndices, keepDims, 'max', backend);
}
const maxConfig = {
    kernelName: Max,
    backendName: 'webgpu',
    kernelFunc: max
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const maximum = binaryKernelFunc({
    opSnippet: BinaryOpType.MAX,
    cpuKernelImpl: maximumImplCPU,
});
const maximumConfig = {
    kernelName: Maximum,
    backendName: 'webgpu',
    kernelFunc: maximum
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function maxPool(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const dilations = 1;
    const convInfo = backend_util.computePool2DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    let program;
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1) {
        if (util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
            return identity({ inputs: { x }, backend });
        }
        program = new PoolWithFilterSizeEqualsOneProgram(convInfo);
    }
    else {
        program = new Pool2DProgram(convInfo, 'max');
    }
    const dimensions = [
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] }, {
            type: 'int32',
            data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        }
    ];
    return backend.runWebGPUProgram(program, [x], x.dtype, dimensions);
}
const maxPoolConfig = {
    kernelName: MaxPool,
    backendName: 'webgpu',
    kernelFunc: maxPool
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function mean(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { keepDims, axis } = attrs;
    return reduce(x, axis, keepDims, 'mean', backend);
}
const meanConfig = {
    kernelName: Mean,
    backendName: 'webgpu',
    kernelFunc: mean
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function min(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    return reduce(x, axis, keepDims, 'min', backend);
}
const minConfig = {
    kernelName: Min,
    backendName: 'webgpu',
    kernelFunc: min
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const minimum = binaryKernelFunc({
    opSnippet: BinaryOpType.MIN,
    cpuKernelImpl: minimumImplCPU,
});
const minimumConfig = {
    kernelName: Minimum,
    backendName: 'webgpu',
    kernelFunc: minimum
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class MirrorPadProgram {
    constructor(xShape, paddings, mode) {
        this.uniforms = '';
        this.uniformsWgsl = '';
        this.variableNames = ['x'];
        this.workGroupSize = [64, 1, 1];
        this.outputShape = paddings.map((p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.xShape = xShape;
        paddings.map((_, i) => {
            this.uniforms += ` ivec2 pad${i};`;
            this.uniformsWgsl += ` pad${i} : vec2<u32>;`;
        });
        this.offset = mode === 'reflect' ? 0 : 1;
        this.shaderKey = `mirrorPad_${mode}`;
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const rank = this.xShape.length;
        // The length of paddings are same with the rank of the input tensor.
        const start = this.xShape.map((_, i) => `pad${i}[0]`).join(',');
        const end = this.xShape
            .map((_, i) => `pad${i}[0] + xShape${rank > 1 ? `[${i}]` : ''}`)
            .join(',');
        const shaderStart = rank === 1 ? 'start' : 'start[i]';
        const shaderEnd = rank === 1 ? 'end' : 'end[i]';
        const shaderOutC = rank === 1 ? 'outC' : 'outC[i]';
        const dtype = getCoordsDataType(rank);
        const unpackedCoords = rank > 1 ?
            ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
            'coords';
        return `
      ${dtype} start = ${dtype}(${start});
      ${dtype} end = ${dtype}(${end});

      void main() {
        ${dtype} outC = getOutputCoords();
        int index = getGlobalIndex();
        if (index < size)
        {
          for (int i = 0; i < ${rank}; i++) {
            if (${shaderOutC} < ${shaderStart}) {
              ${shaderOutC} = ${shaderStart} * 2 - ${shaderOutC} - ${this.offset};
            } else if(${shaderOutC} >= ${shaderEnd}) {
              ${shaderOutC} = (${shaderEnd} - 1) * 2 - ${shaderOutC} + ${this.offset};
            }
          }
          ${dtype} coords = outC - start;
          setOutput(index, getX(${unpackedCoords}));
        }
      }
    `;
    }
    getUserCodeWgsl() {
        const rank = this.xShape.length;
        // The length of paddings are same with the rank of the input tensor.
        const start = this.xShape.map((_, i) => `uniforms.pad${i}[0]`).join(',');
        const end = this.xShape
            .map((_, i) => `uniforms.pad${i}[0] + uniforms.xShape${rank > 1 ? `[${i}]` : ''}`)
            .join(',');
        const shaderStart = rank === 1 ? 'start' : 'start[i]';
        const shaderEnd = rank === 1 ? 'end' : 'end[i]';
        const shaderOutC = rank === 1 ? 'outC' : 'outC[i]';
        const dtype = getCoordsDataTypeWgsl(rank);
        const unpackedCoords = rank > 1 ?
            ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
            'coords';
        return `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let start = ${dtype}(${start});
        let end = ${dtype}(${end});
        var outC = getOutputCoords(globalId, index);
        if (index < uniforms.size)
        {
          for (var i = 0u; i < ${rank}u; i = i + 1u) {
            if (${shaderOutC} < ${shaderStart}) {
              ${shaderOutC} = ${shaderStart} * 2u - ${shaderOutC} - ${this.offset}u;
            } elseif(${shaderOutC} >= ${shaderEnd}) {
              ${shaderOutC} = (${shaderEnd} - 1u) * 2u - ${shaderOutC} + ${this.offset}u;
            }
          }
          let coords = outC - start;
          setOutputFlat(index, getX(${unpackedCoords}));
        }
      }
    `;
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const mirrorPadConfig = {
    kernelName: MirrorPad,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { x } = inputs;
        const { paddings, mode } = attrs;
        const webGPUBackend = backend;
        const uniformData = paddings.map(p => {
            return { type: 'int32', data: [p[0], p[1]] };
        });
        const program = new MirrorPadProgram(x.shape, paddings, mode);
        const output = webGPUBackend.runWebGPUProgram(program, [x], x.dtype, uniformData);
        return output;
    }
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// This doesn't use unaryKernelFunc because negImplCPU is not of type
// SimpleUnaryKernelImplCPU.
function neg(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (backend.shouldExecuteOnCPU([x])) {
        const xData = backend.tensorMap.get(x.dataId);
        const [outValues, newShape] = negImplCPU(xData.values, x.shape, x.dtype);
        return backend.makeTensorInfo(newShape, x.dtype, outValues);
    }
    const program = new UnaryOpProgram(x.shape, UnaryOpType.NEG);
    return backend.runWebGPUProgram(program, [x], x.dtype);
}
const negConfig = {
    kernelName: Neg,
    backendName: 'webgpu',
    kernelFunc: neg
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function nonMaxSuppressionV3(args) {
    console.warn('tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold } = attrs;
    const boxesVals = backend.readSync(boxes.dataId);
    const scoresVals = backend.readSync(scores.dataId);
    const { selectedIndices } = kernel_impls.nonMaxSuppressionV3Impl(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
    return backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices));
}
const nonMaxSuppressionV3Config = {
    kernelName: NonMaxSuppressionV3,
    backendName: 'webgpu',
    kernelFunc: nonMaxSuppressionV3
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function nonMaxSuppressionV5(args) {
    console.warn('tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma } = attrs;
    const boxesVals = backend.readSync(boxes.dataId);
    const scoresVals = backend.readSync(scores.dataId);
    const maxOutputSizeVal = maxOutputSize;
    const iouThresholdVal = iouThreshold;
    const scoreThresholdVal = scoreThreshold;
    const softNmsSigmaVal = softNmsSigma;
    const { selectedIndices, selectedScores } = kernel_impls.nonMaxSuppressionV5Impl(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal, softNmsSigmaVal);
    return [
        backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices)),
        backend.makeTensorInfo([selectedScores.length], 'float32', new Float32Array(selectedScores))
    ];
}
const nonMaxSuppressionV5Config = {
    kernelName: NonMaxSuppressionV5,
    backendName: 'webgpu',
    kernelFunc: nonMaxSuppressionV5
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function zerosLike(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const r = zerosLike({ inputs: { x: realPart }, backend });
        const imagPart = imag({ inputs: { input: x }, backend });
        const i = zerosLike({ inputs: { x: imagPart }, backend });
        const result = complex({ inputs: { real: r, imag: i }, backend });
        backend.disposeData(realPart.dataId);
        backend.disposeData(r.dataId);
        backend.disposeData(imagPart.dataId);
        backend.disposeData(i.dataId);
        return result;
    }
    else {
        return fill({
            attrs: {
                shape: x.shape,
                dtype: x.dtype,
                value: x.dtype === 'string' ? '' : 0
            },
            backend
        });
    }
}
const zerosLikeConfig = {
    kernelName: ZerosLike,
    backendName: 'webgpu',
    kernelFunc: zerosLike
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function onesLike(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (x.dtype === 'string') {
        throw new Error('onesLike is not supported under string dtype');
    }
    else if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const r = onesLike({ inputs: { x: realPart }, backend });
        const imagPart = imag({ inputs: { input: x }, backend });
        const i = zerosLike({ inputs: { x: imagPart }, backend });
        const result = complex({ inputs: { real: r, imag: i }, backend });
        backend.disposeData(realPart.dataId);
        backend.disposeData(r.dataId);
        backend.disposeData(imagPart.dataId);
        backend.disposeData(i.dataId);
        return result;
    }
    else {
        return fill({ attrs: { shape: x.shape, dtype: x.dtype, value: 1 }, backend });
    }
}
const onesLikeConfig = {
    kernelName: OnesLike,
    backendName: 'webgpu',
    kernelFunc: onesLike
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function pack(args) {
    const { inputs, backend, attrs } = args;
    const { axis } = attrs;
    if (inputs.length === 1) {
        return expandDims({ inputs: { input: inputs[0] }, backend, attrs: { dim: axis } });
    }
    const shape = inputs[0].shape;
    const dtype = inputs[0].dtype;
    inputs.forEach(t => {
        util.assertShapesMatch(shape, t.shape, 'All tensors passed to stack must have matching shapes');
        util.assert(dtype === t.dtype, () => 'All tensors passed to stack must have matching dtypes');
    });
    const intermediateTensorInfos = [];
    const expandedTensors = inputs.map(t => {
        const expandedT = expandDims({ inputs: { input: t }, backend, attrs: { dim: axis } });
        intermediateTensorInfos.push(expandedT);
        return expandedT;
    });
    const result = concat({ inputs: expandedTensors, backend, attrs: { axis } });
    intermediateTensorInfos.forEach(t => backend.disposeData(t.dataId));
    return result;
}
const packConfig = {
    kernelName: Pack,
    backendName: 'webgpu',
    kernelFunc: pack
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class PadProgram {
    constructor(xShape, paddings) {
        this.variableNames = ['x'];
        this.uniforms = 'float constantValue;';
        this.uniformsWgsl = 'constantValue : f32;';
        this.workGroupSize = [64, 1, 1];
        this.outputShape = paddings.map((p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        paddings.map((_, i) => {
            this.uniforms += ` ivec2 pad${i};`;
            this.uniformsWgsl += ` pad${i} : vec2<u32>;`;
        });
        this.xShape = xShape;
        this.shaderKey = 'pad';
        this.useWgsl = getUseWgsl();
        this.size = util.sizeFromShape(this.outputShape);
    }
    getUserCode() {
        const rank = this.xShape.length;
        const type = getCoordsDataType(rank);
        // The length of paddings are same with the rank of the input tensor.
        const start = this.xShape.map((_, i) => `pad${i}[0]`).join(',');
        const end = this.xShape
            .map((_, i) => `pad${i}[0] + xShape${rank > 1 ? `[${i}]` : ''}`)
            .join(',');
        const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
        const endValue = rank > 1 ? `${type}(${end})` : `${end}`;
        const leftPadCondition = rank > 1 ? `any(lessThan(outC, start))` : `outC < start`;
        const rightPadCondition = rank > 1 ? `any(greaterThanEqual(outC, end))` : `outC >= end`;
        const unpackedCoords = rank > 1 ?
            ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
            'coords';
        const userCode = `
      ${type} start = ${startValue};
      ${type} end = ${endValue};

      void main() {
        int flatIndex = getGlobalIndex();

          if (flatIndex < size) {
            ${type} outC = getOutputCoords();

            if (${leftPadCondition} || ${rightPadCondition}) {
              setOutput(flatIndex, constantValue);
            } else {
              ${type} coords = outC - start;
              setOutput(flatIndex, getX(${unpackedCoords}));
            }
          }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const rank = this.xShape.length;
        const type = getCoordsDataTypeWgsl(rank);
        // The length of paddings are same with the rank of the input tensor.
        const start = this.xShape.map((_, i) => `uniforms.pad${i}[0]`).join(',');
        const end = this.xShape
            .map((_, i) => `uniforms.pad${i}[0] + uniforms.xShape${rank > 1 ? `[${i}]` : ''}`)
            .join(',');
        const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
        const endValue = rank > 1 ? `${type}(${end})` : `${end}`;
        const leftPadCondition = rank > 1 ? `any(outC < start)` : `outC < start`;
        const rightPadCondition = rank > 1 ? `any(outC >= end)` : `outC >= end`;
        const unpackedCoords = rank > 1 ?
            ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
            'coords';
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let start = ${startValue};
        let end = ${endValue};
        if (index < uniforms.size) {
          let outC = getOutputCoords(globalId, index);

          if (${leftPadCondition} || ${rightPadCondition}) {
            setOutputFlat(index, uniforms.constantValue);
          } else {
            let coords = outC - start;
            setOutputFlat(index, getX(${unpackedCoords}));
          }
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const padV2 = (args) => {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { paddings, constantValue } = attrs;
    if (paddings.every(p => util.arraysEqual(p, [0, 0]))) {
        return identity({ inputs: { x }, backend });
    }
    if (util.sizeFromShape(x.shape) === 0) {
        // Short-circuit the computation, since x doesn't have value, only
        // the shape is used to compute output shape to pad.
        const outputShape = paddings.map((p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
        return fill({
            backend,
            attrs: { shape: outputShape, value: constantValue, dtype: x.dtype }
        });
    }
    const uniformData = [{ type: 'float32', data: [constantValue] }];
    paddings.map(p => uniformData.push({ type: 'int32', data: [p[0], p[1]] }));
    const program = new PadProgram(x.shape, paddings);
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
};
const padV2Config = {
    kernelName: PadV2,
    backendName: 'webgpu',
    kernelFunc: padV2
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const pow = binaryKernelFunc({
    opSnippet: BinaryOpType.POW,
});
const powConfig = {
    kernelName: Pow,
    backendName: 'webgpu',
    kernelFunc: pow
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function prelu(args) {
    const { inputs, backend } = args;
    const { x, alpha } = inputs;
    const program = new BinaryOpProgram(BinaryOpType.PRELU, x.shape, alpha.shape);
    return backend.runWebGPUProgram(program, [x, alpha], x.dtype);
}
const preluConfig = {
    kernelName: Prelu,
    backendName: 'webgpu',
    kernelFunc: prelu
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function prod(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    return reduce(x, axis, keepDims, 'prod', backend);
}
const prodConfig = {
    kernelName: Prod,
    backendName: 'webgpu',
    kernelFunc: prod
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const range = (args) => {
    const { backend, attrs } = args;
    const { start, stop, step, dtype } = attrs;
    const values = rangeImplCPU(start, stop, step, dtype);
    return backend.makeTensorInfo([values.length], dtype, values);
};
const rangeConfig = {
    kernelName: Range,
    backendName: 'webgpu',
    kernelFunc: range
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const realDiv = binaryKernelFunc({ opSnippet: BinaryOpType.DIV });
const realDivConfig = {
    kernelName: RealDiv,
    backendName: 'webgpu',
    kernelFunc: realDiv
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const relu = unaryKernelFunc({ opType: UnaryOpType.RELU });
const reluConfig = {
    kernelName: Relu,
    backendName: 'webgpu',
    kernelFunc: relu
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const relu6 = unaryKernelFunc({ opType: UnaryOpType.RELU6 });
const relu6Config = {
    kernelName: Relu6,
    backendName: 'webgpu',
    kernelFunc: relu6
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class ResizeBilinearProgram {
    constructor(inputShape, newHeight, newWidth, alignCorners) {
        this.variableNames = ['x'];
        this.workGroupSize = [64, 1, 1];
        this.outputShape = [inputShape[0], newHeight, newWidth, inputShape[3]];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.alignCorners = alignCorners;
        this.shaderKey = `resizeBilinear_${alignCorners}_${this.outputShape[1] > 1}_${this.outputShape[2] > 1}`;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const adjustHeight = this.alignCorners && this.outputShape[1] > 1;
        const adjustWidth = this.alignCorners && this.outputShape[2] > 1;
        const userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        if (all(lessThan(coords, outShape))) {
          int b = coords[0];
          int d = coords[3];
          ivec2 rc = coords.yz;

          vec2 effectiveInSize = vec2(
            ${adjustHeight ? `xShape.y - 1.0` : `xShape.y`},
            ${adjustWidth ? `xShape.z - 1.0` : `xShape.z`});

          vec2 effectiveOutSize = vec2(
            ${adjustHeight ? `outShape.y - 1.0` : `outShape.y`},
            ${adjustWidth ? `outShape.z - 1.0` : `outShape.z`});

          vec2 effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          vec2 sourceFracIndexRC = vec2(rc) * effectiveInputOverOutputRatioRC;

          // Compute the four integer indices.
          ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);
          ivec2 sourceCeilRC = ivec2(
            min(xShape.yz - 1.0, ceil(sourceFracIndexRC)));

          float topLeft = getX(b, sourceFloorRC.x, sourceFloorRC.y, d);
          float bottomLeft = getX(b, sourceCeilRC.x, sourceFloorRC.y, d);
          float topRight = getX(b, sourceFloorRC.x, sourceCeilRC.y, d);
          float bottomRight = getX(b, sourceCeilRC.x, sourceCeilRC.y, d);

          vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

          float top = topLeft + (topRight - topLeft) * fracRC.y;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
          float newValue = top + (bottom - top) * fracRC.x;

          setOutput(b, coords[1], coords[2], d, newValue);
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const adjustHeight = this.alignCorners && this.outputShape[1] > 1;
        const adjustWidth = this.alignCorners && this.outputShape[2] > 1;
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coords = getOutputCoords(globalId, index);
        if (all(coords < uniforms.outShape)) {
          let b = coords[0];
          let d = coords[3];
          let rc = coords.yz;

          let effectiveInSize = vec2<f32>(
            ${adjustHeight ? `f32(uniforms.xShape.y) - 1.0` :
            `f32(uniforms.xShape.y)`},
            ${adjustWidth ? `f32(uniforms.xShape.z) - 1.0` :
            `f32(uniforms.xShape.z)`});

          let effectiveOutSize = vec2<f32>(
            ${adjustHeight ? `f32(uniforms.outShape.y) - 1.0` :
            `f32(uniforms.outShape.y)`},
            ${adjustWidth ? `f32(uniforms.outShape.z) - 1.0` :
            `f32(uniforms.outShape.z)`});

          let effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          let sourceFracIndexRC = vec2<f32>(rc) * effectiveInputOverOutputRatioRC;

          // Compute the four integer indices.
          let sourceFloorRC = vec2<u32>(sourceFracIndexRC);
          let sourceCeilRC = vec2<u32>(
            min(vec2<f32>(uniforms.xShape.yz) - vec2<f32>(1.0), ceil(sourceFracIndexRC)));

          let topLeft = getX(b, sourceFloorRC.x, sourceFloorRC.y, d);
          let bottomLeft = getX(b, sourceCeilRC.x, sourceFloorRC.y, d);
          let topRight = getX(b, sourceFloorRC.x, sourceCeilRC.y, d);
          let bottomRight = getX(b, sourceCeilRC.x, sourceCeilRC.y, d);

          let fracRC = sourceFracIndexRC - vec2<f32>(sourceFloorRC);

          let top = topLeft + (topRight - topLeft) * fracRC.y;
          let bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
          let newValue = top + (bottom - top) * fracRC.x;

          setOutput(b, coords[1], coords[2], d, newValue);
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function resizeBilinear(args) {
    const { inputs, backend, attrs } = args;
    const { images } = inputs;
    const { alignCorners, size } = attrs;
    const [newHeight, newWidth] = size;
    const program = new ResizeBilinearProgram(images.shape, newHeight, newWidth, alignCorners);
    return backend.runWebGPUProgram(program, [images], 'float32');
}
const resizeBilinearConfig = {
    kernelName: ResizeBilinear,
    backendName: 'webgpu',
    kernelFunc: resizeBilinear
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class ResizeNearestNeighborProgram {
    constructor(inputShape, newHeight, newWidth, alignCorners, halfPixelCenters) {
        this.variableNames = ['x'];
        this.workGroupSize = [64, 1, 1];
        this.outputShape = [inputShape[0], newHeight, newWidth, inputShape[3]];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.alignCorners = alignCorners;
        this.halfPixelCenters = halfPixelCenters;
        this.shaderKey =
            `resizeNearest_${alignCorners}_${this.outputShape[1] > 1}_${this.outputShape[2] > 1}_${halfPixelCenters}`;
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        // When align corners is false, we rounds the value with floor.
        const roundBase = this.alignCorners ? '0.5' : '0.0';
        let sourceFracIndexRC;
        if (this.halfPixelCenters) {
            sourceFracIndexRC =
                `max((vec2(rc) + vec2(0.5)) * effectiveInputOverOutputRatioRC` +
                    `, vec2(0.0))`;
        }
        else {
            sourceFracIndexRC = `vec2(rc) * effectiveInputOverOutputRatioRC`;
        }
        const adjustHeight = this.alignCorners && this.outputShape[1] > 1;
        const adjustWidth = this.alignCorners && this.outputShape[2] > 1;
        const userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        if (all(lessThan(coords, outShape))) {
          int b = coords[0];
          int d = coords[3];
          ivec2 rc = coords.yz;

          vec2 effectiveInSize = vec2(
            ${adjustHeight ? `xShape.y - 1.0` : `xShape.y`},
            ${adjustWidth ? `xShape.z - 1.0` : `xShape.z`});

          vec2 effectiveOutSize = vec2(
            ${adjustHeight ? `outShape.y - 1.0` : `outShape.y`},
            ${adjustWidth ? `outShape.z - 1.0` : `outShape.z`});

          vec2 effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          vec2 sourceFracIndexRC = ${sourceFracIndexRC};

          // Compute the coordinators of nearest neighbor point.
          const vec2 inputShapeRC = vec2(xShape.y, xShape.z);
          ivec2 sourceNearestRC = ivec2(
            min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${roundBase})));
          float newValue = getX(b, sourceNearestRC.x, sourceNearestRC.y, d);

          setOutput(b, coords[1], coords[2], d, newValue);
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        // When align corners is false, we rounds the value with floor.
        const roundBase = this.alignCorners ? '0.5' : '0.0';
        let sourceFracIndexRC;
        if (this.halfPixelCenters) {
            sourceFracIndexRC =
                `max((vec2<f32>(rc) + vec2<f32>(0.5)) * effectiveInputOverOutputRatioRC` +
                    `, vec2<f32>(0.0))`;
        }
        else {
            sourceFracIndexRC = `vec2<f32>(rc) * effectiveInputOverOutputRatioRC`;
        }
        const adjustHeight = this.alignCorners && this.outputShape[1] > 1;
        const adjustWidth = this.alignCorners && this.outputShape[2] > 1;
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coords = getOutputCoords(globalId, index);
        if (all(coords < uniforms.outShape)) {
          let b = coords[0];
          let d = coords[3];
          let rc = coords.yz;

          let effectiveInSize = vec2<f32>(
            ${adjustHeight ? `f32(uniforms.xShape.y) - 1.0` :
            `f32(uniforms.xShape.y)`},
            ${adjustWidth ? `f32(uniforms.xShape.z) - 1.0` :
            `f32(uniforms.xShape.z)`});

          let effectiveOutSize = vec2<f32>(
            ${adjustHeight ? `f32(uniforms.outShape.y) - 1.0` :
            `f32(uniforms.outShape.y)`},
            ${adjustWidth ? `f32(uniforms.outShape.z) - 1.0` :
            `f32(uniforms.outShape.z)`});

          let effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          let sourceFracIndexRC = ${sourceFracIndexRC};

          // Compute the coordinators of nearest neighbor point.
          let inputShapeRC = vec2<f32>(f32(uniforms.xShape.y), f32(uniforms.xShape.z));
          let sourceNearestRC = vec2<u32>(
            min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${roundBase})));
          let newValue = getX(b, sourceNearestRC.x, sourceNearestRC.y, d);

          setOutput(b, coords[1], coords[2], d, newValue);
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function resizeNearestNeighbor(args) {
    const { inputs, backend, attrs } = args;
    const { images } = inputs;
    const { alignCorners, halfPixelCenters, size } = attrs;
    const [newHeight, newWidth] = size;
    const program = new ResizeNearestNeighborProgram(images.shape, newHeight, newWidth, alignCorners, halfPixelCenters);
    return backend.runWebGPUProgram(program, [images], images.dtype);
}
const resizeNearestNeighborConfig = {
    kernelName: ResizeNearestNeighbor,
    backendName: 'webgpu',
    kernelFunc: resizeNearestNeighbor
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const rsqrt = unaryKernelFunc({ opType: UnaryOpType.RSQRT, cpuKernelImpl: rsqrtImplCPU });
const rsqrtConfig = {
    kernelName: Rsqrt,
    backendName: 'webgpu',
    kernelFunc: rsqrt
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class SelectProgram {
    constructor(cRank, shape, rank) {
        this.variableNames = ['c', 'a', 'b'];
        this.workGroupSize = [64, 1, 1];
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.cRank = cRank;
        this.rank = rank;
        this.shaderKey = 'select';
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        let cCoords;
        let abCoords;
        if (this.rank > 4) {
            throw Error(`Where for rank ${this.rank} is not yet supported`);
        }
        if (this.rank === 1) {
            abCoords = `resRC`;
            cCoords = `resRC`;
        }
        else {
            const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
            const cCoordVars = [];
            const abCoordVars = [];
            for (let i = 0; i < this.outputShape.length; i++) {
                abCoordVars.push(`${currentCoords[i]}`);
                if (i < this.cRank) {
                    cCoordVars.push(`${currentCoords[i]}`);
                }
            }
            cCoords = cCoordVars.join();
            abCoords = abCoordVars.join();
        }
        const dtype = getCoordsDataType(this.rank);
        const userCode = `
      void main() {
        int index = getGlobalIndex();
        if (index < size) {
          ${dtype} resRC = getOutputCoords();

          float cVal = getC(${cCoords});
          if (cVal >= 1.0) {
            setOutput(index, getA(${abCoords}));
          } else {
            setOutput(index, getB(${abCoords}));
          }
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        // TODO(WGSL): below code can be merged with getUserCode.
        let cCoords;
        let abCoords;
        if (this.rank > 4) {
            throw Error(`Where for rank ${this.rank} is not yet supported`);
        }
        if (this.rank === 1) {
            abCoords = `resRC`;
            cCoords = `resRC`;
        }
        else {
            const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
            const cCoordVars = [];
            const abCoordVars = [];
            for (let i = 0; i < this.outputShape.length; i++) {
                abCoordVars.push(`${currentCoords[i]}`);
                if (i < this.cRank) {
                    cCoordVars.push(`${currentCoords[i]}`);
                }
            }
            cCoords = cCoordVars.join();
            abCoords = abCoordVars.join();
        }
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if (index < uniforms.size) {
          let resRC = getOutputCoords(globalId, index);
          let cVal = getC(${cCoords});
          if (cVal >= 1.0) {
            setOutputFlat(index, getA(${abCoords}));
          } else {
            setOutputFlat(index, getB(${abCoords}));
          }
        }
      }
    `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function select$1(args) {
    const { inputs, backend } = args;
    const { condition, t, e } = inputs;
    const program = new SelectProgram(condition.shape.length, t.shape, t.shape.length);
    return backend.runWebGPUProgram(program, [condition, t, e], upcastType(t.dtype, e.dtype));
}
const selectConfig = {
    kernelName: Select,
    backendName: 'webgpu',
    kernelFunc: select$1
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sigmoid = unaryKernelFunc({ opType: UnaryOpType.SIGMOID });
const sigmoidConfig = {
    kernelName: Sigmoid,
    backendName: 'webgpu',
    kernelFunc: sigmoid,
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sub = binaryKernelFunc({
    opSnippet: BinaryOpType.SUB,
    cpuKernelImpl: subImplCPU,
    supportsComplex: true
});
const subConfig = {
    kernelName: Sub,
    backendName: 'webgpu',
    kernelFunc: sub
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function softmax(args) {
    const { inputs, backend, attrs } = args;
    const { logits } = inputs;
    const { dim } = attrs;
    const axes = util.parseAxisParam([dim], logits.shape);
    const maxLogit = max({
        inputs: { x: logits },
        backend,
        attrs: { reductionIndices: axes, keepDims: false }
    });
    const expandedShape = backend_util.expandShapeToKeepDim(maxLogit.shape, axes);
    const maxLogitsReshaped = reshape({ inputs: { x: maxLogit }, backend, attrs: { shape: expandedShape } });
    const a = sub({ inputs: { a: logits, b: maxLogitsReshaped }, backend });
    const b = exp({ inputs: { x: a }, backend });
    const sumExp = sum({ inputs: { x: b }, backend, attrs: { axis: axes, keepDims: false } });
    const sumExpReshaped = reshape({ inputs: { x: sumExp }, backend, attrs: { shape: expandedShape } });
    const res = realDiv({ inputs: { a: b, b: sumExpReshaped }, backend });
    backend.disposeData(maxLogit.dataId);
    backend.disposeData(maxLogitsReshaped.dataId);
    backend.disposeData(a.dataId);
    backend.disposeData(b.dataId);
    backend.disposeData(sumExp.dataId);
    backend.disposeData(sumExpReshaped.dataId);
    return res;
}
const softmaxConfig = {
    kernelName: Softmax,
    backendName: 'webgpu',
    kernelFunc: softmax
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const spaceToBatchND = (args) => {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockShape, paddings } = attrs;
    util.assert(x.shape.length <= 4, () => 'spaceToBatchND for rank > 4 with a WebGPU backend not ' +
        'implemented yet');
    const prod = blockShape.reduce((a, b) => a * b);
    const completePaddings = [[0, 0]];
    completePaddings.push(...paddings);
    for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
        completePaddings.push([0, 0]);
    }
    const toDispose = [];
    const paddedX = padV2({
        inputs: { x },
        backend,
        attrs: { paddings: completePaddings, constantValue: 0 }
    });
    const reshapedPaddedShape = backend_util.getReshaped(paddedX.shape, blockShape, prod, false);
    const permutedReshapedPaddedPermutation = backend_util.getPermuted(reshapedPaddedShape.length, blockShape.length, false);
    const flattenShape = backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);
    const reshapedPaddedX = reshape({ inputs: { x: paddedX }, backend, attrs: { shape: reshapedPaddedShape } });
    const paddedXT = transpose({
        inputs: { x: reshapedPaddedX },
        backend,
        attrs: { perm: permutedReshapedPaddedPermutation }
    });
    const result = reshape({ inputs: { x: paddedXT }, backend, attrs: { shape: flattenShape } });
    toDispose.push(paddedX);
    toDispose.push(reshapedPaddedX);
    toDispose.push(paddedXT);
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return result;
};
const spaceToBatchNDConfig = {
    kernelName: SpaceToBatchND,
    backendName: 'webgpu',
    kernelFunc: spaceToBatchND
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sqrt = unaryKernelFunc({ opType: UnaryOpType.SQRT });
const sqrtConfig = {
    kernelName: Sqrt,
    backendName: 'webgpu',
    kernelFunc: sqrt
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const squareConfig = {
    kernelName: Square,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, backend }) => {
        const { x } = inputs;
        const webGPUBackend = backend;
        const program = new UnaryOpProgram(x.shape, UnaryOpType.SQUARE);
        return webGPUBackend.runWebGPUProgram(program, [x], x.dtype);
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const squaredDifference = binaryKernelFunc({
    opSnippet: BinaryOpType.SQUARED_DIFFERENCE,
});
const squaredDifferenceConfig = {
    kernelName: SquaredDifference,
    backendName: 'webgpu',
    kernelFunc: squaredDifference
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class StridedSliceProgram {
    constructor(destSize) {
        this.variableNames = ['x'];
        // TODO(xing.xu): Increase the workPerThread.
        this.workPerThread = 1;
        this.workGroupSize = [64, 1, 1];
        this.outputShape = destSize;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
        this.dtype = getCoordsDataType(this.outputShape.length);
        this.dtypeWgsl = getCoordsDataTypeWgsl(this.outputShape.length);
        this.uniforms = `${this.dtype} begin; ${this.dtype} strides; `;
        this.uniformsWgsl =
            `begin : ${this.dtypeWgsl};  strides : ${this.dtypeWgsl}; `;
        this.shaderKey = 'stridedSlice';
        this.size = util.sizeFromShape(this.outputShape);
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const rank = this.outputShape.length;
        let newCoords = '';
        if (rank === 1) {
            newCoords = 'coords * strides + begin';
        }
        else {
            let outputAxis = 0;
            newCoords =
                this.outputShape
                    .map((_, i) => {
                    outputAxis++;
                    return this.outputShape.length === 1 ?
                        `coords * strides[${i}] + begin[${i}]` :
                        `coords[${outputAxis - 1}] * strides[${i}] + begin[${i}]`;
                })
                    .join(',');
        }
        const userCode = `
       void main() {
         int index = getGlobalIndex();
         if (index < size)
         {
           ${this.dtype} coords = getOutputCoords();
           setOutput(index, getX(${newCoords}));
         }
       }
     `;
        return userCode;
    }
    getUserCodeWgsl() {
        const rank = this.outputShape.length;
        let newCoords = '';
        if (rank === 1) {
            newCoords = 'coords * uniforms.strides + uniforms.begin';
        }
        else {
            let outputAxis = 0;
            newCoords =
                this.outputShape
                    .map((_, i) => {
                    outputAxis++;
                    return this.outputShape.length === 1 ?
                        `coords * uniforms.strides[${i}] + uniforms.begin[${i}]` :
                        `coords[${outputAxis - 1}] * uniforms.strides[${i}] + uniforms.begin[${i}]`;
                })
                    .join(',');
        }
        const userCode = `
       ${getMainHeaderStringWgsl(this.workGroupSize)} {
         ${getGlobalIndexStringWgsl(this.workGroupSize)}
         if (index < uniforms.size)
         {
           let coords = getOutputCoords(globalId, index);
           setOutputFlat(index, getX(${newCoords}));
         }
       }
     `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function stridedSlice(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask } = attrs;
    const { nonStrided, $begin, $strides, size, newShape, outShape } = slice_util.sliceInfo(x.shape, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    const $x = reshape({ inputs: { x }, backend, attrs: { shape: newShape } });
    let result;
    if (nonStrided) {
        const sliced = slice({ inputs: { x: $x }, backend, attrs: { begin: $begin, size } });
        result = reshape({ inputs: { x: sliced }, backend, attrs: { shape: outShape } });
        backend.disposeData(sliced.dataId);
    }
    else if (outShape.some(axis => axis === 0)) {
        result = backend.makeTensorInfo(outShape, x.dtype, []);
    }
    else {
        const shouldExecuteOnCPU = backend.shouldExecuteOnCPU([$x]);
        if (shouldExecuteOnCPU) {
            const xBufferInfo = backend.tensorMap.get($x.dataId);
            const values = xBufferInfo.values;
            const xBuf = buffer($x.shape, $x.dtype, values);
            const resultValues = stridedSliceImplCPU(outShape, xBuf, $strides, $begin);
            result = backend.makeTensorInfo(outShape, $x.dtype, resultValues.values);
        }
        else {
            const program = new StridedSliceProgram(outShape);
            const uniformData = [{ type: 'int32', data: $begin }, { type: 'int32', data: $strides }];
            result = backend.runWebGPUProgram(program, [$x], $x.dtype, uniformData);
        }
    }
    const resultReshaped = reshape({ inputs: { x: result }, backend, attrs: { shape: outShape } });
    backend.disposeData($x.dataId);
    backend.disposeData(result.dataId);
    return resultReshaped;
}
const stridedSliceConfig = {
    kernelName: StridedSlice,
    backendName: 'webgpu',
    kernelFunc: stridedSlice
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function stringNGrams(args) {
    const { inputs, backend, attrs } = args;
    const { separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences } = attrs;
    const { data, dataSplits } = inputs;
    const $data = backend.readSync(data.dataId);
    const $dataSplits = backend.readSync(dataSplits.dataId);
    const [nGrams, nGramsSplits] = stringNGramsImplCPU($data, $dataSplits, separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences);
    return [
        backend.makeTensorInfo([nGrams.length], 'string', nGrams),
        backend.makeTensorInfo(dataSplits.shape, 'int32', nGramsSplits),
    ];
}
const stringNGramsConfig = {
    kernelName: StringNGrams,
    backendName: 'webgpu',
    kernelFunc: stringNGrams,
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const tanh = unaryKernelFunc({ opType: UnaryOpType.TANH });
const tanhConfig = {
    kernelName: Tanh,
    backendName: 'webgpu',
    kernelFunc: tanh
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class TileProgram {
    constructor(aShape, reps) {
        this.variableNames = ['A'];
        this.workGroupSize = [64, 1, 1];
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[i] * reps[i];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.rank = this.outputShape.length;
        this.size = util.sizeFromShape(this.outputShape);
        this.shaderKey = 'tile';
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.rank);
        const sourceCoords = getSourceCoords$1(this.rank);
        const userCode = `
      void main() {
        int index = getGlobalIndex();
        if (index < size) {
          ${dtype} resRC = getOutputCoords();
          setOutput(index, getA(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
    getUserCodeWgsl() {
        const sourceCoords = getSourceCoords$1(this.rank, 'uniforms.');
        const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if (index < uniforms.size) {
          let resRC = getOutputCoords(globalId, index);
          setOutputFlat(index, getA(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
function getSourceCoords$1(rank, uniformPrefix = '') {
    if (rank >= 5) {
        throw Error(`Tile for rank ${rank} is not yet supported`);
    }
    if (rank === 1) {
        return `(resRC % ${uniformPrefix}aShape)`;
    }
    const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
    const sourceCoords = [];
    for (let i = 0; i < rank; i++) {
        sourceCoords.push(`(${currentCoords[i]} % ${uniformPrefix}aShape[${i}])`);
    }
    return sourceCoords.join();
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function tile(params) {
    const { inputs, backend, attrs } = params;
    const { x } = inputs;
    const { reps } = attrs;
    // tile gpu program cannot handle rank >= 5 case.
    if (backend.shouldExecuteOnCPU([x]) || x.dtype === 'string' ||
        x.shape.length >= 5) {
        // Even thought string tensor is always on CPU, just to be consistent on how
        // to access tensor data.
        const data = backend.readSync(x.dataId);
        const value = x.dtype === 'string' ?
            data.map(d => util.decodeString(d)) :
            data;
        const buf = buffer(x.shape, x.dtype, value);
        const outBuf = tileImplCPU(buf, reps);
        return backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
    }
    const program = new TileProgram(x.shape, reps);
    const output = backend.runWebGPUProgram(program, [x], x.dtype);
    return output;
}
const tileConfig = {
    kernelName: Tile,
    backendName: 'webgpu',
    kernelFunc: tile,
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class TransformProgram {
    constructor(outShape) {
        this.variableNames = ['Image', 'Transforms'];
        this.uniforms = 'int interpolationModeId; int fillModeId; float fillValue;';
        this.uniformsWgsl = 'interpolationModeId : i32; fillModeId : i32; fillValue : f32;';
        this.workGroupSize = [64, 1, 1];
        this.outputShape = outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize);
        this.shaderKey = 'transform';
        this.useWgsl = getUseWgsl();
    }
    getUserCode() {
        const userCode = `
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(fillModeId == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (fillModeId == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (fillModeId == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < imageShape[1] && 0 <= coordX && coordX < imageShape[2]) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = fillValue;
              }
              return outputValue;
            }

          void main() {
            ivec4 coords = getOutputCoords();
            if (coordsInBounds(coords, outShape)) {
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = fillValue;
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(imageShape[2]));
                float mapY = mapCoord(inY, float(imageShape[1]));

                if (interpolationModeId == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(coords[0], coords[1], coords[2], coords[3], outputValue);
            }
          }
        `;
        return userCode;
    }
    getUserCodeWgsl() {
        const userCode = `
          fn mapCoord(outCoord : f32, len : f32) -> f32{
            var inCoord = outCoord;
            if(uniforms.fillModeId == 2) {
              if (inCoord < 0.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz2 = 2.0 * len;
                  if (inCoord < sz2) {
                    inCoord = sz2 * f32(i32(f32(-inCoord / sz2))) +
                    inCoord;
                  }
                  if (inCoord < -len) {
                    inCoord = inCoord + sz2;
                  } else {
                    inCoord = -inCoord - 1.0;
                  }
                }
              } elseif (inCoord > len - 1.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz2 = 2.0 * len;
                  inCoord = inCoord - sz2 * f32(i32(f32(inCoord / sz2)));
                  if (inCoord >= len) {
                    inCoord = sz2 - inCoord - 1.0;
                  }
                }
              }
              return clamp(inCoord, 0.0, len - 1.0);
            } elseif (uniforms.fillModeId == 3) {
              if (inCoord < 0.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz = len - 1.0;
                  inCoord = inCoord + len * (f32(i32(f32(-inCoord / sz))) + 1.0);
                }
              } elseif (inCoord > len - 1.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz = len - 1.0;
                  inCoord = inCoord - len * f32(i32(f32(inCoord / sz)));
                }
              }
              return clamp(inCoord, 0.0, len - 1.0);
            } elseif (uniforms.fillModeId == 4) {
              return clamp(outCoord, 0.0, len - 1.0);
            }
            return outCoord;
          }
          fn readWithFillValue(batch : i32, coordY : i32, coordX : i32,
            channel : i32) -> f32 {
            var outputValue : f32;
            if (0 <= coordY && coordY < i32(uniforms.imageShape[1]) && 0 <= coordX && coordX < i32(uniforms.imageShape[2])) {
                outputValue = getImage(u32(batch), u32(coordY), u32(coordX), u32(channel));
            } else {
              outputValue = uniforms.fillValue;
            }
            return outputValue;
          }

          ${getMainHeaderStringWgsl(this.workGroupSize)} {
            ${getGlobalIndexStringWgsl(this.workGroupSize)}
            let coords = getOutputCoords(globalId, index);
            if (coordsInBounds4D(coords, uniforms.outShape)) {
              var outputValue : f32;
              let batch = coords[0];
              let x = coords[2];
              let y = coords[1];
              let channel = coords[3];
              let xf = f32(x);
              let yf = f32(y);
              let a1 = getTransforms(batch, 0u);
              let a2 = getTransforms(batch, 1u);
              let a3 = getTransforms(batch, 2u);
              let b1 = getTransforms(batch, 3u);
              let b2 = getTransforms(batch, 4u);
              let b3 = getTransforms(batch, 5u);
              let c1 = getTransforms(batch, 6u);
              let c2 = getTransforms(batch, 7u);
              let projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = uniforms.fillValue;
              } else {
                let inX = (a1 * xf + a2 * yf + a3) / projection;
                let inY = (b1 * xf + b2 * yf + b3) / projection;
                let mapX = mapCoord(inX, f32(uniforms.imageShape[2]));
                let mapY = mapCoord(inY, f32(uniforms.imageShape[1]));

                if (uniforms.interpolationModeId == 1) {
                  let coordY = i32(round(mapY));
                  let coordX = i32(round(mapX));
                  outputValue = readWithFillValue(i32(batch), coordY, coordX,
                    i32(channel));
                } else {
                  let yFloor = floor(mapY);
                  let xFloor = floor(mapX);
                  let yCeil = yFloor + 1.0;
                  let xCeil = xFloor + 1.0;
                  let valueYFloor = (xCeil - mapX) *
                  readWithFillValue(i32(batch), i32(yFloor), i32(xFloor), i32(channel)) +
                  (mapX - xFloor) *
                  readWithFillValue(i32(batch), i32(yFloor), i32(xCeil), i32(channel));
                  let valueYCeil = (xCeil - mapX) *
                  readWithFillValue(i32(batch), i32(yCeil), i32(xFloor), i32(channel)) +
                  (mapX - xFloor) *
                  readWithFillValue(i32(batch), i32(yCeil), i32(xCeil), i32(channel));
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(coords[0], coords[1], coords[2], coords[3], outputValue);
            }
          }
        `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function transform(args) {
    const { inputs, backend, attrs } = args;
    const { image, transforms } = inputs;
    const { interpolation, fillMode, fillValue, outputShape } = attrs;
    const [batch, imageHeight, imageWidth, numChannels] = image.shape;
    const [outHeight, outWidth] = outputShape != null ? outputShape : [imageHeight, imageWidth];
    const outShape = [batch, outHeight, outWidth,
        numChannels];
    const program = new TransformProgram(outShape);
    const interpolationModeId = interpolation === 'nearest' ? 1 : 2;
    let fillModeId;
    switch (fillMode) {
        case 'constant':
            fillModeId = 1;
            break;
        case 'reflect':
            fillModeId = 2;
            break;
        case 'wrap':
            fillModeId = 3;
            break;
        case 'nearest':
            fillModeId = 4;
            break;
        default:
            fillModeId = 1;
            break;
    }
    const uniformData = [
        { type: 'int32', data: [interpolationModeId] },
        { type: 'int32', data: [fillModeId] }, { type: 'float32', data: [fillValue] }
    ];
    return backend.runWebGPUProgram(program, [image, transforms], 'float32', uniformData);
}
const transformConfig = {
    kernelName: Transform,
    backendName: 'webgpu',
    kernelFunc: transform
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function unpack(args) {
    const { inputs, backend, attrs } = args;
    const { value } = inputs;
    let { axis } = attrs;
    if (axis < 0) {
        axis += value.shape.length;
    }
    const x = value;
    const xRank = x.shape.length;
    const num = value.shape[axis];
    const outShape = new Array(xRank - 1);
    let outIndex = 0;
    for (let i = 0; i < xRank; i++) {
        if (i !== axis) {
            outShape[outIndex++] = x.shape[i];
        }
    }
    const toDispose = [];
    const begin = new Array(xRank).fill(0);
    const size = x.shape.slice();
    size[axis] = 1;
    const res = new Array(num);
    for (let i = 0; i < res.length; i++) {
        begin[axis] = i;
        const sliced = slice({ inputs: { x }, backend, attrs: { begin, size } });
        const reshaped = reshape({ inputs: { x: sliced }, backend, attrs: { shape: outShape } });
        res[i] = reshaped;
        toDispose.push(sliced);
    }
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return res;
}
const unpackConfig = {
    kernelName: Unpack,
    backendName: 'webgpu',
    kernelFunc: unpack
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// List all kernel configs here
const kernelConfigs = [
    _fusedMatMulConfig,
    absConfig,
    addConfig,
    addNConfig,
    argMaxConfig,
    argMinConfig,
    avgPoolConfig,
    batchMatMulConfig,
    batchToSpaceNDConfig,
    castConfig,
    ceilConfig,
    clipByValueConfig,
    complexConfig,
    concatConfig,
    conv2DConfig,
    conv2DBackpropInputConfig,
    cropAndResizeConfig,
    depthwiseConv2dNativeConfig,
    einsumConfig,
    eluConfig,
    equalConfig,
    expandDimsConfig,
    expConfig,
    expm1Config,
    fillConfig,
    fromPixelsConfig,
    floorConfig,
    floorDivConfig,
    fusedBatchNormConfig,
    fusedConv2DConfig,
    fusedDepthwiseConv2DConfig,
    gatherNdConfig,
    gatherV2Config,
    greaterConfig,
    greaterEqualConfig,
    identityConfig,
    imagConfig,
    lessConfig,
    lessEqualConfig,
    logConfig,
    logicalAndConfig,
    maxConfig,
    maximumConfig,
    maxPoolConfig,
    meanConfig,
    minConfig,
    minimumConfig,
    mirrorPadConfig,
    multiplyConfig,
    negConfig,
    nonMaxSuppressionV3Config,
    nonMaxSuppressionV5Config,
    notEqualConfig,
    onesLikeConfig,
    packConfig,
    padV2Config,
    preluConfig,
    prodConfig,
    powConfig,
    rangeConfig,
    realConfig,
    realDivConfig,
    reluConfig,
    relu6Config,
    reshapeConfig,
    resizeBilinearConfig,
    resizeNearestNeighborConfig,
    rsqrtConfig,
    selectConfig,
    sigmoidConfig,
    sliceConfig,
    stridedSliceConfig,
    stringNGramsConfig,
    softmaxConfig,
    spaceToBatchNDConfig,
    sqrtConfig,
    squareConfig,
    squaredDifferenceConfig,
    subConfig,
    sumConfig,
    tanhConfig,
    tileConfig,
    transformConfig,
    transposeConfig,
    unpackConfig,
    zerosLikeConfig
];
for (const kernelConfig of kernelConfigs) {
    registerKernel(kernelConfig);
}


var Module = (function() {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  return (
function(Module) {
  Module = Module || {};

var d;d||(d=typeof Module !== 'undefined' ? Module : {});d.compileGLSLZeroCopy=function(a,b,c){c=!!c;if("vertex"===b)var e=0;else if("fragment"===b)e=4;else if("compute"===b)e=5;else throw Error("shader_stage must be 'vertex', 'fragment', or 'compute'");b=d._malloc(4);var g=d._malloc(4),f=aa([a,e,c,b,g]);c=ba(b);a=ba(g);d._free(b);d._free(g);if(0===f)throw Error("GLSL compilation failed");b={};g=c/4;b.data=d.HEAPU32.subarray(g,g+a);b.free=function(){d._destroy_output_buffer(f);};return b};
d.compileGLSL=function(a,b,c){a=d.compileGLSLZeroCopy(a,b,c);b=a.data.slice();a.free();return b};var k={},p;for(p in d)d.hasOwnProperty(p)&&(k[p]=d[p]);var ca="./this.program",r=!1,t=!1;r="object"===typeof window;t="function"===typeof importScripts;var u="",w;
if(r||t)t?u=self.location.href:document.currentScript&&(u=document.currentScript.src),_scriptDir&&(u=_scriptDir),0!==u.indexOf("blob:")?u=u.substr(0,u.lastIndexOf("/")+1):u="",t&&(w=function(a){var b=new XMLHttpRequest;b.open("GET",a,!1);b.responseType="arraybuffer";b.send(null);return new Uint8Array(b.response)});var da=d.print||console.log.bind(console),x=d.printErr||console.warn.bind(console);for(p in k)k.hasOwnProperty(p)&&(d[p]=k[p]);k=null;d.thisProgram&&(ca=d.thisProgram);var y;
d.wasmBinary&&(y=d.wasmBinary);"object"!==typeof WebAssembly&&x("no native wasm support detected");function ba(a){var b="i32";"*"===b.charAt(b.length-1)&&(b="i32");switch(b){case "i1":return z[a>>0];case "i8":return z[a>>0];case "i16":return A[a>>1];case "i32":return B[a>>2];case "i64":return B[a>>2];case "float":return C[a>>2];case "double":return D[a>>3];default:E("invalid type for getValue: "+b);}return null}var F,ea=new WebAssembly.Table({initial:861,maximum:861,element:"anyfunc"}),fa=!1;
function ha(){var a=d._convert_glsl_to_spirv;a||E("Assertion failed: Cannot call unknown function convert_glsl_to_spirv, make sure it is exported");return a}
function aa(a){var b=["string","number","boolean","number","number"],c={string:function(a){var b=0;if(null!==a&&void 0!==a&&0!==a){var c=(a.length<<2)+1;b=ia(c);G(a,H,b,c);}return b},array:function(a){var b=ia(a.length);z.set(a,b);return b}},e=ha(),g=[],f=0;if(a)for(var h=0;h<a.length;h++){var m=c[b[h]];m?(0===f&&(f=ja()),g[h]=m(a[h])):g[h]=a[h];}a=e.apply(null,g);0!==f&&ka(f);return a}var la="undefined"!==typeof TextDecoder?new TextDecoder("utf8"):void 0;
function ma(a,b,c){var e=b+c;for(c=b;a[c]&&!(c>=e);)++c;if(16<c-b&&a.subarray&&la)return la.decode(a.subarray(b,c));for(e="";b<c;){var g=a[b++];if(g&128){var f=a[b++]&63;if(192==(g&224))e+=String.fromCharCode((g&31)<<6|f);else {var h=a[b++]&63;g=224==(g&240)?(g&15)<<12|f<<6|h:(g&7)<<18|f<<12|h<<6|a[b++]&63;65536>g?e+=String.fromCharCode(g):(g-=65536,e+=String.fromCharCode(55296|g>>10,56320|g&1023));}}else e+=String.fromCharCode(g);}return e}function I(a){return a?ma(H,a,void 0):""}
function G(a,b,c,e){if(0<e){e=c+e-1;for(var g=0;g<a.length;++g){var f=a.charCodeAt(g);if(55296<=f&&57343>=f){var h=a.charCodeAt(++g);f=65536+((f&1023)<<10)|h&1023;}if(127>=f){if(c>=e)break;b[c++]=f;}else {if(2047>=f){if(c+1>=e)break;b[c++]=192|f>>6;}else {if(65535>=f){if(c+2>=e)break;b[c++]=224|f>>12;}else {if(c+3>=e)break;b[c++]=240|f>>18;b[c++]=128|f>>12&63;}b[c++]=128|f>>6&63;}b[c++]=128|f&63;}}b[c]=0;}}
function na(a){for(var b=0,c=0;c<a.length;++c){var e=a.charCodeAt(c);55296<=e&&57343>=e&&(e=65536+((e&1023)<<10)|a.charCodeAt(++c)&1023);127>=e?++b:b=2047>=e?b+2:65535>=e?b+3:b+4;}return b}"undefined"!==typeof TextDecoder&&new TextDecoder("utf-16le");function oa(a){0<a%65536&&(a+=65536-a%65536);return a}var J,z,H,A,pa,B,K,C,D;
function qa(a){J=a;d.HEAP8=z=new Int8Array(a);d.HEAP16=A=new Int16Array(a);d.HEAP32=B=new Int32Array(a);d.HEAPU8=H=new Uint8Array(a);d.HEAPU16=pa=new Uint16Array(a);d.HEAPU32=K=new Uint32Array(a);d.HEAPF32=C=new Float32Array(a);d.HEAPF64=D=new Float64Array(a);}var ra=d.TOTAL_MEMORY||16777216;d.wasmMemory?F=d.wasmMemory:F=new WebAssembly.Memory({initial:ra/65536});F&&(J=F.buffer);ra=J.byteLength;qa(J);B[79464]=5560896;
function L(a){for(;0<a.length;){var b=a.shift();if("function"==typeof b)b();else {var c=b.T;"number"===typeof c?void 0===b.R?d.dynCall_v(c):d.dynCall_vi(c,b.R):c(void 0===b.R?null:b.R);}}}var sa=[],ta=[],ua=[],va=[];function wa(){var a=d.preRun.shift();sa.unshift(a);}var M=0,N=null;d.preloadedImages={};d.preloadedAudios={};function E(a){if(d.onAbort)d.onAbort(a);da(a);x(a);fa=!0;throw new WebAssembly.RuntimeError("abort("+a+"). Build with -s ASSERTIONS=1 for more info.");}
function ya(){var a=O;return String.prototype.startsWith?a.startsWith("data:application/octet-stream;base64,"):0===a.indexOf("data:application/octet-stream;base64,")}var O=wasmuri;if(!ya()){var za=O;O=d.locateFile?d.locateFile(za,u):u+za;}function Aa(){try{if(y)return new Uint8Array(y);if(w)return w(O);throw "both async and sync fetching of the wasm failed";}catch(a){E(a);}}
function Ba(){return y||!r&&!t||"function"!==typeof fetch?new Promise(function(a){a(Aa());}):fetch(O,{credentials:"same-origin"}).then(function(a){if(!a.ok)throw "failed to load wasm binary file at '"+O+"'";return a.arrayBuffer()}).catch(function(){return Aa()})}ta.push({T:function(){Ca();}});var Da=[null,[],[]],Ea=0;function Fa(){Ea+=4;return B[Ea-4>>2]}var Ga={};
function Ha(a){switch(a){case 1:return 0;case 2:return 1;case 4:return 2;case 8:return 3;default:throw new TypeError("Unknown type size: "+a);}}var Ia=void 0;function P(a){for(var b="";H[a];)b+=Ia[H[a++]];return b}var Ja={},Ka={};function Na(a,b){if(void 0===a)a="_unknown";else {a=a.replace(/[^a-zA-Z0-9_]/g,"$");var c=a.charCodeAt(0);a=48<=c&&57>=c?"_"+a:a;}return (new Function("body","return function "+a+'() {\n    "use strict";    return body.apply(this, arguments);\n};\n'))(b)}
function Oa(a){var b=Error,c=Na(a,function(b){this.name=a;this.message=b;b=Error(b).stack;void 0!==b&&(this.stack=this.toString()+"\n"+b.replace(/^Error(:[^\n]*)?\n/,""));});c.prototype=Object.create(b.prototype);c.prototype.constructor=c;c.prototype.toString=function(){return void 0===this.message?this.name:this.name+": "+this.message};return c}var Pa=void 0;function Q(a){throw new Pa(a);}
function R(a,b,c){c=c||{};if(!("argPackAdvance"in b))throw new TypeError("registerType registeredInstance requires argPackAdvance");var e=b.name;a||Q('type "'+e+'" must have a positive integer typeid pointer');if(Ka.hasOwnProperty(a)){if(c.U)return;Q("Cannot register type '"+e+"' twice");}Ka[a]=b;Ja.hasOwnProperty(a)&&(b=Ja[a],delete Ja[a],b.forEach(function(a){a();}));}var Qa=[],S=[{},{value:void 0},{value:null},{value:!0},{value:!1}];
function Ra(a){switch(a){case void 0:return 1;case null:return 2;case !0:return 3;case !1:return 4;default:var b=Qa.length?Qa.pop():S.length;S[b]={W:1,value:a};return b}}function Sa(a){return this.fromWireType(K[a>>2])}function Ta(a){if(null===a)return "null";var b=typeof a;return "object"===b||"array"===b||"function"===b?a.toString():""+a}
function Ua(a,b){switch(b){case 2:return function(a){return this.fromWireType(C[a>>2])};case 3:return function(a){return this.fromWireType(D[a>>3])};default:throw new TypeError("Unknown float type: "+a);}}
function Va(a,b,c){switch(b){case 0:return c?function(a){return z[a]}:function(a){return H[a]};case 1:return c?function(a){return A[a>>1]}:function(a){return pa[a>>1]};case 2:return c?function(a){return B[a>>2]}:function(a){return K[a>>2]};default:throw new TypeError("Unknown integer type: "+a);}}var Wa={};
function Xa(){if(!Ya){var a={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:("object"===typeof navigator&&navigator.languages&&navigator.languages[0]||"C").replace("-","_")+".UTF-8",_:ca},b;for(b in Wa)a[b]=Wa[b];var c=[];for(b in a)c.push(b+"="+a[b]);Ya=c;}return Ya}var Ya;function T(a){return 0===a%4&&(0!==a%100||0===a%400)}function Za(a,b){for(var c=0,e=0;e<=b;c+=a[e++]);return c}var U=[31,29,31,30,31,30,31,31,30,31,30,31],V=[31,28,31,30,31,30,31,31,30,31,30,31];
function W(a,b){for(a=new Date(a.getTime());0<b;){var c=a.getMonth(),e=(T(a.getFullYear())?U:V)[c];if(b>e-a.getDate())b-=e-a.getDate()+1,a.setDate(1),11>c?a.setMonth(c+1):(a.setMonth(0),a.setFullYear(a.getFullYear()+1));else {a.setDate(a.getDate()+b);break}}return a}
function $a(a,b,c,e){function g(a,b,c){for(a="number"===typeof a?a.toString():a||"";a.length<b;)a=c[0]+a;return a}function f(a,b){return g(a,b,"0")}function h(a,b){function c(a){return 0>a?-1:0<a?1:0}var f;0===(f=c(a.getFullYear()-b.getFullYear()))&&0===(f=c(a.getMonth()-b.getMonth()))&&(f=c(a.getDate()-b.getDate()));return f}function m(a){switch(a.getDay()){case 0:return new Date(a.getFullYear()-1,11,29);case 1:return a;case 2:return new Date(a.getFullYear(),0,3);case 3:return new Date(a.getFullYear(),
0,2);case 4:return new Date(a.getFullYear(),0,1);case 5:return new Date(a.getFullYear()-1,11,31);case 6:return new Date(a.getFullYear()-1,11,30)}}function q(a){a=W(new Date(a.J+1900,0,1),a.P);var b=m(new Date(a.getFullYear()+1,0,4));return 0>=h(m(new Date(a.getFullYear(),0,4)),a)?0>=h(b,a)?a.getFullYear()+1:a.getFullYear():a.getFullYear()-1}var l=B[e+40>>2];e={Z:B[e>>2],Y:B[e+4>>2],N:B[e+8>>2],M:B[e+12>>2],K:B[e+16>>2],J:B[e+20>>2],O:B[e+24>>2],P:B[e+28>>2],ia:B[e+32>>2],X:B[e+36>>2],$:l?I(l):""};
c=I(c);l={"%c":"%a %b %d %H:%M:%S %Y","%D":"%m/%d/%y","%F":"%Y-%m-%d","%h":"%b","%r":"%I:%M:%S %p","%R":"%H:%M","%T":"%H:%M:%S","%x":"%m/%d/%y","%X":"%H:%M:%S","%Ec":"%c","%EC":"%C","%Ex":"%m/%d/%y","%EX":"%H:%M:%S","%Ey":"%y","%EY":"%Y","%Od":"%d","%Oe":"%e","%OH":"%H","%OI":"%I","%Om":"%m","%OM":"%M","%OS":"%S","%Ou":"%u","%OU":"%U","%OV":"%V","%Ow":"%w","%OW":"%W","%Oy":"%y"};for(var n in l)c=c.replace(new RegExp(n,"g"),l[n]);var v="Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
La="January February March April May June July August September October November December".split(" ");l={"%a":function(a){return v[a.O].substring(0,3)},"%A":function(a){return v[a.O]},"%b":function(a){return La[a.K].substring(0,3)},"%B":function(a){return La[a.K]},"%C":function(a){return f((a.J+1900)/100|0,2)},"%d":function(a){return f(a.M,2)},"%e":function(a){return g(a.M,2," ")},"%g":function(a){return q(a).toString().substring(2)},"%G":function(a){return q(a)},"%H":function(a){return f(a.N,2)},
"%I":function(a){a=a.N;0==a?a=12:12<a&&(a-=12);return f(a,2)},"%j":function(a){return f(a.M+Za(T(a.J+1900)?U:V,a.K-1),3)},"%m":function(a){return f(a.K+1,2)},"%M":function(a){return f(a.Y,2)},"%n":function(){return "\n"},"%p":function(a){return 0<=a.N&&12>a.N?"AM":"PM"},"%S":function(a){return f(a.Z,2)},"%t":function(){return "\t"},"%u":function(a){return a.O||7},"%U":function(a){var b=new Date(a.J+1900,0,1),c=0===b.getDay()?b:W(b,7-b.getDay());a=new Date(a.J+1900,a.K,a.M);return 0>h(c,a)?f(Math.ceil((31-
c.getDate()+(Za(T(a.getFullYear())?U:V,a.getMonth()-1)-31)+a.getDate())/7),2):0===h(c,b)?"01":"00"},"%V":function(a){var b=m(new Date(a.J+1900,0,4)),c=m(new Date(a.J+1901,0,4)),e=W(new Date(a.J+1900,0,1),a.P);return 0>h(e,b)?"53":0>=h(c,e)?"01":f(Math.ceil((b.getFullYear()<a.J+1900?a.P+32-b.getDate():a.P+1-b.getDate())/7),2)},"%w":function(a){return a.O},"%W":function(a){var b=new Date(a.J,0,1),c=1===b.getDay()?b:W(b,0===b.getDay()?1:7-b.getDay()+1);a=new Date(a.J+1900,a.K,a.M);return 0>h(c,a)?f(Math.ceil((31-
c.getDate()+(Za(T(a.getFullYear())?U:V,a.getMonth()-1)-31)+a.getDate())/7),2):0===h(c,b)?"01":"00"},"%y":function(a){return (a.J+1900).toString().substring(2)},"%Y":function(a){return a.J+1900},"%z":function(a){a=a.X;var b=0<=a;a=Math.abs(a)/60;return (b?"+":"-")+String("0000"+(a/60*100+a%60)).slice(-4)},"%Z":function(a){return a.$},"%%":function(){return "%"}};for(n in l)0<=c.indexOf(n)&&(c=c.replace(new RegExp(n,"g"),l[n](e)));n=ab(c);if(n.length>b)return 0;z.set(n,a);return n.length-1}
for(var bb=Array(256),X=0;256>X;++X)bb[X]=String.fromCharCode(X);Ia=bb;Pa=d.BindingError=Oa("BindingError");d.InternalError=Oa("InternalError");d.count_emval_handles=function(){for(var a=0,b=5;b<S.length;++b)void 0!==S[b]&&++a;return a};d.get_first_emval=function(){for(var a=5;a<S.length;++a)if(void 0!==S[a])return S[a];return null};function ab(a){var b=Array(na(a)+1);G(a,b,0,b.length);return b}
var db={j:function(){},g:function(){d.___errno_location&&(B[d.___errno_location()>>2]=63);return -1},v:function(a,b){Ea=b;try{var c=Fa();var e=Fa();if(-1===c||0===e)var g=-28;else {var f=Ga.V[c];if(f&&e===f.fa){var h=(void 0).da(f.ca);Ga.ba(c,h,e,f.flags);(void 0).ha(h);Ga.V[c]=null;f.aa&&Y(f.ga);}g=0;}return g}catch(m){return E(m),-m.S}},d:function(){},s:function(a,b,c,e,g){var f=Ha(c);b=P(b);R(a,{name:b,fromWireType:function(a){return !!a},toWireType:function(a,b){return b?e:g},argPackAdvance:8,readValueFromPointer:function(a){if(1===
c)var e=z;else if(2===c)e=A;else if(4===c)e=B;else throw new TypeError("Unknown boolean type size: "+b);return this.fromWireType(e[a>>f])},L:null});},q:function(a,b){b=P(b);R(a,{name:b,fromWireType:function(a){var b=S[a].value;4<a&&0===--S[a].W&&(S[a]=void 0,Qa.push(a));return b},toWireType:function(a,b){return Ra(b)},argPackAdvance:8,readValueFromPointer:Sa,L:null});},e:function(a,b,c){c=Ha(c);b=P(b);R(a,{name:b,fromWireType:function(a){return a},toWireType:function(a,b){if("number"!==typeof b&&"boolean"!==
typeof b)throw new TypeError('Cannot convert "'+Ta(b)+'" to '+this.name);return b},argPackAdvance:8,readValueFromPointer:Ua(b,c),L:null});},b:function(a,b,c,e,g){function f(a){return a}b=P(b);-1===g&&(g=4294967295);var h=Ha(c);if(0===e){var m=32-8*c;f=function(a){return a<<m>>>m};}var q=-1!=b.indexOf("unsigned");R(a,{name:b,fromWireType:f,toWireType:function(a,c){if("number"!==typeof c&&"boolean"!==typeof c)throw new TypeError('Cannot convert "'+Ta(c)+'" to '+this.name);if(c<e||c>g)throw new TypeError('Passing a number "'+
Ta(c)+'" from JS side to C/C++ side to an argument of type "'+b+'", which is outside the valid range ['+e+", "+g+"]!");return q?c>>>0:c|0},argPackAdvance:8,readValueFromPointer:Va(b,h,0!==e),L:null});},a:function(a,b,c){function e(a){a>>=2;var b=K;return new g(b.buffer,b[a+1],b[a])}var g=[Int8Array,Uint8Array,Int16Array,Uint16Array,Int32Array,Uint32Array,Float32Array,Float64Array][b];c=P(c);R(a,{name:c,fromWireType:e,argPackAdvance:8,readValueFromPointer:e},{U:!0});},f:function(a,b){b=P(b);var c="std::string"===
b;R(a,{name:b,fromWireType:function(a){var b=K[a>>2];if(c){var f=H[a+4+b],e=0;0!=f&&(e=f,H[a+4+b]=0);var m=a+4;for(f=0;f<=b;++f){var q=a+4+f;if(0==H[q]){m=I(m);if(void 0===l)var l=m;else l+=String.fromCharCode(0),l+=m;m=q+1;}}0!=e&&(H[a+4+b]=e);}else {l=Array(b);for(f=0;f<b;++f)l[f]=String.fromCharCode(H[a+4+f]);l=l.join("");}Y(a);return l},toWireType:function(a,b){b instanceof ArrayBuffer&&(b=new Uint8Array(b));var f="string"===typeof b;f||b instanceof Uint8Array||b instanceof Uint8ClampedArray||b instanceof
Int8Array||Q("Cannot pass non-string to std::string");var e=(c&&f?function(){return na(b)}:function(){return b.length})(),g=cb(4+e+1);K[g>>2]=e;if(c&&f)G(b,H,g+4,e+1);else if(f)for(f=0;f<e;++f){var q=b.charCodeAt(f);255<q&&(Y(g),Q("String has UTF-16 code units that do not fit in 8 bits"));H[g+4+f]=q;}else for(f=0;f<e;++f)H[g+4+f]=b[f];null!==a&&a.push(Y,g);return g},argPackAdvance:8,readValueFromPointer:Sa,L:function(a){Y(a);}});},r:function(a,b,c){c=P(c);if(2===b){var e=function(){return pa};var g=
1;}else 4===b&&(e=function(){return K},g=2);R(a,{name:c,fromWireType:function(a){for(var b=e(),c=K[a>>2],f=Array(c),l=a+4>>g,n=0;n<c;++n)f[n]=String.fromCharCode(b[l+n]);Y(a);return f.join("")},toWireType:function(a,c){var f=c.length,h=cb(4+f*b),l=e();K[h>>2]=f;for(var n=h+4>>g,v=0;v<f;++v)l[n+v]=c.charCodeAt(v);null!==a&&a.push(Y,h);return h},argPackAdvance:8,readValueFromPointer:Sa,L:function(a){Y(a);}});},t:function(a,b){b=P(b);R(a,{ea:!0,name:b,argPackAdvance:0,fromWireType:function(){},toWireType:function(){}});},
c:function(){E();},n:function(a,b,c){H.set(H.subarray(b,b+c),a);},o:function(a){if(2147418112<a)return !1;for(var b=Math.max(z.length,16777216);b<a;)536870912>=b?b=oa(2*b):b=Math.min(oa((3*b+2147483648)/4),2147418112);a:{try{F.grow(b-J.byteLength+65535>>16);qa(F.buffer);var c=1;break a}catch(e){}c=void 0;}return c?!0:!1},h:function(a,b){var c=0;Xa().forEach(function(e,g){var f=b+c;g=B[a+4*g>>2]=f;for(f=0;f<e.length;++f)z[g++>>0]=e.charCodeAt(f);z[g>>0]=0;c+=e.length+1;});return 0},i:function(a,b){var c=
Xa();B[a>>2]=c.length;var e=0;c.forEach(function(a){e+=a.length+1;});B[b>>2]=e;return 0},l:function(){return 0},m:function(){return 0},k:function(a,b,c,e){try{for(var g=0,f=0;f<c;f++){for(var h=B[b+8*f>>2],m=B[b+(8*f+4)>>2],q=0;q<m;q++){var l=H[h+q],n=Da[a];0===l||10===l?((1===a?da:x)(ma(n,0)),n.length=0):n.push(l);}g+=m;}B[e>>2]=g;return 0}catch(v){return E(v),v.S}},memory:F,w:function(){},p:function(){},u:function(a,b,c,e){return $a(a,b,c,e)},table:ea},eb=function(){function a(a){d.asm=a.exports;M--;
d.monitorRunDependencies&&d.monitorRunDependencies(M);0==M&&(N&&(a=N,N=null,a()));}function b(b){a(b.instance);}function c(a){return Ba().then(function(a){return WebAssembly.instantiate(a,e)}).then(a,function(a){x("failed to asynchronously prepare wasm: "+a);E(a);})}var e={env:db,wasi_unstable:db};M++;d.monitorRunDependencies&&d.monitorRunDependencies(M);if(d.instantiateWasm)try{return d.instantiateWasm(e,a)}catch(g){return x("Module.instantiateWasm callback failed with error: "+
g),!1}(function(){if(y||"function"!==typeof WebAssembly.instantiateStreaming||ya()||"function"!==typeof fetch)return c(b);fetch(O,{credentials:"same-origin"}).then(function(a){return WebAssembly.instantiateStreaming(a,e).then(b,function(a){x("wasm streaming compile failed: "+a);x("falling back to ArrayBuffer instantiation");c(b);})});})();return {}}();d.asm=eb;var Ca=d.___wasm_call_ctors=function(){return d.asm.x.apply(null,arguments)};d._convert_glsl_to_spirv=function(){return d.asm.y.apply(null,arguments)};
d._destroy_output_buffer=function(){return d.asm.z.apply(null,arguments)};var cb=d._malloc=function(){return d.asm.A.apply(null,arguments)},Y=d._free=function(){return d.asm.B.apply(null,arguments)};d.___getTypeName=function(){return d.asm.C.apply(null,arguments)};d.___embind_register_native_and_builtin_types=function(){return d.asm.D.apply(null,arguments)};
var ja=d.stackSave=function(){return d.asm.E.apply(null,arguments)},ia=d.stackAlloc=function(){return d.asm.F.apply(null,arguments)},ka=d.stackRestore=function(){return d.asm.G.apply(null,arguments)};d.dynCall_vi=function(){return d.asm.H.apply(null,arguments)};d.dynCall_v=function(){return d.asm.I.apply(null,arguments)};d.asm=eb;var Z;d.then=function(a){if(Z)a(d);else {var b=d.onRuntimeInitialized;d.onRuntimeInitialized=function(){b&&b();a(d);};}return d};N=function fb(){Z||gb();Z||(N=fb);};
function gb(){function a(){if(!Z&&(Z=!0,!fa)){L(ta);L(ua);if(d.onRuntimeInitialized)d.onRuntimeInitialized();if(d.postRun)for("function"==typeof d.postRun&&(d.postRun=[d.postRun]);d.postRun.length;){var a=d.postRun.shift();va.unshift(a);}L(va);}}if(!(0<M)){if(d.preRun)for("function"==typeof d.preRun&&(d.preRun=[d.preRun]);d.preRun.length;)wa();L(sa);0<M||(d.setStatus?(d.setStatus("Running..."),setTimeout(function(){setTimeout(function(){d.setStatus("");},1);a();},1)):a());}}d.run=gb;
if(d.preInit)for("function"==typeof d.preInit&&(d.preInit=[d.preInit]);0<d.preInit.length;)d.preInit.pop()();gb();


  return Module
}
);
})();
var glslangInit = (() => {
  const initialize = () => {
    return new Promise(resolve => {
      Module({
        locateFile() {
          return wasmuri;
        },
        onRuntimeInitialized() {
          resolve({
            compileGLSLZeroCopy: this.compileGLSLZeroCopy,
            compileGLSL: this.compileGLSL,
          });
        },
      });
    });
  };

  let instance;
  return () => {
    if (!instance) {
      instance = initialize();
    }
    return instance;
  };
})();

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class BufferManager {
    constructor(device) {
        this.device = device;
        this.numUsedBuffers = 0;
        this.numFreeBuffers = 0;
        this.freeBuffers = new Map();
        this.usedBuffers = new Map();
        this.numBytesUsed = 0;
        this.numBytesAllocated = 0;
    }
    acquireBuffer(byteSize, usage) {
        const key = getBufferKey(byteSize, usage);
        if (!this.freeBuffers.has(key)) {
            this.freeBuffers.set(key, []);
        }
        if (!this.usedBuffers.has(key)) {
            this.usedBuffers.set(key, []);
        }
        this.numBytesUsed += byteSize;
        this.numUsedBuffers++;
        if (this.freeBuffers.get(key).length > 0) {
            this.numFreeBuffers--;
            const newBuffer = this.freeBuffers.get(key).shift();
            this.usedBuffers.get(key).push(newBuffer);
            return newBuffer;
        }
        this.numBytesAllocated += byteSize;
        const newBuffer = this.device.createBuffer({ size: byteSize, usage });
        this.usedBuffers.get(key).push(newBuffer);
        return newBuffer;
    }
    releaseBuffer(buffer, byteSize, usage) {
        if (this.freeBuffers == null) {
            return;
        }
        const key = getBufferKey(byteSize, usage);
        if (!this.freeBuffers.has(key)) {
            this.freeBuffers.set(key, []);
        }
        this.freeBuffers.get(key).push(buffer);
        this.numFreeBuffers++;
        this.numUsedBuffers--;
        const bufferList = this.usedBuffers.get(key);
        const bufferIndex = bufferList.indexOf(buffer);
        if (bufferIndex < 0) {
            throw new Error('Cannot release a buffer that was never provided by this ' +
                'buffer manager');
        }
        bufferList.splice(bufferIndex, 1);
        this.numBytesUsed -= byteSize;
    }
    getNumUsedBuffers() {
        return this.numUsedBuffers;
    }
    getNumFreeBuffers() {
        return this.numFreeBuffers;
    }
    reset() {
        this.freeBuffers = new Map();
        this.usedBuffers = new Map();
        this.numUsedBuffers = 0;
        this.numFreeBuffers = 0;
        this.numBytesUsed = 0;
        this.numBytesAllocated = 0;
    }
    dispose() {
        if (this.freeBuffers == null && this.usedBuffers == null) {
            return;
        }
        this.freeBuffers.forEach((buffers, key) => {
            buffers.forEach(buff => {
                buff.destroy();
            });
        });
        this.usedBuffers.forEach((buffers, key) => {
            buffers.forEach(buff => {
                buff.destroy();
            });
        });
        this.freeBuffers = null;
        this.usedBuffers = null;
        this.numUsedBuffers = 0;
        this.numFreeBuffers = 0;
        this.numBytesUsed = 0;
        this.numBytesAllocated = 0;
    }
}
function getBufferKey(byteSize, usage) {
    return `${byteSize}_${usage}`;
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class FromPixelsProgram {
    constructor() {
        this.outputShape = [0];
        this.variableNames = [];
        this.workGroupSize = [256, 1, 1]; // The empirical value.
        this.lastUniformData = [];
        this.inputTexture = null;
        this.layout = null;
        this.lastPixelSize = { width: 0, height: 0 };
        this.disposed = false;
        this.shaderKey = 'fromPixels';
    }
    updateOutputShape(outputShape) {
        if (util.arraysEqual(this.outputShape, outputShape)) {
            return;
        }
        this.outputShape = outputShape;
        this.workPerThread = outputShape[2]; // numChannels in outputShape.
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workGroupSize, [this.workPerThread, 1, 1]);
    }
    getUserCode() {
        const userCode = `
    layout (local_size_x = ${this.workGroupSize[0]},
      local_size_y = 1,
      local_size_z = 1) in;
    layout(set = 0, binding = 1, rgba8) uniform readonly image2D srcImage;
    layout(set = 0, binding = 2) uniform Meta {
      int size;
      int numChannels;
      ivec2 outShapeStrides;
    };

    ivec3 getCoordsFromFlatIndex(int flatIndexBase);

    void main() {
      int flatIndexBase = int(gl_GlobalInvocationID.x) * numChannels;
      ivec3 coords = getCoordsFromFlatIndex(flatIndexBase);
      int texR = coords[0];
      int texC = coords[1];
      int depth = coords[2];
      vec4 values = imageLoad(srcImage, ivec2(texC, texR));
      for(int i = 0; i < numChannels; i++) {
        float value = values[i];
        int flatIndex = flatIndexBase + i;
        if (flatIndex < size) {
          result[flatIndex] = int(floor(255.0 * value));
        }
      }
    }
    `;
        return userCode;
    }
    setPipeline(pipeline) {
        this.pipeline = pipeline;
    }
    setUniform(device, uniformData) {
        // Create the uniform buffer if it does not exist.
        // The uniform buffer size is fixed so we can hold
        // and reuse it always.
        if (!this.uniform) {
            const uniformBuffer = device.createBuffer({
                size: uniformData.length *
                    4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.uniform = uniformBuffer;
        }
        // No need to update uniform buffer if no changes.
        if (!uniformData ||
            ((uniformData.length === this.lastUniformData.length) &&
                uniformData.every((v, i) => v === this.lastUniformData[i]))) {
            return;
        }
        device.queue.writeBuffer(this.uniform, 0, new Uint32Array(uniformData));
        this.lastUniformData = uniformData;
    }
    makeInputTexture(device, pixelWidth, pixelHeight) {
        if (!this.inputTexture || this.lastPixelSize.width !== pixelWidth ||
            this.lastPixelSize.height !== pixelHeight) {
            if (this.inputTexture) {
                this.inputTexture.destroy();
            }
            this.inputTexture = device.createTexture({
                size: [pixelWidth, pixelHeight],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });
            this.lastPixelSize.width = pixelWidth;
            this.lastPixelSize.height = pixelHeight;
        }
        return this.inputTexture;
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        if (this.uniform) {
            this.uniform.destroy();
        }
        if (this.inputTexture) {
            this.inputTexture.destroy();
        }
        this.disposed = true;
    }
    getLayout(device) {
        if (this.layout === null) {
            this.layout = this.createTextureLayout(device);
        }
        return this.layout;
    }
    createTextureLayout(device) {
        const bindGroupLayoutEntries = [];
        // Output buffer binding layout.
        bindGroupLayoutEntries.push({
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        });
        // Input buffer binding layout.
        bindGroupLayoutEntries.push({
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'read-only', format: 'rgba8unorm' }
        });
        // Uniform buffer binding layout.
        bindGroupLayoutEntries.push({
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' }
        });
        const fromPixelBindGroupLayout = device.createBindGroupLayout({ entries: bindGroupLayoutEntries });
        const fromPixelPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [fromPixelBindGroupLayout] });
        return {
            bindGroupLayout: fromPixelBindGroupLayout,
            pipelineLayout: fromPixelPipelineLayout
        };
    }
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class FromPixelsImportProgram extends FromPixelsProgram {
    constructor() {
        super(...arguments);
        this.useWgsl = true;
        this.layout = null;
    }
    getUserCodeWgsl() {
        const userCode = `
    [[binding(1), group(0)]] var src: texture_external;

    [[stage(compute), workgroup_size(${this.workGroupSize[0]}, 1, 1)]]
    fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
      var flatIndexBase = i32(GlobalInvocationID.x) * uniforms.numChannels;
      var coords: vec3<u32> = getCoordsFromFlatIndex(u32(flatIndexBase));
      var texR: i32 = i32(coords[0]);
      var texC: i32 = i32(coords[1]);
      var depth: i32 = i32(coords[2]);
      var values = textureLoad(src, vec2<i32>(texC, texR));
      for (var i: i32 = 0; i < uniforms.numChannels; i = i + 1) {
        var value = values[i];
        var flatIndex = i32(flatIndexBase) + i;
        if (flatIndex < uniforms.size) {
          result.numbers[u32(flatIndex)] = i32(floor(255.0 * value));
        }
      }
    }
`;
        return userCode;
    }
    getLayout(device) {
        if (this.layout === null) {
            this.layout = this.createTextureImportLayout(device);
        }
        return this.layout;
    }
    createTextureImportLayout(device) {
        const bindGroupLayoutEntries = [];
        // Output buffer binding layout.
        bindGroupLayoutEntries.push({
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        });
        // Input buffer binding layout.
        bindGroupLayoutEntries.push({
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            externalTexture: {},
        });
        // Uniform buffer binding layout.
        bindGroupLayoutEntries.push({
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' }
        });
        const fromPixelImportBindGroupLayout = device.createBindGroupLayout({ entries: bindGroupLayoutEntries });
        const fromPixelImportPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [fromPixelImportBindGroupLayout] });
        return {
            bindGroupLayout: fromPixelImportBindGroupLayout,
            pipelineLayout: fromPixelImportPipelineLayout
        };
    }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD = env().getNumber('CPU_HANDOFF_SIZE_THRESHOLD');
class WebGPUBackend extends KernelBackend {
    constructor(device, glslang, supportTimeQuery = false) {
        super();
        this.commandQueueOwnedIds = new WeakSet();
        this.tensorDisposalQueue = [];
        this.uniformDisposalQueue = [];
        this.disposed = false;
        this.uploadWaitMs = 0;
        this.downloadWaitMs = 0;
        this.computePassNumberInEncoder = 0;
        if (!isWebGPUSupported()) {
            throw new Error('WebGPU is not supported on this device');
        }
        this.layoutCache = {};
        this.pipelineCache = {};
        this.device = device;
        this.queue = device.queue;
        this.currentCommandEncoder = null;
        this.glslang = glslang;
        this.supportTimeQuery = supportTimeQuery;
        this.bufferManager = new BufferManager(this.device);
        this.tensorMap = new DataStorage(this, engine());
        if (this.supportTimeQuery) {
            this.querySet = this.device.createQuerySet({
                type: 'timestamp',
                count: 2,
            });
        }
        // Profiling tools like PIX needs this dummy canvas to
        // trigger capturing a frame.
        if (env().getBool('WEBGPU_USE_PROFILE_TOOL')) {
            this.dummyCanvas = document.createElement('canvas');
            this.dummyCanvas.width = 1;
            this.dummyCanvas.height = 1;
            this.dummyContext = this.dummyCanvas.getContext('gpupresent');
            this.dummyContext.configure({
                device,
                format: 'bgra8unorm',
            });
            document.body.appendChild(this.dummyCanvas);
        }
        // Create FromPixelsProgram instance is light weight;
        this.fromPixelProgram = {
            copyExternal: new FromPixelsProgram(),
            import: new FromPixelsImportProgram()
        };
    }
    nextDataId() {
        return WebGPUBackend.nextDataId++;
    }
    floatPrecision() {
        return 32;
    }
    defaultGpuBufferUsage() {
        return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST;
    }
    flushDisposalQueue() {
        this.tensorDisposalQueue.forEach(d => {
            this.maybeReleaseBuffer(d);
            this.tensorMap.delete(d);
        });
        this.uniformDisposalQueue.forEach(d => this.bufferManager.releaseBuffer(d.buffer, d.byteSize, d.usage));
        this.tensorDisposalQueue = [];
        this.uniformDisposalQueue = [];
    }
    /**
     * Dispose the memory if the dataId has 0 refCount. Return true if the memory
     * is released or memory is not managed in this backend, false if memory is
     * not cleared.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    disposeData(dataId, force = false) {
        if (this.tensorMap.has(dataId)) {
            const data = this.tensorMap.get(dataId);
            data.refCount--;
            if (!force && data.refCount > 0) {
                return false;
            }
            if (this.commandQueueOwnedIds.has(dataId)) {
                this.tensorDisposalQueue.push(dataId);
                return false;
            }
            else {
                this.maybeReleaseBuffer(dataId);
            }
            const { complexTensorInfos } = this.tensorMap.get(dataId);
            if (complexTensorInfos != null) {
                this.disposeData(complexTensorInfos.real.dataId, true);
                this.disposeData(complexTensorInfos.imag.dataId, true);
            }
            this.tensorMap.delete(dataId);
        }
        return true;
    }
    memory() {
        return {
            numBytesInGPU: this.bufferManager.numBytesUsed,
            numBytesAllocatedInGPU: this.bufferManager.numBytesAllocated,
            unreliable: false
        };
    }
    getBufferManager() {
        return this.bufferManager;
    }
    acquireBuffer(byteSize, usage = this.defaultGpuBufferUsage()) {
        return this.bufferManager.acquireBuffer(byteSize, usage);
    }
    maybeReleaseBuffer(dataId) {
        const info = this.tensorMap.get(dataId);
        if (info != null && info.bufferInfo.buffer != null) {
            this.bufferManager.releaseBuffer(info.bufferInfo.buffer, info.bufferInfo.byteSize, info.bufferInfo.usage);
            info.bufferInfo.buffer = null;
        }
    }
    /** Return refCount of a `TensorData`. */
    refCount(dataId) {
        if (this.tensorMap.has(dataId)) {
            const tensorData = this.tensorMap.get(dataId);
            return tensorData.refCount;
        }
        return 0;
    }
    /** Increase refCount of a `TensorData`. */
    incRef(dataId) {
        const tensorData = this.tensorMap.get(dataId);
        tensorData.refCount++;
    }
    /** Decrease refCount of a `TensorData`. */
    decRef(dataId) {
        if (this.tensorMap.has(dataId)) {
            const tensorData = this.tensorMap.get(dataId);
            tensorData.refCount--;
        }
    }
    write(values, shape, dtype) {
        if (dtype === 'complex64' && values != null) {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        const dataId = { id: this.nextDataId() };
        const byteSize = util.sizeFromShape(shape) * GPUBytesPerElement(dtype);
        // bool is stored in Uint8Array, converted it to Int32Array.
        if (dtype === 'bool' && values instanceof Uint8Array) {
            values = Int32Array.from(values);
        }
        this.tensorMap.set(dataId, {
            dtype,
            values,
            bufferInfo: { byteSize, usage: this.defaultGpuBufferUsage() },
            refCount: 1
        });
        return dataId;
    }
    move(dataId, values, shape, dtype, refCount) {
        if (dtype === 'complex64') {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        const byteSize = util.sizeFromShape(shape) * GPUBytesPerElement(dtype);
        this.tensorMap.set(dataId, {
            dtype,
            values,
            bufferInfo: { byteSize, usage: this.defaultGpuBufferUsage() },
            refCount
        });
    }
    submitQueue() {
        this.queue.submit([this.currentCommandEncoder.finish()]);
        this.currentCommandEncoder = null;
        this.computePassNumberInEncoder = 0;
        this.commandQueueOwnedIds = new WeakSet();
        this.flushDisposalQueue();
    }
    getBuffer(dataId) {
        this.uploadToGPU(dataId);
        return this.tensorMap.get(dataId).bufferInfo.buffer;
    }
    getFromPixelsProgram(type) {
        switch (type) {
            case 'copyExternal': {
                if (!this.fromPixelProgram.copyExternal) {
                    this.fromPixelProgram.copyExternal = new FromPixelsProgram();
                }
                return this.fromPixelProgram.copyExternal;
            }
            case 'import': {
                if (!this.fromPixelProgram.import) {
                    this.fromPixelProgram.import = new FromPixelsImportProgram();
                }
                return this.fromPixelProgram.import;
            }
            default:
                util.assert(false, () => `Unsupported fromPixels shape`);
                return undefined;
        }
    }
    ensureCommandEncoderReady() {
        if (!this.currentCommandEncoder) {
            this.currentCommandEncoder = this.device.createCommandEncoder();
        }
    }
    async getBufferData(info) {
        if (info.values != null) {
            // Data is on the CPU.
            return info.values;
        }
        const staging = this.acquireBuffer(info.bufferInfo.byteSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        this.ensureCommandEncoderReady();
        this.currentCommandEncoder.copyBufferToBuffer(info.bufferInfo.buffer, 0, staging, 0, info.bufferInfo.byteSize);
        this.submitQueue();
        await staging.mapAsync(GPUMapMode.READ);
        const values = staging.getMappedRange().slice(0);
        staging.unmap();
        if (staging != null) {
            this.bufferManager.releaseBuffer(staging, info.bufferInfo.byteSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        }
        // Need to get texture from swapChain to enable profiling tool
        // to capture a frame
        if (env().getBool('WEBGPU_USE_PROFILE_TOOL')) {
            util.assert(this.dummyContext !== undefined, () => `Fail to get context for profiling tool`);
            this.dummyContext.getCurrentTexture();
        }
        return values;
    }
    convertAndCacheOnCPU(dataId, data) {
        const info = this.tensorMap.get(dataId);
        this.maybeReleaseBuffer(dataId);
        info.values = data;
        return info.values;
    }
    // TODO: Remove once this is fixed:
    // https://github.com/tensorflow/tfjs/issues/1595
    readSync(dataId) {
        const texData = this.tensorMap.get(dataId);
        const { values } = texData;
        if (values == null) {
            throw new Error('WebGPU readSync is only available for CPU-resident tensors.');
        }
        return values;
    }
    async read(dataId) {
        if (!this.tensorMap.has(dataId)) {
            throw new Error(`Tensor ${dataId} was not registered!`);
        }
        const info = this.tensorMap.get(dataId);
        const { values } = info;
        if (values != null) {
            // TODO(xing.xu@intel.com): Merge backend_util.BackendValues and
            // backend_util.TypedArray.
            return this.convertAndCacheOnCPU(dataId, values);
        }
        // Download the values from the GPU.
        let vals;
        if (info.dtype === 'complex64') {
            const ps = await Promise.all([
                this.read(info.complexTensorInfos.real.dataId),
                this.read(info.complexTensorInfos.imag.dataId)
            ]);
            const realValues = ps[0];
            const imagValues = ps[1];
            vals = backend_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        else {
            const data = await this.getBufferData(info);
            vals =
                ArrayBufferToTypedArray(data, info.dtype);
        }
        this.convertAndCacheOnCPU(dataId, vals);
        return vals;
    }
    bufferSync(t) {
        const data = this.readSync(t.dataId);
        let decodedData = data;
        if (t.dtype === 'string') {
            try {
                // Decode the bytes into string.
                decodedData = data.map(d => util.decodeString(d));
            }
            catch (_a) {
                throw new Error('Failed to decode encoded string bytes into utf-8');
            }
        }
        return buffer(t.shape, t.dtype, decodedData);
    }
    async time(f) {
        const oldActiveTimers = this.activeTimers;
        const newActiveTimers = [];
        let outerMostTime = false;
        if (this.programTimersStack == null) {
            this.programTimersStack = newActiveTimers;
            outerMostTime = true;
        }
        else {
            this.activeTimers.push(newActiveTimers);
        }
        this.activeTimers = newActiveTimers;
        f();
        const flattenedActiveTimerQueries = util.flatten(this.activeTimers.map((d) => d.query))
            .filter(d => d != null);
        const flattenedActiveTimerNames = util.flatten(this.activeTimers.map((d) => d.name))
            .filter(d => d != null);
        this.activeTimers = oldActiveTimers;
        if (outerMostTime) {
            this.programTimersStack = null;
        }
        const res = {
            uploadWaitMs: this.uploadWaitMs,
            downloadWaitMs: this.downloadWaitMs,
            kernelMs: null,
            wallMs: null
        };
        const kernelMs = await Promise.all(flattenedActiveTimerQueries);
        res['kernelMs'] = util.sum(kernelMs);
        res['getExtraProfileInfo'] = () => kernelMs.map((d, i) => ({ name: flattenedActiveTimerNames[i], ms: d }))
            .map(d => `${d.name}: ${d.ms}`)
            .join(', ');
        this.uploadWaitMs = 0;
        this.downloadWaitMs = 0;
        return res;
    }
    getAndSavePipeline(key, getPipeline) {
        if (!(key in this.pipelineCache)) {
            this.pipelineCache[key] = getPipeline();
        }
        return this.pipelineCache[key];
    }
    makeTensorInfo(shape, dtype, values) {
        let dataId;
        if (dtype === 'string' && values != null && values.length > 0 &&
            util.isString(values[0])) {
            const encodedValues = values.map(d => util.encodeString(d));
            dataId = this.write(encodedValues, shape, dtype);
        }
        else {
            dataId = this.write(values, shape, dtype);
        }
        return { dataId, shape, dtype };
    }
    tensorToBinding(tensor) {
        if (!tensor) {
            return null;
        }
        const tensorData = this.tensorMap.get(tensor.dataId);
        return {
            offset: 0,
            size: tensorData.bufferInfo.byteSize,
            buffer: tensorData.bufferInfo.buffer
        };
    }
    async getQueryTime(query) {
        if (this.supportTimeQuery) {
            return this.getTimeFromQuerySet(query);
        }
        else {
            return 0;
        }
    }
    uploadToGPU(dataId) {
        const info = this.tensorMap.get(dataId);
        if (info.bufferInfo.buffer != null) {
            // Already on the GPU.
            return;
        }
        info.bufferInfo.buffer = this.acquireBuffer(info.bufferInfo.byteSize);
        if (info.values) {
            this.queue.writeBuffer(info.bufferInfo.buffer, 0, info.values);
            // TODO: WebGPU doesn't support read data synchronously from GPU to CPU.
            // So it will report error when switching backend from WebGPU to others.
            // There are two situations: 1) swithcing the backend after running a
            // model; 2) swithcing the backend within the model. Temporarilly keep the
            // values on CPU to solve the first issue.
            // info.values = null;
        }
    }
    makeUniformsDataView(data) {
        const dimensionsBuffer = this.acquireBuffer(data.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
        this.queue.writeBuffer(dimensionsBuffer, 0, data);
        return { offset: 0, size: data.byteLength, buffer: dimensionsBuffer };
    }
    arrayToDataView(arrays, length) {
        const BYTES_PER_ELEMENT = 4;
        const uniformDataView = new DataView(new ArrayBuffer(length * BYTES_PER_ELEMENT));
        let dataViewIndex = 0;
        arrays.forEach(array => {
            const arrayData = array.data;
            if (array.type !== 'int32' && array.type !== 'float32' &&
                array.type !== 'uint32') {
                throw new Error(`${array.type} not supported!`);
            }
            if (array.type === 'int32') {
                arrayData.forEach(d => {
                    uniformDataView.setInt32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
                    dataViewIndex++;
                });
            }
            else if (array.type === 'uint32') {
                arrayData.forEach(d => {
                    uniformDataView.setUint32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
                    dataViewIndex++;
                });
            }
            else {
                arrayData.forEach(d => {
                    uniformDataView.setFloat32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
                    dataViewIndex++;
                });
            }
        });
        return uniformDataView;
    }
    computePadding(uniformsWithType) {
        let currentOffset = 0;
        let padding = 0;
        let dataViewIndex = 0;
        const dimUniformsData = [];
        uniformsWithType.forEach((d, i) => {
            if (d.data.length === 0) {
                d.data = [1];
            }
            // Complete std140 layout rules are documented here:
            // tslint:disable-next-line:max-line-length
            // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
            let baseAlignment;
            switch (d.data.length) {
                case 0:
                    baseAlignment = 1;
                    break;
                case 1:
                    baseAlignment = 1;
                    break;
                case 2:
                    baseAlignment = 2;
                    break;
                case 3:
                    baseAlignment = 4;
                    break;
                case 4:
                    baseAlignment = 4;
                    break;
                default:
                    util.assert(false, () => `Unsupported ${d.data.length}D shape`);
            }
            padding = Math.ceil(currentOffset / baseAlignment) * baseAlignment -
                currentOffset;
            for (let p = 0; p < padding; ++p) {
                dimUniformsData.push({ type: d.type, data: [0] });
                dataViewIndex++;
            }
            dimUniformsData.push({ type: d.type, data: d.data });
            dataViewIndex = dataViewIndex + d.data.length;
            currentOffset += d.data.length + padding;
        });
        return this.arrayToDataView(dimUniformsData, dataViewIndex);
    }
    // This layout is used by all programs except fromPixel.
    createLayout(inputEntrySize) {
        const bindGroupLayoutEntries = [];
        // Output buffer binding layout.
        bindGroupLayoutEntries.push({
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        });
        // Input buffer binding layout. Depends on variableNames length.
        for (let i = 0; i < inputEntrySize; i++) {
            bindGroupLayoutEntries.push({
                binding: i + 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' }
            });
        }
        bindGroupLayoutEntries.push({
            binding: inputEntrySize + 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' }
        });
        const bindGroupLayout = this.device.createBindGroupLayout({ entries: bindGroupLayoutEntries });
        const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
        return { bindGroupLayout, pipelineLayout };
    }
    getCachedOrCreateLayout(inputEntrySize) {
        if (!(inputEntrySize in this.layoutCache)) {
            this.layoutCache[inputEntrySize] = this.createLayout(inputEntrySize);
        }
        return this.layoutCache[inputEntrySize];
    }
    runWebGPUProgram(program, inputs, outputDtype, programUniforms) {
        const output = this.makeTensorInfo(program.outputShape, outputDtype);
        const outData = this.tensorMap.get(output.dataId);
        if (util.sizeFromShape(output.shape) === 0) {
            // Short-circuit the computation since the result is empty (has 0 in its
            // shape).
            outData.values =
                util.getTypedArrayFromDType(output.dtype, 0);
            return output;
        }
        // There are five kinds of uniforms: NAN, shapes, shape strides, program
        // size, program defined uniforms.
        let uniformsWithType = [{ type: 'float32', data: [NaN] }];
        const bufferShapes = inputs.concat(output).map(d => d.shape);
        let uniformsType = 'int32';
        if (program.useWgsl) {
            uniformsType = 'uint32';
        }
        bufferShapes.map(d => {
            uniformsWithType.push({ type: uniformsType, data: d });
        });
        const strides = util.computeStrides(output.shape);
        uniformsWithType.push({ type: uniformsType, data: strides });
        if (program.size != null) {
            uniformsWithType.push({ type: uniformsType, data: [program.size] });
        }
        uniformsWithType.push({ type: 'int32', data: program.dispatch });
        if (programUniforms) {
            uniformsWithType = [...uniformsWithType, ...programUniforms];
        }
        let uniforms = null;
        const uniformsDataView = this.computePadding(uniformsWithType);
        const uniformsByteLength = uniformsDataView.byteLength;
        uniforms = this.makeUniformsDataView(uniformsDataView);
        const inputsData = inputs.map((input, i) => {
            if (input.dtype === 'complex64') {
                throw new Error(`GPGPUProgram does not support complex64 input. For complex64 ` +
                    `dtypes, please separate the program into real and imaginary ` +
                    `parts.`);
            }
            this.uploadToGPU(input.dataId);
            return {
                // Returning dtype from tensorMap because it reflects dtype
                // of underlying buffer, rather than abstract dtype.
                dtype: this.tensorMap.get(input.dataId).dtype,
                shape: input.shape,
                name: program.variableNames[i]
            };
        });
        this.uploadToGPU(output.dataId);
        const bufferTypes = inputsData.map(d => d.dtype).concat(output.dtype);
        const broadcastDims = inputsData.map(d => backend_util.getBroadcastDims(d.shape, output.shape));
        const inputShapesEqualsOutShape = inputsData.map(d => util.arraysEqual(d.shape, output.shape)).join('_');
        const broadcastDimsKey = broadcastDims.map(d => d.join('_')).join(';');
        const key = makeShaderKey(program, bufferShapes, bufferTypes, broadcastDimsKey, inputShapesEqualsOutShape);
        const { bindGroupLayout, pipelineLayout } = this.getCachedOrCreateLayout(program.variableNames.length);
        const pipeline = this.getAndSavePipeline(key, () => {
            return compileProgram(this.glslang, this.device, program, pipelineLayout, inputsData, output);
        });
        const shouldTimeProgram = this.activeTimers != null;
        // Creating bind groups on the fly should never be a bottleneck.
        const bg = makeBindGroup(this.device, bindGroupLayout, inputs.map(t => this.tensorToBinding(t)), this.tensorToBinding(output), uniforms);
        this.ensureCommandEncoderReady();
        const pass = this.currentCommandEncoder.beginComputePass();
        if (shouldTimeProgram) {
            if (this.supportTimeQuery) {
                pass.writeTimestamp(this.querySet, 0);
            }
        }
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatch(program.dispatch[0], program.dispatch[1], program.dispatch[2]);
        if (shouldTimeProgram) {
            if (this.supportTimeQuery) {
                pass.writeTimestamp(this.querySet, 1);
            }
        }
        pass.endPass();
        this.computePassNumberInEncoder++;
        inputs.forEach(input => {
            this.commandQueueOwnedIds.add(input.dataId);
        });
        this.commandQueueOwnedIds.add(output.dataId);
        if (uniforms) {
            const uniformInfo = {
                byteSize: uniformsByteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
                buffer: uniforms.buffer
            };
            this.uniformDisposalQueue.push(uniformInfo);
        }
        if (env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE') <= this.computePassNumberInEncoder) {
            this.submitQueue();
        }
        if (shouldTimeProgram) {
            this.activeTimers.push({
                name: program.constructor.name,
                query: this.getQueryTime(this.querySet)
            });
        }
        return output;
    }
    runFromPixelsProgram(program, output, layout, externalResource, outputId) {
        const bindGroup = this.device.createBindGroup({
            layout: layout.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: output,
                    }
                },
                {
                    binding: 1,
                    resource: externalResource,
                },
                {
                    binding: 2,
                    resource: {
                        buffer: program.uniform,
                    }
                }
            ],
        });
        this.ensureCommandEncoderReady();
        const passEncoder = this.currentCommandEncoder.beginComputePass();
        const shouldTimeProgram = this.activeTimers != null;
        if (shouldTimeProgram) {
            if (this.supportTimeQuery) {
                passEncoder.writeTimestamp(this.querySet, 0);
            }
        }
        passEncoder.setPipeline(program.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatch(program.dispatch[0], program.dispatch[1], program.dispatch[2]);
        if (shouldTimeProgram) {
            if (this.supportTimeQuery) {
                passEncoder.writeTimestamp(this.querySet, 1);
            }
        }
        passEncoder.endPass();
        this.commandQueueOwnedIds.add(outputId);
        this.submitQueue();
        if (shouldTimeProgram) {
            this.activeTimers.push({
                name: program.constructor.name,
                query: this.getQueryTime(this.querySet)
            });
        }
    }
    async getTimeFromQuerySet(querySet) {
        const queryBuffer = this.acquireBuffer(16, GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE);
        const dst = this.acquireBuffer(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        this.ensureCommandEncoderReady();
        this.currentCommandEncoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);
        this.currentCommandEncoder.copyBufferToBuffer(queryBuffer, 0, dst, 0, 16);
        this.submitQueue();
        await dst.mapAsync(GPUMapMode.READ);
        const arrayBuf = new BigUint64Array(dst.getMappedRange());
        const timeElapsedNanos = Number((arrayBuf[1] - arrayBuf[0]));
        dst.unmap();
        this.bufferManager.releaseBuffer(dst, 16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        this.bufferManager.releaseBuffer(queryBuffer, 16, GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE);
        // Return milliseconds.
        return timeElapsedNanos / 1000000;
    }
    shouldExecuteOnCPU(inputs, sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD) {
        return env().getBool('WEBGPU_CPU_FORWARD') &&
            inputs.every(input => this.tensorMap.get(input.dataId).bufferInfo.buffer == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
    }
    numDataIds() {
        return this.tensorMap.numDataIds() - this.tensorDisposalQueue.length;
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        this.bufferManager.dispose();
        if (this.fromPixelProgram.copyExternal) {
            this.fromPixelProgram.copyExternal.dispose();
        }
        if (this.fromPixelProgram.import) {
            this.fromPixelProgram.import.dispose();
        }
        this.disposed = true;
    }
}
WebGPUBackend.nextDataId = 0;

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

var webgpu = /*#__PURE__*/Object.freeze({
  __proto__: null,
  webgpu_util: webgpu_util,
  WebGPUBackend: WebGPUBackend
});

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
if (device_util.isBrowser() && isWebGPUSupported()) {
    registerBackend('webgpu', async () => {
        // Remove it once we figure out how to correctly read the tensor data
        // before the tensor is disposed in profiling mode.
        env().set('CHECK_COMPUTATION_FOR_ERRORS', false);
        const glslang = await glslangInit();
        const gpuDescriptor = {
            powerPreference: env().get('WEBGPU_USE_LOW_POWER_GPU') ?
                'low-power' :
                'high-performance'
        };
        const adapter = await navigator.gpu.requestAdapter(gpuDescriptor);
        let deviceDescriptor = {};
        const supportTimeQuery = adapter.features.has('timestamp-query');
        if (supportTimeQuery) {
            deviceDescriptor = { requiredFeatures: ['timestamp-query'] };
        }
        else {
            console.warn(`This device doesn't support timestamp-query extension. ` +
                `Start Chrome browser with flag ` +
                `--disable-dawn-features=disallow_unsafe_apis then try again. ` +
                `Or zero will shown for the kernel time when profiling mode is` +
                `enabled. Using performance.now is not workable for webgpu since` +
                `it doesn't support synchronously to read data from GPU.`);
        }
        const device = await adapter.requestDevice(deviceDescriptor);
        return new WebGPUBackend(device, glslang, supportTimeQuery);
    }, 3 /*priority*/);
}

export { webgpu };
//# sourceMappingURL=tf-backend-webgpu.fesm.js.map