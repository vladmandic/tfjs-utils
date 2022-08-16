/**
 * Analyze SavedModel or GraphModel input/output tensors
 * Based on either model signature or model executor
 *
 * @param modelPath: string
 */

const fs = require('fs');
const path = require('path');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');

const excludedKernelOps = ['Placeholder'];

async function analyzeGraph(modelPath) {
  log.info('graph model:', path.resolve(modelPath));
  const stat = fs.statSync(modelPath);
  log.info('created on:', stat.birthtime);
  let model;
  try {
    model = await tf.loadGraphModel(`file://${modelPath}`);
  } catch (err) {
    log.error('graph model load error:', err.message);
  }
  if (!model) return;
  const version = model.version === 'undefined.undefined' ? undefined : model.version;
  log.info('metadata');
  log.blank({ generatedBy: model.artifacts.generatedBy, convertedBy: model.artifacts.convertedBy, version });

  async function analyzeInputs() {
    const inputsSig = [];
    if (model.modelSignature?.['inputs']) {
      log.info('model inputs based on signature');
      for (const [key, val] of Object.entries(model.modelSignature['inputs'])) {
        const shape = val.tensorShape.dim.map((a) => parseInt(a.size));
        inputsSig.push({ name: key, dtype: val.dtype, shape });
      }
      for (const input of inputsSig) log.blank('', input);
    } else {
      log.warn('model inputs: no signature found');
    }
    const inputsExe = [];
    if (model.executor?.graph?.['inputs']) {
      log.info('model inputs based on executor');
      // @ts-ignore
      for (const t of model.executor.graph['inputs']) {
        inputsExe.push({ name: t.name, dtype: t.attrParams.dtype.value, shape: t.attrParams.shape.value });
      }
      for (const input of inputsExe) log.blank('', input);
    } else {
      log.warn('model inputs: no executor data found');
    }
    return inputsSig.length > 0 ? inputsSig : inputsExe;
  }

  async function analyzeOutputs() {
    const outputsSig = [];
    let i = 0;
    if (model.modelSignature?.['outputs'] && Object.values(model.modelSignature?.['outputs'])[0]?.dtype) {
      log.info('model outputs based on signature');
      for (const [key, val] of Object.entries(model.modelSignature['outputs'])) {
        const shape = val.tensorShape?.dim.map((a) => parseInt(a.size));
        outputsSig.push({ id: i++, name: key, dytpe: val.dtype, shape });
      }
      for (const output of outputsSig) log.blank('', output);
    } else {
      log.warn('model outputs: no signature found');
    }
    const outputsExe = [];
    i = 0;
    if (model.executor?.graph?.['outputs']) {
      log.info('model outputs based on executor');
      for (const t of model.executor.graph['outputs']) {
        const shape = t.attrParams.shape?.value;
        outputsExe.push({ id: i++, name: t.name, dtype: t.attrParams.dtype?.value || t.rawAttrs.T.type, shape });
      }
      for (const output of outputsExe) log.blank('', output);
    } else {
      log.warn('model outputs: no executor data found');
    }
    return outputsSig.length > 0 ? outputsSig : outputsExe;
  }

  async function analyzeKernelOps() {
    const ops = {};
    // @ts-ignore
    for (const op of Object.values(model.executor.graph.nodes)) {
      if (excludedKernelOps.includes(op.op)) continue;
      if (!ops[op.category]) ops[op.category] = [];
      if (!ops[op.category].includes(op.op)) ops[op.category].push(op.op);
    }
    return ops;
  }

  async function analyzeWeights() {
    const weights = [];
    for (const [name, tensors] of Object.entries(model.weights)) {
      for (const weight of tensors) weights.push({ weight: name, dtype: weight.dtype, size: weight.size, shape: weight.shape });
    }
    const weightsSize = weights.reduce((prev, curr) => prev + curr.size, 0);
    const weightTypes = [];
    const weightSizes = [];
    for (const weight of weights) {
      if (weightTypes[weight.dtype]) weightTypes[weight.dtype]++;
      else weightTypes[weight.dtype] = 1;
      if (weightSizes[weight.dtype]) weightSizes[weight.dtype] += weight.size;
      else weightSizes[weight.dtype] = weight.size;
    }
    const weightQuant = [];
    for (const weight of model.artifacts.weightSpecs) weightQuant.push({ runtime: weight.dtype, original: weight.quantization?.original_dtype, quant: weight.quantization?.dtype || 'none' });
    const weightQuantTypes = [];
    for (const weight of weightQuant) {
      if (weightQuantTypes[weight.quant]) weightQuantTypes[weight.quant]++;
      else weightQuantTypes[weight.quant] = 1;
    }
    const data = fs.readFileSync(modelPath);
    const json = JSON.parse(data);
    const weightFiles = [];
    for (const weight of json.weightsManifest) weightFiles.push(...weight.paths);
    return {
      files: weightFiles,
      size: { disk: model.artifacts.weightData.byteLength, memory: tf.engine().memory().numBytes },
      count: { total: weights.length, ...weightTypes },
      quantized: { ...weightQuantTypes },
      values: { total: weightsSize, ...weightSizes },
    };
  }

  async function analyzeTopology() {
    const nodes = model.artifacts?.modelTopology?.node;
    if (!nodes) return undefined;
    /*
    let executionOps = 0;
    const maxDepth = 100;
    const maxChildren = 1;
    const walkExecutor = (node) => {
      if (typeof walkExecutor.depth === 'undefined') walkExecutor.depth = 0;
      else walkExecutor.depth++;
      executionOps += 1;
      if (node && node.children && Array.isArray(node.children)) {
        if (walkExecutor.depth < maxDepth) {
          for (let i = 0; (i < node.children.length) && (i < maxChildren); i++) walkExecutor(node.children[i]);
        }
      }
      walkExecutor.depth--;
    };
    walkExecutor(model.executor.graph.placeholders[0]);
    */
    return {
      nodes: nodes.length,
      // executionOps,
    };
  }

  function analyzeProfiling(profile) {
    if (!profile) return [];
    const kernels = {};
    let total = 0;
    for (const kernel of profile.kernels) { // sum kernel time values per kernel
      if (kernels[kernel.name]) kernels[kernel.name] += kernel.kernelTimeMs;
      else kernels[kernel.name] = kernel.kernelTimeMs;
      total += kernel.kernelTimeMs;
    }
    const kernelArr = [];
    Object.entries(kernels).forEach((key) => kernelArr.push({ kernel: key[0], time: key[1], perc: 0 })); // convert to array
    for (const kernel of kernelArr) {
      kernel.perc = Math.round(1000 * kernel.time / total) / 1000;
      kernel.time = Math.round(1000 * kernel.time) / 1000;
    }
    kernelArr.sort((a, b) => b.time - a.time); // sort
    if (kernelArr.length > 5) kernelArr.length = 5; // crop
    return kernelArr;
  }

  async function analyzeExecution(input) {
    if (input.dtype.includes('DT_FLOAT')) input.dtype = 'float32';
    if (input.dtype.includes('DT_INT') || input.dtype.includes('DT_UINT')) input.dtype = 'int32';
    let requireAsync;
    let success = false;
    const tensor = tf.randomUniform(input.shape, 0, 1, input.dtype);
    const t0 = process.hrtime.bigint();
    const profile = await tf.profile(async () => {
      if (input.shape.length > 0) input.shape[0] = Math.abs(input.shape[0]);
      // const tensor = tf.zeros(input.shape, input.dtype);
      let res;
      if (!success) {
        try {
          res = model.execute(tensor);
          success = true;
          requireAsync = false;
        } catch { /**/ }
      }
      if (!success) {
        try {
          res = await model.executeAsync(tensor);
          success = true;
          requireAsync = true;
        } catch { /**/ }
      }
      if (Array.isArray(res)) tf.dispose(...res);
      else tf.dispose(res);
    });
    const t1 = process.hrtime.bigint();
    tf.dispose(tensor);
    // eslint-disable-next-line no-return-assign
    const kernelTime = profile ? profile.kernels.reduce((prev, curr) => prev += curr.kernelTimeMs, 0) : undefined;
    const topKernels = analyzeProfiling(profile);
    if (success) {
      return {
        requireAsync,
        wallTime: Math.round(100 * Number(t1 - t0) / 1000000) / 100,
        kernelTime: Math.round(100 * kernelTime) / 100,
        numKernels: profile?.kernels?.length,
        peakBytes: profile?.peakBytes,
        topKernels,
      };
    }
    return { success };
  }

  const inputs = await analyzeInputs();
  await analyzeOutputs();
  log.info('tensors:', tf.engine().memory().numTensors);
  log.data('topology:', await analyzeTopology());
  log.data('weights:', await analyzeWeights());
  log.data('kernel ops:', await analyzeKernelOps());
  for (const input of inputs) log.data('execution profile:', input, await analyzeExecution(input));
}

async function analyzeSaved(modelPath) {
  const meta = await tf.node.getMetaGraphsFromSavedModel(modelPath);
  log.info('saved model:', path.resolve(modelPath));
  log.data('tags:', meta[0].tags);
  log.data('signature:', Object.keys(meta[0].signatureDefs));
  const sign = Object.values(meta[0].signatureDefs)[0];
  if (!sign) {
    log.error('model is missing full signature');
    return;
  }
  const inputs = Object.values(sign.inputs)[0];
  // @ts-ignore
  const inputShape = inputs.shape?.map((a) => a.array[0]);
  log.data('inputs:', { name: inputs.name, dtype: inputs.dtype, shape: inputShape });
  const outputs = [];
  let i = 0;
  for (const [key, val] of Object.entries(sign.outputs)) {
    // @ts-ignore
    const shape = val.shape?.map((a) => a.array[0]);
    outputs.push({ id: i++, name: key, dytpe: val.dtype, shape });
  }
  log.data('outputs:', outputs);
}

async function main() {
  // log.options.timeStamp = false;
  // log.options.inspect.breakLength = 300;
  log.configure({ timeStamp: false, inspect: { breakLength: 200, compact: 3, showProxy: true } });
  const param = process.argv[2];
  if (process.argv.length !== 3) {
    log.error('path required');
    process.exit(1);
  } else if (!fs.existsSync(param)) {
    log.error(`path does not exist: ${param}`);
    process.exit(1);
  }
  const stat = fs.statSync(param);
  if (stat.isFile()) {
    if (param.endsWith('.json')) await analyzeGraph(param);
  }
  if (stat.isDirectory()) {
    if (fs.existsSync(path.join(param, '/saved_model.pb'))) await analyzeSaved(param);
    if (fs.existsSync(path.join(param, '/model.json'))) await analyzeGraph(path.join(param, '/model.json'));
  }
  process.exit(0);
}

main();
