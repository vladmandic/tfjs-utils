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
  log.info('metadata:', { generatedBy: model.artifacts.generatedBy, convertedBy: model.artifacts.convertedBy, version });

  async function analyzeInputs() {
    const inputs = [];
    if (model.modelSignature && model.modelSignature['inputs']) {
      log.info('model inputs based on signature');
      for (const [key, val] of Object.entries(model.modelSignature['inputs'])) {
        const shape = val.tensorShape.dim.map((a) => parseInt(a.size));
        inputs.push({ name: key, dtype: val.dtype, shape });
      }
    // @ts-ignore
    } else if (model.executor.graph['inputs']) {
      log.info('model inputs based on executor');
      // @ts-ignore
      for (const t of model.executor.graph['inputs']) {
        inputs.push({ name: t.name, dtype: t.attrParams.dtype.value, shape: t.attrParams.shape.value });
      }
    } else {
      log.warn('model inputs: cannot determine');
    }
    return inputs;
  }

  async function analyzeOutputs() {
    const outputs = [];
    let i = 0;
    if (model.modelSignature && model.modelSignature['outputs'] && Object.values(model.modelSignature['outputs'])[0].dtype) {
      log.info('model outputs based on signature');
      for (const [key, val] of Object.entries(model.modelSignature['outputs'])) {
        const shape = val.tensorShape?.dim.map((a) => parseInt(a.size));
        outputs.push({ id: i++, name: key, dytpe: val.dtype, shape });
      }
    // @ts-ignore
    } else if (model.executor.graph['outputs']) {
      log.info('model outputs based on executor');
      // @ts-ignore
      for (const t of model.executor.graph['outputs']) {
        outputs.push({ id: i++, name: t.name, dtype: t.attrParams.dtype?.value || t.rawAttrs.T.type, shape: t.attrParams.shape?.value });
      }
    } else {
      log.warn('model outputs: cannot determine');
    }
    return outputs;
  }

  async function analyzeKernelOps() {
    const ops = {};
    // @ts-ignore
    for (const op of Object.values(model.executor.graph.nodes)) {
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

  for (const input of await analyzeInputs()) log.blank('', input);
  for (const output of await analyzeOutputs()) log.blank('', output);
  log.info('tensors:', tf.engine().memory().numTensors);
  log.data('weights:', await analyzeWeights());
  log.data('kernel ops:', await analyzeKernelOps());
}

async function analyzeSaved(modelPath) {
  const meta = await tf.node.getMetaGraphsFromSavedModel(modelPath);
  log.info('saved model:', path.resolve(modelPath));
  const sign = Object.values(meta[0].signatureDefs)[0];
  log.data('tags:', meta[0].tags);
  log.data('signature:', Object.keys(meta[0].signatureDefs));
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
  log.options.timeStamp = false;
  const param = process.argv[2];
  if (process.argv.length !== 3) {
    log.error('path required');
    process.exit(0);
  } else if (!fs.existsSync(param)) {
    log.error(`path does not exist: ${param}`);
    process.exit(0);
  }
  const stat = fs.statSync(param);
  if (stat.isFile()) {
    if (param.endsWith('.json')) analyzeGraph(param);
  }
  if (stat.isDirectory()) {
    if (fs.existsSync(path.join(param, '/saved_model.pb'))) analyzeSaved(param);
    if (fs.existsSync(path.join(param, '/model.json'))) analyzeGraph(path.join(param, '/model.json'));
  }
}

main();
