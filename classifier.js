const fs = require('fs');
const path = require('path');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');

// app options
let inputSize = 224;
const divFactor = 1.0;
const minScore = 0.1;

function getTensorFromImage(image, dtype) {
  if (!fs.existsSync(image)) {
    log.error('Not found:', image);
    return tf.zeros(inputSize[0]);
  }
  // load & decode jpeg
  const data = fs.readFileSync(image);
  const bufferT = tf.node.decodeImage(data);
  // resize
  const resizeT = tf.image.resizeBilinear(bufferT, [inputSize, inputSize]);
  tf.dispose(bufferT);
  // add dimension as detection requires 4d tensor
  const expandT = tf.expandDims(resizeT, 0);
  tf.dispose(resizeT);
  // cast if needed
  let imageT;
  if ((dtype === 'float16') || (dtype === 'float32') || (dtype === 'DT_FLOAT')) {
    const casted = tf.cast(expandT, 'float32');
    imageT = tf.mul(casted, [1.0 / divFactor]);
    tf.dispose(casted);
  } else {
    imageT = tf.clone(expandT);
  }
  tf.dispose(expandT);
  // return image tensor
  log.info('Image:', image, bufferT.size, 'bytes with shape:', imageT.shape, 'dtype:', dtype);
  return imageT;
}

async function loadSavedModel(modelPath) {
  // get model signature
  let model;
  log.state('Loading saved model:', modelPath);
  const meta = await tf.node.getMetaGraphsFromSavedModel(modelPath);
  const def = meta[0];
  log.data('Model signature:', def);
  const tags = def['tags'];
  const signature = Object.keys(def.signatureDefs)[0];
  try {
    // @ts-ignore
    const size = Object.values(def.signatureDefs[signature].inputs)[0]['shape'][1].array[0];
    if (size && size > 0) inputSize = size;
  } catch { /**/ }
  log.data('Image size:', inputSize);
  // load model
  try {
    model = await tf.node.loadSavedModel(modelPath, tags, signature);
  } catch (err) {
    log.error('Error loading saved model:', modelPath, err.message, err);
    return null;
  }
  // @ts-ignore
  model.dtype = Object.values(def.signatureDefs[signature].inputs)[0]['dtype'];
  return model;
}

async function loadGraphModel(modelPath) {
  let model;
  log.state('Loading graph model:', modelPath);
  try {
    model = await tf.loadGraphModel(`file://${path.join(modelPath, '/model.json')}`);
  } catch (err) {
    log.error('Error loading graph model:', modelPath, err.message, err);
    return null;
  }
  log.data('Model signature:', model.modelSignature);
  try {
    // @ts-ignore
    const size = Object.values(model.modelSignature['inputs'])[0].tensorShape.dim[1].size;
    if (size && size > 0) inputSize = size;
    const dtype = Object.values(model.modelSignature['inputs'])[0].dtype;
    // @ts-ignore
    model.dtype = dtype || 'float32';
  } catch { /**/ }
  log.data('Image size:', inputSize);
  return model;
}

async function classify(modelPath, image) {
  let model;
  if (fs.existsSync(path.join(modelPath, 'saved_model.pb'))) model = await loadSavedModel(modelPath);
  else if (fs.existsSync(path.join(modelPath, 'model.json'))) model = await loadGraphModel(modelPath);
  else {
    log.error('Cannot determine model type');
    return;
  }
  if (!model) {
    log.error('Model not loaded');
    return;
  }
  // get image tensor
  // @ts-ignore
  const imageT = getTensorFromImage(image, model?.dtype);
  // run prediction
  let resT;
  try {
    resT = await model.predict(imageT || tf.tensor([]));
  } catch (err) {
    log.error('Error executing graph model:', modelPath, err.message);
  }
  // @ts-ignore
  if (resT) {
    log.data('Result', resT);
    // @ts-ignore
    const res = resT.dataSync();
    const scores = [];
    res.forEach((a, i) => scores.push({ index: i + 1, score: a }));
    scores.sort((a, b) => b.score - a.score);
    const top = scores.filter((a) => a.score > minScore).slice(0, 5);
    const classes = JSON.parse(fs.readFileSync(path.join(modelPath, 'classes.json')).toString());
    for (const score of top) log.data(`Score: ${Math.trunc(100 * score.score)}% ID: ${classes[score.index].id} Class: ${classes[score.index].displayName}`);
  } else log.data('Result empty');
  imageT?.dispose();
  tf.dispose(resT);
}

async function main() {
  log.header();
  log.info('TensorFlow/JS Version', tf.version_core);
  await tf.setBackend('tensorflow'); // 'tensorflow' is tfjs-node backend mapped to libtensorflow.so
  await tf.enableProdMode();
  await tf.ENV.set('DEBUG', false);
  await tf.ready();
  log.info('TensorFlow/JS Backend', tf.getBackend());
  log.info('TensorFlow/JS Flags', tf.env().getFlags());
  log.state('Params:', process.argv);
  if (process.argv.length !== 4) {
    log.error('Required params: <modelPath> <image>');
    process.exit(1);
  }
  const modelPath = fs.existsSync(process.argv[2]) && fs.statSync(process.argv[2]).isDirectory() ? process.argv[2] : null;
  if (!modelPath) {
    log.error('<modelPath> is not a valid directory');
    process.exit(1);
  }
  const image = fs.existsSync(process.argv[3]) && fs.statSync(process.argv[3]).isFile() ? process.argv[3] : null;
  if (!image) {
    log.error('<image> is not a valid file');
    process.exit(1);
  }
  await classify(modelPath, image);
}

main();
