const fs = require('fs');
const path = require('path');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');

// app options
const debug = true;
const imgInput = 'inputs/';
const imgOutput = 'outputs/';
const outSize = 1920; // set to 0 to avoid image resizing
const scaleOutput = true;
const divFactor = 255.0;

// model options
const minScore = 0.35;
const maxResults = 50;
const iouThreshold = 0.1;

const coco = JSON.parse(fs.readFileSync('./coco.json'));
const openimages = JSON.parse(fs.readFileSync('./openimages.json'));
const models = [];
let performances = {};

function perf(model, time) {
  const t = Math.trunc(parseInt(time) / 1000 / 1000);
  if (!performances[model]) performances[model] = { total: 0, count: 0, avg: 0, max: 0 };
  performances[model].total += t;
  performances[model].count += 1;
  performances[model].avg = Math.trunc(1.0 * performances[model].total / performances[model].count);
  performances[model].max = performances[model].max > t ? performances[model].max : t;
}

function rect({ drawCanvas = null, x = 0, y = 0, width = 0, height = 0, radius = 8, lineWidth = 2, color = 'white', title = null, font = 'small-caps 28px "Segoe UI"' }) {
  const ctx = drawCanvas.getContext('2d');
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.lineWidth = 2;
  ctx.fillStyle = color;
  ctx.font = font;
  if (title) ctx.fillText(title, x + 4, y + 24);
}

function getTensorFromImage(image, dtype) {
  if (!fs.existsSync(image)) {
    log.error('Not found:', image);
    return null;
  }
  // load & decode jpeg
  const data = fs.readFileSync(image);
  const bufferT = tf.node.decodeImage(data);
  // add dimension as detection requires 4d tensor
  const expanded = tf.expandDims(bufferT, 0);
  // cast if needed
  let imageT;
  if ((dtype === 'float16') || (dtype === 'float32') || (dtype === 'DT_FLOAT')) {
    const casted = tf.cast(expanded, 'float32');
    imageT = tf.mul(casted, [1.0 / divFactor]);
    tf.dispose(casted);
  } else {
    imageT = tf.clone(expanded);
  }
  // return image tensor
  imageT.file = image;
  if (debug) log.info('Image:', imageT.file, bufferT.size, 'bytes with shape:', imageT.shape, 'dtype:', dtype);
  tf.dispose(expanded);
  tf.dispose(bufferT);
  return imageT;
}

async function saveProcessedImage(inImage, outImage, data) {
  if (!data || !data[0]) return false;
  return new Promise(async (resolve) => {
    // create canvas
    const scale = outSize > 0 ? (data[0].image.width / outSize) : 1;
    const c = new canvas.Canvas(data[0].image.width / scale, data[0].image.height / scale);
    const ctx = c.getContext('2d');
    // load and draw original image
    const image = await canvas.loadImage(inImage);
    ctx.drawImage(image, 0, 0, c.width, c.height);
    // draw image title
    rect({ drawCanvas: c, x: 0, y: 0, width: 0, height: 0, title: data[0].model });
    // draw all detected objects
    for (const obj of data) {
      rect({ drawCanvas: c, x: obj.rect.x / scale, y: obj.rect.y / scale, width: obj.rect.width / scale, height: obj.rect.height / scale, title: `${Math.round(100 * obj.score)}% ${obj.class} #${obj.classId}` });
    }
    // write canvas to jpeg
    const out = fs.createWriteStream(outImage);
    out.on('finish', () => {
      if (debug) log.state('Created image:', outImage);
      resolve(true);
    });
    out.on('error', (err) => {
      log.error('Error creating image:', outImage, err);
      resolve(true);
    });
    const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
    stream.pipe(out);
  });
}

async function processPrediction(res, image, model) {
  // get results according to map
  const classes = await res[model.map['detection_classes']].data();
  const scores = await res[model.map['detection_scores']].data();
  const boxes = await res[model.map['detection_boxes']].array();
  console.log(classes, scores, boxes);
  const numClasses = Math.max(...classes);
  const labels = numClasses <= 100 ? coco : openimages;
  // sort & filter results
  const overlapT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, maxResults, iouThreshold, minScore);
  const overlap = await overlapT.data();
  tf.dispose(overlapT);
  const results = [];
  // create result object
  for (const i in overlap) {
    const id = parseInt(i);
    results.push({
      image: { file: image.file, width: image.shape[2], height: image.shape[1] },
      score: Math.trunc(10000 * scores[i]) / 10000,
      classId: classes[id],
      class: labels[classes[id]]?.displayName || labels[id]?.displayName || `UNKNOWN:${id}`,
      bbox: boxes[0][id].map((a) => Math.trunc(10000 * a) / 10000),
      model: model.name,
      rect: {
        x: Math.trunc(boxes[0][id][1] * (scaleOutput ? image.shape[2] : 1)),
        y: Math.trunc(boxes[0][id][0] * (scaleOutput ? image.shape[1] : 1)),
        width: Math.trunc((boxes[0][id][3] - boxes[0][id][1]) * (scaleOutput ? image.shape[2] : 1)),
        height: Math.trunc((boxes[0][id][2] - boxes[0][id][0]) * (scaleOutput ? image.shape[1] : 1)),
      },
    });
  }
  return results;
}

async function processSavedModel(image, modelPath) {
  if (!models[modelPath]) {
    log.state('Loading saved model:', modelPath);
    // get model signature
    const meta = await tf.node.getMetaGraphsFromSavedModel(modelPath);
    const def = meta[0];
    if (debug) log.data('Model signature:', def);
    const tags = def['tags'];
    const signature = Object.keys(def.signatureDefs)[0];
    const outputs = def.signatureDefs[signature].outputs;
    // load model
    try {
      models[modelPath] = await tf.node.loadSavedModel(modelPath, tags, signature);
    } catch (err) {
      log.error('Error loading graph model:', modelPath, err.message, err);
      models[modelPath] = modelPath;
      return;
    }
    models[modelPath].dtype = Object.values(def.signatureDefs[signature].inputs)[0]['dtype'];
    models[modelPath].name = modelPath;
    // build model map
    models[modelPath].map = {};
    for (const i in Object.keys(outputs)) models[modelPath].map[Object.keys(outputs)[i]] = i;
    if (debug) log.data('Model output map:', models[modelPath].map);
  }
  // get image tensor
  const imageT = getTensorFromImage(image, models[modelPath].dtype);
  // run prediction
  const t0 = process.hrtime.bigint();
  let resT;
  try {
    // resT = models[modelPath].predict ? await models[modelPath].predict(imageT, { score: minScore, iou: iouThreshold, topk: maxResults }) : null;
    resT = await models[modelPath].executeAsync(imageT);
  } catch (err) {
    log.error('Error executing graph model:', modelPath, err.message);
  }
  const t1 = process.hrtime.bigint();
  // parse outputs
  const parsed = resT ? await processPrediction(resT, imageT, models[modelPath]) : [];
  if (parsed.length > 0) perf(modelPath, t1 - t0);
  // free up memory
  tf.dispose(imageT);
  tf.dispose(resT);
  // save processed image
  if (parsed.length > 0) await saveProcessedImage(image, `${imgOutput}${path.basename(image)}-${path.basename(modelPath)}.jpg`, parsed);
  // results
  if (parsed.length > 0) log.data('Exec:', image, modelPath, Math.trunc(parseInt(t1 - t0) / 1000 / 1000), 'ms', 'detected', parsed.length, 'objects');
}

async function processGraphModel(image, modelPath) {
  if (!models[modelPath]) {
    log.state('Loading graph model:', modelPath);
    // load model
    try {
      models[modelPath] = await tf.loadGraphModel(`file://${path.join(modelPath, '/model.json')}`);
    } catch (err) {
      log.error('Error loading graph model:', modelPath, err.message, err);
      models[modelPath] = modelPath;
      return;
    }
    // static model map since graph model looses signature info
    const sig = models[modelPath].executor._signature;
    models[modelPath].map = {};
    models[modelPath].map['detection_classes'] = Object.keys(sig['outputs']).findIndex((a) => a === 'detection_classes:0');
    models[modelPath].map['detection_scores'] = Object.keys(sig['outputs']).findIndex((a) => a === 'detection_scores:0');
    models[modelPath].map['detection_boxes'] = Object.keys(sig['outputs']).findIndex((a) => a === 'detection_boxes:0');
    // models[modelPath].map['detection_classes'] = Object.keys(sig['outputs']).findIndex((a) => a === 'Identity_2:0');
    // models[modelPath].map['detection_scores'] = Object.keys(sig['outputs']).findIndex((a) => a === 'Identity_4:0');
    // models[modelPath].map['detection_boxes'] = Object.keys(sig['outputs']).findIndex((a) => a === 'Identity_1:0');
    models[modelPath].dtype = Object.values(sig['inputs'])[0]['dtype'];
    models[modelPath].name = modelPath;
    if (debug) log.data('Model signature:', sig);
    if (debug) log.data('Model output map:', models[modelPath].map);
  }
  // get image tensor
  const imageT = getTensorFromImage(image, models[modelPath].dtype);
  // run prediction
  const t0 = process.hrtime.bigint();
  let resT;
  try {
    resT = models[modelPath].executeAsync ? await models[modelPath].executeAsync(imageT) : null;
  } catch (err) {
    log.error('Error executing graph model:', modelPath, err.message);
  }
  const t1 = process.hrtime.bigint();
  // parse outputs
  const parsed = resT ? await processPrediction(resT, imageT, models[modelPath], coco) : [];
  if (parsed.length > 0) perf(modelPath, t1 - t0);
  // free up memory
  tf.dispose(imageT);
  tf.dispose(resT);
  // save processed image
  if (parsed.length > 0) await saveProcessedImage(image, `${imgOutput}${path.basename(image)}-${path.basename(modelPath)}.jpg`, parsed);
  // results
  log.data('Exec:', image, modelPath, Math.trunc(parseInt(t1 - t0) / 1000 / 1000), 'ms', 'detected', parsed.length || 0, 'objects');
}

// eslint-disable-next-line no-unused-vars
async function testAll({ graph = true, saved = true, single = false }) {
  // enumerate all images in input and run prediction loop for each model
  const images = fs.readdirSync(imgInput);
  log.info('Images: ', images.length);
  // enumerate all local models
  const modelsSaved = fs.readdirSync('models/saved/');
  const modelsGraph = fs.readdirSync('models/graph/');
  log.info('Models: Saved:', modelsSaved.length, 'Graph:', modelsGraph.length);
  // loop through all images and saved models
  if (saved) {
    performances = {};
    if (single) {
      for (const model of modelsSaved) await processSavedModel('inputs/cars.jpg', `models/saved/${model}`);
    } else {
      for (const image of images) {
        for (const model of modelsSaved) await processSavedModel(`inputs/${image}`, `models/saved/${model}`);
      }
    }
    // print performance info and dispose models
    for (const [model, data] of Object.entries(performances)) log.info(model, data);
    for (const model in models) tf.dispose(model);
  }
  if (graph) {
    // loop through all images and graph models
    performances = {};
    if (single) {
      for (const model of modelsSaved) await processGraphModel('inputs/cars.jpg', `models/graph/${model}`);
    } else {
      for (const image of images) {
        for (const model of modelsGraph) await processGraphModel(`inputs/${image}`, `models/graph/${model}`);
      }
    }
    // print performance info and dispose models
    for (const [model, data] of Object.entries(performances)) log.info(model, data);
    for (const model in models) tf.dispose(model);
  }
}

// eslint-disable-next-line no-unused-vars
async function testSingle() {
  performances = {};

  // await processSavedModel('inputs/bar.jpg', 'models/saved/faster-rcnn-inception-resnet-v2-atrous-v4-oi');
  await processGraphModel('inputs/bar.jpg', 'models/graph/openimages-faster-rcnn-inception-resnet-v2-atrous-v4');

  for (const [model, data] of Object.entries(performances)) log.info(model, data);
  for (const model in models) tf.dispose(model);
}

async function main() {
  log.header();
  log.info('TensorFlow/JS Version', tf.version_core);
  await tf.setBackend('tensorflow'); // 'tensorflow' is tfjs-node backend mapped to libtensorflow.so
  await tf.enableProdMode();
  await tf.ENV.set('DEBUG', false);
  tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);
  await tf.ready();
  log.info('TensorFlow/JS Backend', tf.getBackend());
  log.info('TensorFlow/JS Flags', tf.ENV.flags);

  // await testAll({ graph: false, saved: true, single: true });
  await testSingle();
}

main();
