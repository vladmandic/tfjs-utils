const fs = require('fs');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');

// app options
const debug = true;

// image options
const image = {
  save: true,
  saveSize: 1920, // set to 0 to avoid image resizing
  blurNude: true,
  blurRadius: 50,
};

// model options
const options = {
  minScore: 0.30,
  maxResults: 50,
  iouThreshold: 0.5,
};

const models = [];

const labels = {
  0: { id: 0, displayName: 'exposed anus' },
  1: { id: 1, displayName: 'exposed armpits' },
  2: { id: 2, displayName: 'belly' },
  3: { id: 3, displayName: 'exposed belly' },
  4: { id: 4, displayName: 'buttocks' },
  5: { id: 5, displayName: 'exposed buttocks' },
  6: { id: 6, displayName: 'female' },
  7: { id: 7, displayName: 'male' },
  8: { id: 8, displayName: 'feet' },
  9: { id: 9, displayName: 'exposed feet' },
  10: { id: 10, displayName: 'breast' },
  11: { id: 11, displayName: 'exposed breast' },
  12: { id: 12, displayName: 'vagina' },
  13: { id: 13, displayName: 'exposed vagina' },
  14: { id: 14, displayName: 'male breast' },
  15: { id: 15, displayName: 'exposed male breast' },
};

const labelPerson = [6, 7];
const labelSexy = [1, 2, 3, 4, 8, 9, 10, 15];
const labelNude = [0, 5, 11, 12, 13];

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

function blur({ drawCanvas = null, left = 0, top = 0, width = 0, height = 0 }) {
  const blurCanvas = new canvas.Canvas(width / image.blurRadius, height / image.blurRadius);
  const blurCtx = blurCanvas.getContext('2d');
  blurCtx.imageSmoothingEnabled = true;
  // blurCtx.drawImage(drawCanvas, 0, 0, width / image.blurRadius, height / image.blurRadius);
  // drawImage(image: Canvas | Image, sx: number, sy: number, sw: number, sh: number, dx: number, dy: number, dw: number, dh: number): void
  blurCtx.drawImage(drawCanvas, left, top, width, height, 0, 0, width / image.blurRadius, height / image.blurRadius);
  const canvasCtx = drawCanvas.getContext('2d');
  canvasCtx.drawImage(blurCanvas, left, top, width, height);
}

function getTensorFromImage(imageFile) {
  if (!fs.existsSync(imageFile)) {
    log.error('Not found:', imageFile);
    return null;
  }
  const data = fs.readFileSync(imageFile);
  const bufferT = tf.node.decodeImage(data);
  const expandedT = tf.expandDims(bufferT, 0);
  const imageT = tf.cast(expandedT, 'float32');
  imageT.file = imageFile;
  tf.dispose(expandedT);
  tf.dispose(bufferT);
  if (debug) log.info('Image:', imageT.file, 'width:', imageT.shape[2], 'height:', imageT.shape[1]);
  return imageT;
}

async function saveProcessedImage(inImage, outImage, data) {
  if (!data) return false;
  return new Promise(async (resolve) => {
    // create canvas
    const scale = image.saveSize > 0 ? (data.image.width / image.saveSize) : 1;
    const c = new canvas.Canvas(data.image.width / scale, data.image.height / scale);
    const ctx = c.getContext('2d');
    // load and draw original image
    const original = await canvas.loadImage(inImage);
    ctx.drawImage(original, 0, 0, c.width, c.height);
    // draw all detected objects
    for (const obj of data.detected) {
      if (labelNude.includes(obj.classId) && image.blurNude) blur({ drawCanvas: c, left: obj.bbox.x / scale, top: obj.bbox.y / scale, width: obj.bbox.width / scale, height: obj.bbox.height / scale });
      rect({ drawCanvas: c, x: obj.bbox.x / scale, y: obj.bbox.y / scale, width: obj.bbox.width / scale, height: obj.bbox.height / scale, title: `${Math.round(100 * obj.score)}% ${obj.class} #${obj.classId}` });
    }
    // write canvas to jpeg
    const out = fs.createWriteStream(outImage);
    out.on('finish', () => {
      if (debug) log.state('Created output image:', outImage);
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

async function processPrediction(res, imageT) {
  // get results according to map
  const classesT = res.find((a) => a.dtype === 'int32');
  const scoresT = res.find((a) => a.shape.length === 2);
  const boxesT = res.find((a) => a.shape.length === 3);
  const classes = await classesT.data();
  const scores = await scoresT.data();
  const boxes = await boxesT.array();
  // sort & filter results
  const overlapT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, options.maxResults, options.iouThreshold, options.minScore);
  const overlap = await overlapT.data();
  tf.dispose(overlapT);
  const detected = [];
  // create result object
  for (const i in overlap) {
    const id = parseInt(i);
    detected.push({
      score: Math.trunc(10000 * scores[i]) / 10000,
      classId: classes[id],
      class: labels[classes[id]]?.displayName,
      bbox: {
        x: Math.trunc(boxes[0][id][0]),
        y: Math.trunc(boxes[0][id][1]),
        width: Math.trunc((boxes[0][id][3] - boxes[0][id][1])),
        height: Math.trunc((boxes[0][id][2] - boxes[0][id][0])),
      },
    });
  }
  const obj = { detected };
  obj.image = { file: imageT.file, width: imageT.shape[2], height: imageT.shape[1] };
  obj.person = detected.filter((a) => labelPerson.includes(a.classId));
  obj.sexy = detected.filter((a) => labelSexy.includes(a.classId));
  obj.nude = detected.filter((a) => labelNude.includes(a.classId));
  if (debug) log.data(obj);
  return obj;
}

async function processSavedModel(modelPath, inImage, outImage) {
  if (!models[modelPath]) {
    if (debug) log.state('Loading saved model:', modelPath);
    // const meta = await tf.node.getMetaGraphsFromSavedModel(modelPath);
    try {
      models[modelPath] = await tf.node.loadSavedModel(modelPath, ['serve'], 'predict');
      models[modelPath].path = modelPath;
    } catch (err) {
      log.error('Error loading graph model:', modelPath, err.message);
      return null;
    }
  }
  // get image tensor
  const imageT = getTensorFromImage(inImage);
  // run prediction
  let resT;
  try {
    resT = models[modelPath].predict ? await models[modelPath].predict(imageT) : null;
  } catch (err) {
    log.error('Error executing graph model:', modelPath, err.message);
  }
  // parse outputs
  const res = resT ? await processPrediction(resT, imageT, models[modelPath]) : [];
  // free up memory
  imageT.dispose();
  for (const tensorT of resT) tensorT.dispose();
  // save processed image and return result
  await saveProcessedImage(inImage, outImage, res);
  log.state(`Exec: model:${modelPath} input:${inImage} output:${outImage} objects:`, res.detected.length);
  return res;
}

async function processGraphModel(modelPath, inImage, outImage) {
  if (!models[modelPath]) {
    if (debug) log.state('Loading graph model:', modelPath);
    // load model
    try {
      models[modelPath] = await tf.loadGraphModel(modelPath);
      models[modelPath].path = modelPath;
    } catch (err) {
      log.error('Error loading graph model:', modelPath, err.message, err);
      return null;
    }
  }
  // get image tensor
  const imageT = getTensorFromImage(inImage);
  // run prediction
  let resT;
  try {
    resT = models[modelPath].executeAsync ? await models[modelPath].executeAsync(imageT) : null;
  } catch (err) {
    log.error('Error executing graph model:', modelPath, err.message);
  }
  // parse outputs
  const res = resT ? await processPrediction(resT, imageT, models[modelPath]) : [];
  // free up memory
  imageT.dispose();
  for (const tensorT of resT) tensorT.dispose();
  // save processed image and return result
  await saveProcessedImage(inImage, outImage, res);
  log.state(`Exec: model:${modelPath} input:${inImage} output:${outImage} objects:`, res.detected.length);
  return res;
}

async function main() {
  log.header();

  await tf.enableProdMode();
  await tf.ENV.set('DEBUG', false);
  // await tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);
  await tf.ready();

  if (debug) log.info('TensorFlow/JS Version', tf.version_core);
  if (debug) log.info('TensorFlow/JS Backend', tf.getBackend());
  if (debug) log.info('TensorFlow/JS Flags', tf.ENV.flags);

  await processSavedModel('models/saved/nudenet', 'inputs/nude1.jpg', 'outputs/nude1-saved.jpg');
  await processGraphModel('file://models/graph/nudenet/model.json', 'inputs/nude2.jpg', 'outputs/nude2-graph.jpg');

  for (const model in models) tf.dispose(model);
}

main();
