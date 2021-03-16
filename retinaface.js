const fs = require('fs');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');

const image = {
  width: 640,
  height: 640,
  save: true,
};

const options = {
  strides: 20, // can be 20/40/80
  minScore: 0.30,
  maxResults: 50,
  iouThreshold: 0.5,
};

const models = [];

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

function getTensorFromImage(imageFile) {
  if (!fs.existsSync(imageFile)) {
    log.error('Not found:', imageFile);
    return null;
  }
  const imageT = tf.tidy(() => {
    const data = fs.readFileSync(imageFile);
    const bufferT = tf.node.decodeImage(data);
    // should pad for correct image aspect ration instead of fixed resize
    const resizeT = bufferT.resizeBilinear([640, 640]);
    const normalizeT = resizeT.div(127.5).sub(1); // normalize range from 0..255 to -1..1
    const transposeT = tf.transpose(normalizeT, [2, 0, 1]);
    const expandedT = tf.expandDims(transposeT, 0);
    const castT = tf.cast(expandedT, 'float32');
    return castT;
  });
  log.info('Image:', imageFile, 'shape:', imageT.shape);
  return imageT;
}

async function saveProcessedImage(inImage, outImage, data) {
  if (!data) return false;
  return new Promise(async (resolve) => {
    const scale = image.saveSize > 0 ? (data.image.width / image.saveSize) : 1;
    const c = new canvas.Canvas(data.image.width / scale, data.image.height / scale);
    const ctx = c.getContext('2d');
    const original = await canvas.loadImage(inImage);
    ctx.drawImage(original, 0, 0, c.width, c.height);
    for (const obj of data.detected) {
      rect({ drawCanvas: c, x: obj.bbox.x / scale, y: obj.bbox.y / scale, width: obj.bbox.width / scale, height: obj.bbox.height / scale, title: `${Math.round(100 * obj.score)}%` });
    }
    const out = fs.createWriteStream(outImage);
    out.on('finish', () => {
      log.state('Created output image:', outImage);
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

async function processPrediction(res) {
  const scoresT = res.find((a) => (a.shape[1] === 4 && a.shape[2] === options.strides)); // scores?
  const boxesT = res.find((a) => (a.shape[1] === 8 && a.shape[2] === options.strides)); // boxes?
  const landmarksT = res.find((a) => (a.shape[1] === 20 && a.shape[2] === options.strides)); // landmarks?
  log.data('Outputs', 'scores:', scoresT.shape, 'boxes:', boxesT.shape, 'landmarks:', landmarksT.shape);

  const pool = tf.conv2dTranspose(boxesT, [1, 8, 20, 4], [1, 640, 640, 4], 20, 'same');
  console.log(pool.shape, pool.arraySync);
  // how to reshape
  // transpose from NCHW to NHWC?
  // run conv2d or not?
  // boxes are top/left/bottom/right or top/let/width/height or centerx/centery/4 x side?
  // at the end coords need to be multipled with strides
  // do nms
}

async function processGraphModel(modelPath, inImage, outImage) {
  if (!models[modelPath]) {
    log.state('Loading graph model:', modelPath);
    models[modelPath] = await tf.loadGraphModel(modelPath);
    models[modelPath].path = modelPath;
  }
  const imageT = getTensorFromImage(inImage);
  const resT = await models[modelPath].executeAsync(imageT);
  const res = resT ? await processPrediction(resT) : [];
  if (imageT) imageT.dispose();
  for (const t of resT) t.dispose();
  // await saveProcessedImage(inImage, outImage, res);
  // log.state(`Exec: model:${modelPath} input:${inImage} output:${outImage} objects:`, res.detected.length);
  return res;
}

async function main() {
  log.header();
  if (process.argv.length !== 4) {
    log.error(`Usage: ${process.argv[1]} <input-image> <output-image>`);
    return;
  }
  await tf.enableProdMode();
  await tf.ENV.set('DEBUG', false);
  await tf.ready();
  const input = process.argv[2];
  const output = process.argv[3];
  await processGraphModel('file://models/retinaafce-mb-025/graph-resize/model.json', input, output);
  for (const model in models) tf.dispose(model);
}

main();
