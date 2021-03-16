const fs = require('fs');
const path = require('path');
const process = require('process');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');

const modelOptions = {
  modelPath: 'file://models/nanodet/nanodet.json',
  minScore: 0.15, // low confidence, but still remove irrelevant
  iouThreshold: 0.1, // be very aggressive with removing overlapped boxes
  maxResults: 10, // high number of results, but likely never reached
  scaleBox: 2.5, // increase box size
};

// eslint-disable-next-line max-len
const labelsCoco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'vehicle', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'bear', 'animal', 'animal', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'pastry', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

async function saveImage(img, res) {
  // create canvas
  const c = new canvas.Canvas(img.inputShape[0], img.inputShape[1]);
  const ctx = c.getContext('2d');

  // load and draw original image
  const original = await canvas.loadImage(img.fileName);
  ctx.drawImage(original, 0, 0, c.width, c.height);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'white';
  ctx.fillStyle = 'white';
  ctx.font = 'small-caps 28px "Segoe UI"';

  // draw all detected objects
  for (const obj of res) {
    ctx.fillText(`${Math.round(100 * obj.score)}% [${obj.strideSize}] ${obj.label}`, obj.center[0] + 1, obj.center[1] + 1);
    ctx.rect(obj.box[0], obj.box[1], obj.box[2] - obj.box[0], obj.box[3] - obj.box[1]);
  }
  ctx.stroke();

  // write canvas to jpeg
  const outImage = path.basename(img.fileName, '.jpg') + '-nanodet.jpg';
  const out = fs.createWriteStream(outImage);
  out.on('finish', () => log.state('Created output image:', outImage));
  out.on('error', (err) => log.error('Error creating image:', outImage, err));
  const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
  stream.pipe(out);
}

// load image from file and prepares image tensor that fits the model
function loadImage(fileName, inputSize) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data);
    const resize = tf.image.resizeBilinear(buffer, [inputSize, inputSize]);
    const cast = resize.cast('float32');
    const normalize = cast.div(255);
    const expand = normalize.expandDims(0);
    const transpose = expand.transpose([0, 3, 1, 2]);
    const tensor = transpose;
    const img = { fileName, tensor, inputShape: [buffer.shape[1], buffer.shape[0]], outputShape: tensor.shape, size: buffer.size };
    return img;
  });
  return obj;
}

async function processResults(res, inputSize, outputShape) {
  let results = [];
  for (const strideSize of [1, 2, 4]) { // try each stride size as it detects large/medium/small objects
    // find scores, boxes, classes
    tf.tidy(() => { // wrap in tidy to automatically deallocate temp tensors
      const baseSize = strideSize * 13; // 13x13=169, 26x26=676, 52x52=2704
      // find boxes and scores output depending on stride
      // log.info('Variation:', strideSize, 'strides', baseSize, 'baseSize');
      const scores = res.find((a) => (a.shape[1] === (baseSize ** 2) && a.shape[2] === 80))?.squeeze();
      const features = res.find((a) => (a.shape[1] === (baseSize ** 2) && a.shape[2] === 32))?.squeeze();
      // log.state('Found features tensor:', features?.shape);
      // log.state('Found scores tensor:', scores?.shape);
      const scoreIdx = scores.argMax(1).dataSync();
      const scoresMax = scores.max(1).dataSync();
      const boxesMax = features.reshape([-1, 4, 8]);
      const boxIdx = boxesMax.argMax(2).arraySync();
      for (let i = 0; i < scores.shape[0]; i++) {
        if (scoreIdx[i] !== 0 && scoresMax[i] > modelOptions.minScore) {
          const cx = (0.5 + Math.trunc(i % baseSize)) / baseSize;
          const cy = (0.5 + Math.trunc(i / baseSize)) / baseSize;
          const boxOffset = boxIdx[i].map((a) => a * (baseSize / strideSize / inputSize));
          const boxRaw = [
            cx - (modelOptions.scaleBox / strideSize * boxOffset[0]),
            cy - (modelOptions.scaleBox / strideSize * boxOffset[1]),
            cx + (modelOptions.scaleBox / strideSize * boxOffset[2]),
            cy + (modelOptions.scaleBox / strideSize * boxOffset[3]),
          ];
          const box = [
            boxRaw[0] * outputShape[0],
            boxRaw[1] * outputShape[1],
            boxRaw[2] * outputShape[0],
            boxRaw[3] * outputShape[1],
          ];
          const result = {
            score: scoresMax[i],
            strideSize,
            class: scoreIdx[i] + 1,
            label: labelsCoco[scoreIdx[i]],
            center: [Math.trunc(outputShape[0] * cx), Math.trunc(outputShape[1] * cy)],
            centerRaw: [cx, cy],
            box: box.map((a) => Math.trunc(a)),
            boxRaw,
          };
          results.push(result);
        }
      }
    });
  }

  // deallocate tensors
  res.forEach((t) => tf.dispose(t));

  // normally nms is run on raw results, but since boxes need to be normalized this way we skip processing of unnecessary boxes
  const nmsBoxes = results.map((a) => a.boxRaw);
  const nmsScores = results.map((a) => a.score);
  const nms = await tf.image.nonMaxSuppressionAsync(nmsBoxes, nmsScores, modelOptions.maxResults, modelOptions.iouThreshold, modelOptions.minScore);
  const nmsIdx = nms.dataSync();
  tf.dispose(nms);

  // filter & sort results
  results = results
    .filter((a, idx) => nmsIdx.includes(idx))
    .sort((a, b) => b.score - a.score);

  return results;
}

async function main() {
  log.header();

  // init tensorflow
  await tf.enableProdMode();
  await tf.setBackend('tensorflow');
  await tf.ENV.set('DEBUG', false);
  await tf.ready();

  // load model
  const model = await tf.loadGraphModel(modelOptions.modelPath);
  // log.info('Model signature:', model.modelSignature);
  log.info('Loaded model', modelOptions, 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);

  // load image and get approprite tensor for it
  const inputSize = Object.values(model.modelSignature['inputs'])[0].tensorShape.dim[2].size;
  const imageFile = process.argv.length > 2 ? process.argv[2] : null;
  if (!imageFile || !fs.existsSync(imageFile)) {
    log.error('Specify a valid image file');
    process.exit();
  }
  const img = loadImage(imageFile, inputSize);
  log.info('Loaded image:', img.fileName, 'inputShape:', img.inputShape, 'outputShape:', img.outputShape);

  // run actual prediction
  const res = model.predict(img.tensor);

  // process results
  const results = await processResults(res, inputSize, img.inputShape);

  // print results
  log.data('Results:', results);

  // save processed image
  await saveImage(img, results);
}

main();
