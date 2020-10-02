const fs = require('fs');
const path = require('path');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');

const debug = false;
const minScore = 0.2;
const maxResults = 50;
const iouThreshold = 0.1;
const outSize = 1920; // set to 0 to avoid image resizing
const imgInput = 'inputs/';
const imgOutput = 'outputs/'

const coco = JSON.parse(fs.readFileSync('./coco.json'));
// const openimages = JSON.parse(fs.readFileSync('./openimages.json'));

const defaultFont = 'small-caps 12px "Segoe UI"';
const models = [];

function time(t1, t0) {
  return Math.trunc(parseInt(t1 - t0) / 1000 / 1000);
}

function rect({ canvas = null, x = 0, y = 0, width = 0, height = 0, radius = 8, lineWidth = 2, color = 'white', title = null, font = null }) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
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
  ctx.lineWidth = 1;
  ctx.fillStyle = color;
  ctx.font = font || defaultFont;
  if (title) ctx.fillText(title, x + 4, y + 12);
}

function getTensorFromImage(path, dtype) {
  if (!fs.existsSync(path)) {
    log.error('Not found:', path);
    return null;
  }
  // load binary data
  const data = fs.readFileSync(path);
  // decode jpeg
  const tfimage = tf.node.decodeImage(data);
  // add dimension as detection requires 4d tensor
  const expanded = tf.expandDims(tfimage, 0);

  let imageT;
  if ((dtype === 'float16') || (dtype === 'float32') || (dtype === 'DT_FLOAT')) {
    const casted = tf.cast(expanded, 'float32');
    imageT = tf.mul(casted, [1.0 / 255.0]);
    tf.dispose(casted);
  } else {
    imageT = tf.clone(expanded);
  }
  tf.dispose(expanded);

  imageT.file = path;
  imageT.bytes = tfimage.size;
  log.info('Image:', imageT.file, imageT.bytes, 'bytes with shape:', imageT.shape, 'dtype:', dtype);
  tf.dispose(tfimage);
  return imageT;
}

async function saveProcessedImage(inImage, outImage, data) {
  if (!data || !data[0]) return;
  return new Promise(async (resolve) => {
    // create canvas
    const scale = outSize > 0 ? (data[0].image.width / outSize) : 1;
    const c = new canvas.Canvas(data[0].image.width / scale, data[0].image.height / scale);
    const ctx = c.getContext('2d');
    // load and draw original image
    const image = await canvas.loadImage(inImage);
    ctx.drawImage(image, 0, 0, c.width, c.height);
    // draw image title
    rect({ canvas: c, x: 0, y: 0, width: 0, height: 0, title: data[0].model });
    // draw all detected objects
    for (const obj of data) {
      rect({ canvas: c, x: obj.rect.x / scale, y: obj.rect.y / scale, width: obj.rect.width / scale, height: obj.rect.height / scale, title: `${Math.round(100 * obj.score)}% ${obj.class}` });
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

async function processPrediction(res, image, model, labels) {
  // get results according to map
  const classes = await res[model.map['detection_classes']].data();
  const scores = await res[model.map['detection_scores']].data();
  const boxes = await res[model.map['detection_boxes']].array();
  // sort & filter results
  const overlapT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, maxResults, iouThreshold, minScore)
  const overlap = await overlapT.data();
  tf.dispose(overlapT);
  const results = [];
  // create result object
  for (const i in overlap) {
    const id = parseInt(i);
    results.push({
      image: { file: image.file, bytes: image.bytes, width: image.shape[2], height: image.shape[1] },
      score: Math.trunc(10000 * scores[i]) / 10000,
      classId: classes[id],
      class: labels[classes[id]]?.displayName || labels[id]?.displayName || `UNKNOWN:${id}`,
      bbox: boxes[0][id].map((a) => Math.trunc(10000 * a) / 10000),
      model: model.name,
      rect: {
        x: Math.trunc(boxes[0][id][1] * image.shape[2]),
        y: Math.trunc(boxes[0][id][0] * image.shape[1]),
        width: Math.trunc((boxes[0][id][3] - boxes[0][id][1]) * image.shape[2]),
        height: Math.trunc((boxes[0][id][2] - boxes[0][id][0]) * image.shape[1]),
      }
    })
  }
  return results;
}

async function processSavedModel(image, modelPath) {
  log.info('Analyzing saved model:', modelPath);
  const t0 = process.hrtime.bigint();

  let model;
  if (!models[modelPath]) {
    // build model map
    const def = (await tf.node.getMetaGraphsFromSavedModel(modelPath))[0];
    if (debug) log.data('Model signature:', def)
    const tags = def['tags'];
    const signature = Object.keys(def.signatureDefs)[0]
    let outputs = def.signatureDefs[signature].outputs;
    const map = {};
    for (let i in Object.keys(outputs)) map[Object.keys(outputs)[i]] = i;
    if (debug) log.data('Model output map:', map);

    // load model
    log.info('Loading saved model:', modelPath);
    model = await tf.node.loadSavedModel(modelPath, tags, signature);
    model.dtype = Object.values(def.signatureDefs[signature].inputs)[0]['dtype']
    model.name = modelPath;
    model.map = map;
    models[modelPath] = model;
  } else {
    model = models[modelPath];
  }
  const t2 = process.hrtime.bigint();

  // get image tensor
  const imageT = getTensorFromImage(image, model.dtype);
  if (!imageT) return;

  // run prediction
  const resT = await model.predict(imageT, {score: minScore, iou: iouThreshold, topk: maxResults });
  const t3 = process.hrtime.bigint();

  // parse outputs
  const parsed = await processPrediction(resT, imageT, model, coco);
  const t4 = process.hrtime.bigint();

  // free up memory
  tf.dispose(imageT);
  tf.dispose(resT);

  // results
  log.data('Timings:', modelPath, 'load:', time(t2, t0), 'predict:', time(t3, t2), 'process:', time(t4, t3));
  return parsed;
}

async function processGraphModel(image, modelPath) {
  const t0 = process.hrtime.bigint();
  // load model
  let model;
  if (!models[modelPath]) {
    log.info('Loading graph model:', modelPath);
    // static map since graph model looses signature info
    model = await tf.loadGraphModel(`file://${path.join(modelPath, '/model.json')}`);
    const map = {};
    const sig = model.executor._signature;
    map['detection_classes'] = Object.keys(sig['outputs']).findIndex((a) => a === 'Identity_2:0');
    map['detection_scores'] = Object.keys(sig['outputs']).findIndex((a) => a === 'Identity_4:0');
    map['detection_boxes'] = Object.keys(sig['outputs']).findIndex((a) => a === 'Identity_1:0');
    model.dtype = Object.values(sig['inputs'])[0]['dtype']
    model.name = modelPath;
    model.map = map;
    if (debug) log.data('Model signature:', sig);
    if (debug) log.data('Model output map:', map);
    models[modelPath] = model;
  } else {
    model = models[modelPath];
  }
  const t2 = process.hrtime.bigint();

  // get image tensor
  const imageT = getTensorFromImage(image, model.dtype);
  if (!imageT) return;

  // run prediction
  const resT = await model.executeAsync(imageT);
  const t3 = process.hrtime.bigint();

  // parse outputs
  const parsed = await processPrediction(resT, imageT, model, coco);
  const t4 = process.hrtime.bigint();

  // free up memory
  tf.dispose(imageT);
  tf.dispose(resT);

  // results
  log.data('Timings:', modelPath, 'load:', time(t2, t0), 'predict:', time(t3, t2), 'process:', time(t4, t3));
  return parsed;
}

/*
async function processLayersModel(image, modelPath) {
  // needs rewrite
  log.info('Loading layers model:', modelPath);
  const model = await tf.loadLayersModel(`file://${path.join(__dirname, modelPath)}`);
  log.data('Model signature:', model.executor._signature);
  log.state('TensorFlow/JS Memory', tf.memory());
  const res = await model.executeAsync(image);
  log.data('Detected', res);
}
*/

async function testSavedModel(image, model) {
  const data = await processSavedModel(image, model);
  // log.data('Results', model, data.map((a) => [a.score, a.class]));
  if (debug) log.data('Results:', model, 'detected', data.length, 'objects');
  await saveProcessedImage(image, `${imgOutput}${path.basename(image)}-${path.basename(model)}.jpg`, data);
}

async function testGraphModel(image, model) {
  const data = await processGraphModel(image, model);
  // log.data('Results', model, data.map((a) => [a.score, a.class]));
  if (!data) return;
  log.data('Results:', model, 'detected', data.length, 'objects');
  await saveProcessedImage(image, `${imgOutput}${path.basename(image)}-${path.basename(model)}.jpg`, data);
}

async function main() {
  log.header();
  log.info('TensorFlow/JS Version', tf.version_core);
  await tf.setBackend('tensorflow'); // 'tensorflow' is tfjs-node backend mapped to libtensorflow.so
  await tf.enableProdMode();
  tf.ENV.set('DEBUG', false);
  tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);
  await tf.ready();
  log.info('TensorFlow/JS Backend', tf.getBackend());
  log.info('TensorFlow/JS Flags', tf.ENV.flags);

  const files = fs.readdirSync(imgInput);
  log.info('Analyzing', files.length, 'images');
  for (const file of files) {
    const image = `inputs/${file}`
    await testSavedModel(image, 'models/ssd-mobilenet-v2-saved');
    await testGraphModel(image, 'models/ssd-mobilenet-v2-graph');
    await testGraphModel(image, 'models/ssd-mobilenet-v2-graph-f16');
    await testGraphModel(image, 'models/ssd-mobilenet-v2-graph-uint16');
    await testGraphModel(image, 'models/ssd-mobilenet-v2-graph-uint8');

    await testSavedModel(image, 'models/centernet-resnet-50-v2-saved');
    // await testGraphModel(image, 'models/centernet-resnet-50-v2-graph'); // TypeError: Cannot read property 'children' of undefined at /home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3465:33
    
    await testSavedModel(image, 'models/centernet-resnet-50-v1-fpn-saved');
    // await testGraphModel(image, 'models/centernet-resnet-50-v1-fpn-graph'); // TypeError: Cannot read property 'children' of undefined at /home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3465:33

    await testSavedModel(image, 'models/centernet-resnet-101-v1-fpn-saved');
    // await testGraphModel(image, 'models/centernet-resnet-101-v1-fpn-graph'); // TypeError: Cannot read property 'children' of undefined at /home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3465:33

    await testSavedModel(image, 'models/efficientdet-d0-saved');
    // await testGraphModel(image, 'models/efficientdet-d0-graph'); // TypeError: Cannot read property 'children' of undefined at /home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3465:33

    await testSavedModel(image, 'models/efficientdet-d5-saved');
    // await testGraphModel(image, 'models/efficientdet-d5-graph'); // TypeError: Cannot read property 'children' of undefined at /home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3465:33

    await testSavedModel(image, 'models/faster-rcnn-resnet-50-v1-saved'); 
    // await testGraphModel(image, 'models/faster-rcnn-resnet-50-v1-graph'); // ValueError: Unsupported Ops in the model before optimization: BroadcastArgs

    await testSavedModel(image, 'models/retinanet-resnet-50-v1-saved');
    await testGraphModel(image, 'models/retinanet-resnet-50-v1-graph');
    await testGraphModel(image, 'models/retinanet-resnet-50-v1-graph-f16');
    await testGraphModel(image, 'models/retinanet-resnet-50-v1-graph-uint16');
    await testGraphModel(image, 'models/retinanet-resnet-50-v1-graph-uint8');

    // await testSavedModel(image, 'models/ssd-mobilenet-v2-oi'); // session fail to run with error: Table not initialized.
    // await testGraphModel(image, 'models/ssd-mobilenet-v2-oi-graph'); // no tags in saved model

    // await testSavedModel(image, 'models/faster-rcnn-inception-resnet-v2-oi'); // session fail to run with error: Table not initialized.
    // await testGraphModel(image, 'models/faster-rcnn-inception-resnet-v2-oi-graph'); // no tags in saved model
  }
  for (const model in models) tf.dispose(model);
  log.info('Done');
}

main();

// tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --strip_debug_ops=* --control_flow_v2=* --skip_op_check <src> <tgt>
// tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --strip_debug_ops=* --control_flow_v2=* --skip_op_check --quantize_float16 <src> <tgt>
// tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --strip_debug_ops=* --control_flow_v2=* --skip_op_check --quantize_uin16 <src> <tgt>
// tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --strip_debug_ops=* --control_flow_v2=* --skip_op_check --quantize_uint8 <src> <tgt>
