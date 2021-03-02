const tf = require('@tensorflow/tfjs-node');

async function main() {
  const modelPath = 'file://models/blazepalm/graph/blazepalm.json';
  console.log('model path:', modelPath);
  console.log('platform: browser', tf.env().getBool('IS_BROWSER'), 'node', tf.env().getBool('IS_NODE'));
  const model = await tf.loadGraphModel(modelPath);
  console.log('loaded using:', model.handler);
  console.log('bytes used:', tf.engine().state.numBytes);
  console.log('model signature:', model.signature);
  tf.dispose(model);
}

main();
