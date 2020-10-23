TODO:
- Quantize NudeNet
- Convert EfficientDet
- Test CenterNet



- I've checked out TFJS from master and did a full rebuild
- Fresh download of a `saved_model` [CenterNet]<https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1> and converted it using `tensorflowjs_converter`
- Loaded `saved_model` using `loadSavedModel`:  
  Result: **Pass**
- Loaded `graph_model` using `loadGraphModel`  
  Result: **Fail**

```log
[stack]: "TypeError: Cannot read property 'children' of undefined\n" +
    '    at /home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3523:33\n' +
    '    at Array.forEach (<anonymous>)\n' +
    '    at /home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3520:29\n' +
    '    at Array.forEach (<anonymous>)\n' +
    '    at OperationMapper.mapFunction (/home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3518:18)\n' +
    '    at /home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3353:55\n' +
    '    at Array.reduce (<anonymous>)\n' +
    '    at OperationMapper.transformGraph (/home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:3352:48)\n' +
    '    at GraphModel.loadSync (/home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:6910:68)\n' +
    '    at GraphModel.load (/home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:6892:21)\n' +
    '    at async Object.loadGraphModel (/home/vlado/dev/tfjs/tfjs-converter/dist/tf-converter.node.js:7160:5)\n' +
    '    at async processGraphModel (/home/vlado/dev/node-detector-test/detector.js:199:27)\n' +
    '    at async testSingle (/home/vlado/dev/node-detector-test/detector.js:283:3)\n' +
    '    at async main (/home/vlado/dev/node-detector-test/detector.js:301:3)',
```
