# tfjs utils

collection of generic js utils when working with tf models

## tools

- `signature.js`: analyze saved model or graph model input/output tensors based on either model signature or model executor
- `classifier.js`: generic tf image classifier
- `detector.js`: generic tf object detector
- `freeze.py`: convert tf frozen model to saved model
- `kernels.js`: generate html table of supported tfjs kernels per backend

## other

- `tfjs`: custom built tfjs esm modules  
  moved to `vladmandic/tfjs`
- `/server/serve.js` lightweight http/https server with built-in watcher and built toolchain  
  integrated into `vladmandic/build`

## classes

`/classes/*.json`:  
contains class label definitions for different datasets in JSON format:

- CoCo
- ImageNet 1k
- OpenImages v4
- DeepDetect
- Places365

`/classes/wordnet-synset.json.gz`:  
full wordnet word definitions that can be used to parse word IDs included in individual classes definitions
