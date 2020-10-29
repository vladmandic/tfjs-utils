<https://github.com/lutzroeder/netron>

prep the toolkit
```
git clone https://github.com/tensorflow/tensorflow
bazel build --local_ram_resources=HOST_RAM*.5 --local_cpu_resources=HOST_CPUS*.5 tensorflow/tools/graph_transforms:*
bazel build --local_ram_resources=HOST_RAM*.5 --local_cpu_resources=HOST_CPUS*.5 tensorflow/python/tools:freeze_graph
```

get model signatures:
```
saved_model_cli show --dir=nudenet/default --all
  MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
  signature_def['predict']:
    The given SavedModel SignatureDef contains the following input(s):
      inputs['images'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, -1, -1, 3)
          name: input_1:0
    The given SavedModel SignatureDef contains the following output(s):
      outputs['output1'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 300, 4)
          name: filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0
      outputs['output2'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 300)
          name: filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0
      outputs['output3'] tensor_info:
          dtype: DT_INT32
          shape: (-1, 300)
          name: filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0
    Method name is: tensorflow/serving/predict
```

convert checkpoint to frozen:
```
tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_saved_model_dir=nudenet/default \
  --output_node_names=filtered_detections/map/TensorArrayStack/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3 \
  --output_graph=nudenet/frozen/saved_model.pb
```

convert frozen to saved:
```
python tensorflow/tensorflow/python/tools/optimize_for_inference.py \
  --input=nudenet/saved/saved_model.pb -frozen_graph \
  --input_names=input_1 \
  --output_names=filtered_detections/map/TensorArrayStack/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3 \
  --output=nudenet/test/saved_model.pb
```

looking at saved model, all variables are gone and it's all constants now:
```
tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=nudenet/saved/saved_model.pb
  Found 1 possible inputs: (name=input_1, type=float(1), shape=[?,?,?,3])
  No variables spotted.
  Found 3 possible outputs: (name=filtered_detections/map/TensorArrayStack/TensorArrayGatherV3, op=TensorArrayGatherV3) (name=filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3, op=TensorArrayGatherV3) (name=filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3, op=TensorArrayGatherV3)
  Found 36615722 (36.62M) const parameters, 0 (0) variable parameters, and 348 control_edges
  Op types used: 865 Const, 127 StridedSlice, 111 BiasAdd, 111 Conv2D, 90 Relu, 82 Pack, 55 Reshape, 49 Mul, 44 Shape, 43 AddV2, 35 GatherV2, 33 GatherNd, 21 Fill, 17 Pad, 16 Greater, 16 Where, 16 NonMaxSuppressionV3, 15 Range, 13 Enter, 12 Cast, 10 Size, 7 Sub, 5 Transpose, 5 Sigmoid, 5 Switch, 5 ExpandDims, 5 Merge, 5 Tile, 5 TensorArrayV3, 5 NextIteration, 5 Maximum, 5 Minimum, 4 ConcatV2, 3 TensorArrayGatherV3, 3 TensorArraySizeV3, 3 TensorArrayWriteV3, 3 Exit, 3 PadV2, 2 ResizeNearestNeighbor, 2 TensorArrayReadV3, 2 TensorArrayScatterV3, 2 Less, 2 Unpack, 1 Placeholder, 1 MaxPool, 1 LoopCond, 1 LogicalAnd, 1 TopKV2, 1 Identity
```

let's try few more optimizations:
```
tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=nudenet/default/saved_model.pb \
  --out_graph=nudenet/optimized/saved_model.pb \
  --inputs=input_1 \
  --outputs=filtered_detections/map/TensorArrayStack/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3 \
  --transforms='strip_unused_nodes fold_constants fold_batch_norms fold_old_batch_norms'
```

and quantize from float32 to uint8:
```
tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=nudenet/saved/saved_model.pb \
  --out_graph=nudenet/quantized/saved_model.pb \
  --inputs=input_1 \
  --outputs=filtered_detections/map/TensorArrayStack/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3 \
  --transforms='strip_unused_nodes fold_constants fold_batch_norms fold_old_batch_norms quantize_weights quantize_nodes'
```
