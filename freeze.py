import tensorflow as tf

# configure as needed
input_model_dir = "nudenet/default"
frozen_model_dir = "nudenet/frozen"
saved_model_dir = "nudenet/saved"

# from "saved_model_cli show --dir nudenet/default --all"
tag = "serve"
signature = "predict" # tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY is default
input_nodes = ["input_1"]
output_nodes = ["filtered_detections/map/TensorArrayStack/TensorArrayGatherV3", "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3", "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3"]

# load saved model with variables
# everything is based on tf.compat.v1 since tf v2 handles things totally differently
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session(graph=tf.Graph())
model = tf.compat.v1.saved_model.loader.load(sess, [tag], input_model_dir)
graph_def = sess.graph.as_graph_def()

# clean it up
clean = tf.compat.v1.graph_util.remove_training_nodes(graph_def)
# convert variables to constants
frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, clean, output_nodes)
# write frozen graph if needed for future usage, but can be skipped here since it's already in current session
tf.io.write_graph(frozen, frozen_model_dir, "saved_model.pb", as_text=False)

# rename model input/outputs to expected format
def get_ops_dict(ops, graph, name):
  out_dict = dict()
  for i, op in enumerate(ops):
    out_dict[name + str(i)] = tf.compat.v1.saved_model.build_tensor_info(graph.get_tensor_by_name(op + ':0'))
  return out_dict

# finally create a clean saved model
with tf.Graph().as_default() as graph:
  tf.python.framework.importer.import_graph_def(frozen, name="")
  inputs_dict = get_ops_dict(input_nodes, graph, name='input_')
  outputs_dict = get_ops_dict(output_nodes, graph, name='output_')
  prediction_signature = (
    tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs_dict,
    outputs=outputs_dict,
    method_name=tf.saved_model.PREDICT_METHOD_NAME))
  legacy_init_op = tf.group(tf.compat.v1.tables_initializer(), name='legacy_init_op')
  builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_dir)
  builder.add_meta_graph_and_variables(
    sess,
    tags=[tag],
    signature_def_map={signature: prediction_signature},
    legacy_init_op=legacy_init_op)
  builder.save()
