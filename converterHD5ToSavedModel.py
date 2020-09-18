from keras import backend as K
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  from keras.models import load_model
  model = load_model(model)

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                    outputs={'income': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()

savedModel = "/home/pablovin/experiments/facechannel/weights.01-0.02.h5"

exportModel = "/home/pablovin/experiments/facechannel/allFramesTrained"

import tensorflow as tf
model = tf.keras.models.load_model(savedModel)
tf.saved_model.save(model, exportModel)
