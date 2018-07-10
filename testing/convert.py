from keras.models import load_model
import coremltools
import tensorflow as tf
from keras import backend as K
from keras.utils import custom_object_scope

import sys
sys.path.append("../utils/")
import relu6, smoothL1

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph

MODEL_PATH = "../landmark_model/Mobilenet_v1.hdf5"
ML_MODEL_PATH = "../landmark_model/Mobilenet_v1.mlmodel"

def keras_to_coreml():
    with custom_object_scope({'smoothL1': smoothL1, 'relu6': relu6}):
        ml_model = load_model(MODEL_PATH)
    coreml_model = coremltools.converters.keras.convert(ml_model, 
                                                        input_names='image', image_input_names='image', 
                                                        is_bgr=False)
    coreml_model.save(ML_MODEL_PATH)

if __name__ == "__main__":
    keras_to_coreml()