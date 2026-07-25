import os
import random
import numpy as np
import tensorflow as tf
import keras


def set_seed(seed=42):
    """
    Configure deterministic behaviour for TensorFlow,
    NumPy and Python's random module.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)


def tensorflow_version():
    """
    Return the installed TensorFlow version.
    """

    return tf.__version__


def keras_version():
    """
    Return the installed Keras version.
    """

    return keras.__version__