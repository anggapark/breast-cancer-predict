""" 
This test ensures that the model's output shape matches the expected shape 
by verifying input shape by model and produces expected output shape

image data shape:
<_PrefetchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int64, name=None))

"""


import unittest
import numpy as np
import tensorflow as tf
from cnn_model import create_model
from transfer_learning_model import build_model

BATCH_SIZE = 32
HEIGHT = 224
WIDTH = 224
CHANNELS = 3


class TestCNNModel(unittest.TestCase):
    def setUp(self):
        # self.model = create_model()
        self.model = build_model()

    def tearDown(self) -> None:
        del self.model
        tf.keras.backend.clear_session()

    def create_dummy_data(self, BATCH_SIZE, HEIGHT, WIDTH, channels):
        return np.random.rand(BATCH_SIZE, HEIGHT, WIDTH, channels)

    def test_model_output_shape(self):
        # input_shape = (None, 224, 224, 3)
        inputs = self.create_dummy_data(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
        outputs = self.model(inputs)
        expected_output_shape = (BATCH_SIZE, 1)
        self.assertEqual(outputs.shape, expected_output_shape)

    def test_model_summary(self):
        self.model.summary()


if __name__ == "__main__":
    unittest.main()
