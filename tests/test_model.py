import tensorflow as tf
from mnist_model.model import SimpleModel


def test_simple_model():
    image_shape = (28, 28, 1)
    num_filter_layer_1 = 32
    num_filter_layer_2 = 64
    kernel_size_layers = (3, 3)
    dropout_rate = 0.25
    num_units = 128
    num_labels = 10

    model = SimpleModel(image_shape, num_filter_layer_1, num_filter_layer_2, kernel_size_layers, dropout_rate, num_units, num_labels)
    assert model.image_shape == image_shape
    assert model.num_filter1 == num_filter_layer_1
    assert model.num_filter2 == num_filter_layer_2
    assert model.kernel_size == kernel_size_layers
    assert model.dropout_rate == dropout_rate
    assert model.num_units == num_units
    assert model.num_labels == num_labels

    # test the call method
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (1, 10)
