import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow.keras.applications as apps
from tensorflow.keras.models import Model as _Model
from tensorflow.keras import losses, regularizers


def wrap(tensor):
    return layers.TimeDistributed(tensor, name="td_" + tensor.name)


class Model:
    def pre_branch(self, input):
        x = wrap(layers.DepthwiseConv2D(11, 2, padding="same",
                                        activation=None, use_bias=False))(input)
        x = wrap(layers.BatchNormalization())(x)
        x = wrap(layers.ReLU(6))(x)

        x = wrap(layers.Conv2D(64, 1, padding="same",
                               activation=None, use_bias=False))(x)
        x = wrap(layers.BatchNormalization())(x)

        x = wrap(layers.SpatialDropout2D(0.2))(x)

        x = wrap(layers.DepthwiseConv2D(5, 2, padding="same",
                                        activation=None, use_bias=False))(x)
        x = wrap(layers.BatchNormalization())(x)
        x = wrap(layers.ReLU(6))(x)

        x = wrap(layers.Conv2D(128, 1, padding="same",
                               activation=None, use_bias=False))(x)
        x = wrap(layers.BatchNormalization())(x)

        x = wrap(layers.SpatialDropout2D(0.2))(x)

        x = layers.ConvLSTM2D(16, 3, 1, padding="same", activation="relu")(x)

        x = layers.SpatialDropout2D(0.3)(x)

        return x

    def post_branch(self, input):
        x = layers.GlobalAveragePooling2D()(input)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(8, activation="sigmoid", name="Output")(x)

        return x

    def __init__(self, image_resultion):

        batch_size = None
        self.input_layer = layers.Input(
            shape=[None] + image_resultion, batch_size=batch_size, dtype=tf.float32, name="Input")

        pre_branch = self.pre_branch(self.input_layer)
        mobile_net = apps.mobilenet_v2.MobileNetV2(
            input_tensor=pre_branch, include_top=False, weights=None)
        post_branch = self.post_branch(mobile_net.outputs[0])

        self.model = _Model(inputs=self.input_layer,
                            outputs=post_branch)

    @property
    def loss(self):
        def loss(y_true, y_pred):
            return losses.binary_crossentropy(y_true, y_pred)

        return loss