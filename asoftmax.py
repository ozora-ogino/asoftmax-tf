import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers


class Asoftmax(tf.keras.layers.Layer):
    def __init__(
        self,
        n_classes=10,
        scale=30.0,
        margin=0.50,
        regularizer=None,
        **kwargs,
    ):
        super(Asoftmax, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(Asoftmax, self).build(input_shape[0])
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[0][-1], self.n_classes),
            initializer="glorot_uniform",
            trainable=True,
            regularizer=self.regularizer,
        )

    def _train_op(self, inputs):
        x, y = inputs

        # Normalization
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        # Dot product
        logits = x @ W

        # Add margin and clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.margin)
        logits = logits * (1 - y) + target_logits * y

        # Rescale the feature
        logits *= self.scale
        out = tf.nn.softmax(logits)
        return out

    def _predict_op(self, inputs):
        # Normalization
        x = tf.nn.l2_normalize(inputs, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        out = tf.nn.softmax(logits)
        return out

    def call(self, inputs, training=False):
        if training:
            out = self._train_op(inputs)
        else:
            out = self._predict_op(inputs)
        return out