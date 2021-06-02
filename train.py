import numpy as np
import tensorflow as tf
from asoftmax import Asoftmax
from tensorflow.keras import regularizers


class DNN(tf.keras.models.Model):
    def __init__(self, num_classes=10, weight_decay=1e-4):
        super(DNN, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(32, activation="relu")
        self.layer_2 = tf.keras.layers.Dense(10)

        self.out = Asoftmax(
            n_classes=num_classes,
            regularizer=regularizers.l2(weight_decay),
        )

    def call(self, x, training=False):
        if training:
            x, y = x[0], x[1]
        # x, y = x[0], x[1]
        x = self.layer_1(x)
        x = self.layer_2(x)
        if training:
            out = self.out([x, y])
        else:
            out = self.out(x)
        # out = self.out([x, y])
        return out


def preprocessing(data, labels):
    data = x_train.reshape((len(data), -1)) / 255.0
    labels = tf.keras.utils.to_categorical(data, 10)
    return data, labels


def acc(pred, label):
    return len(pred[label == np.argmax(pred, axis=1)]) / len(pred)


if __name__ == "__main__":
    epochs = 5
    batch_size = 256

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = preprocessing(x_train, y_train)
    model = DNN()
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.categorical_crossentropy

    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
    model.fit(
        [x_train, y_train],
        y_train,
        epochs=5,
        batch_size=batch_size,
    )

    x_train, _ = preprocessing(x_train, y_train)

    pred = model.predict(x_test)
    accuracy = acc(pred, y_test)
    print(f"Accuracy: {acc*100}%")
