import numpy as np
import tensorflow as tf
from asoftmax import ASoftmax
from tensorflow.keras import regularizers


class DNN(tf.keras.models.Model):
    def __init__(self, num_classes=10, weight_decay=1e-4):
        super(DNN, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(32, activation="relu")
        self.layer_2 = tf.keras.layers.Dense(10)

        self.out = ASoftmax(
            n_classes=num_classes,
            regularizer=regularizers.l2(weight_decay),
        )

    def call(self, x, training=False):
        if training:
            x, y = x[0], x[1]
        x = self.layer_1(x)
        x = self.layer_2(x)

        if training:
            # When training, you need to pass label to ASoftmax
            out = self.out([x, y])
        else:
            out = self.out(x)
        return out


def preprocessing(data, labels):
    data = data.reshape((len(data), -1)) / 255.0
    labels = tf.keras.utils.to_categorical(labels, 10)
    return data, labels


def acc(pred, label):
    return len(pred[label == np.argmax(pred, axis=1)]) / len(pred)


if __name__ == "__main__":
    epochs = 10
    batch_size = 256

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = preprocessing(x_train, y_train)

    model = DNN()
    optimizer = tf.keras.optimizers.Adam()
    # Please note that you need to use categorical_crossentropy. (NOT sparse_categorical_crossentropy)
    loss = tf.keras.losses.categorical_crossentropy

    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])

    model.fit(
        [x_train, y_train],
        y_train,
        epochs=5,
        batch_size=batch_size,
    )

    x_test, _ = preprocessing(x_test, y_test)

    pred = model.predict(x_test)
    accuracy = acc(pred, y_test)
    print(f"Accuracy: {accuracy*100}%")  # Accuracy: 96.07%
