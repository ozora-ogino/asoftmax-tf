# ASoftmax on tf.Keras

This repository contains code for ASoftmax, for other word ArcFace based on [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698).

**You don't need to pass labels to your model when predicting because I implemented predict operation and train operation differently!**

## Usage

Model

```train.py
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

        # Important!!
        if training:
            # When training, you need to pass label to ASoftmax
            out = self.out([x, y])
        else:
            out = self.out(x)
        return out

```

Please note that you need to use one-hot-encoding for training label.

## Contribution
Contribution is more than welcome!
If there are some problems, please open issue.