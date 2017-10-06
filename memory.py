#!/usr/bin/env python
"""GPUメモリ使用量を手動で調べてみた時のコード。"""
import os
import pathlib

import better_exceptions
import numpy as np

import pytoolkit as tk

BATCH_SIZE = 100
MAX_EPOCH = 300


def _create_model(nb_classes: int, input_shape: tuple):
    import keras

    x = inp = keras.layers.Input(input_shape)
    x = tk.dl.conv2d(64, (1, 1), padding='same', activation='relu', name='start')(x)

    # ■ 何もなし: 249MiB, #params=6,948

    # ■ conv×8: 729MiB, #params=303,908
    # for c in range(8):
    #     x = tk.dl.conv2d(64, (3, 3), padding='same', activation='relu', name='conv_' + str(c))(x)

    # ■ ResNet: 729MiB, #params=303,908
    # for c in range(4):
    #     sc = x
    #     x = tk.dl.conv2d(64, (3, 3), padding='same', activation='relu', name='conv1_' + str(c))(x)
    #     x = tk.dl.conv2d(64, (3, 3), padding='same', activation=None, name='conv2_' + str(c))(x)
    #     x = keras.layers.Add()([sc, x])

    # ■ DenseNet: 539MiB, #params=24,356
    # x = tk.dl.conv2d(32, (1, 1), padding='same', activation='relu', name='sq')(x)
    # for c in range(4):
    #     b = tk.dl.conv2d(8, (3, 3), padding='same', activation='relu', name='conv1_' + str(c))(x)
    #     b = tk.dl.conv2d(8, (3, 3), padding='same', activation='relu', name='conv2_' + str(c))(b)
    #     x = keras.layers.Concatenate()([x, b])

    # ■ DenseNet-C: 861MiB, #params=75,812
    # for c in range(4):
    #     b = tk.dl.conv2d(16, (3, 3), padding='same', activation='relu', name='conv1_' + str(c))(x)
    #     b = tk.dl.conv2d(16, (3, 3), padding='same', activation='relu', name='conv2_' + str(c))(b)
    #     x = keras.layers.Concatenate()([x, b])
    # x = tk.dl.conv2d(64, (1, 1), padding='same', activation='relu', name='sq')(x)

    # ■ DenseNet-BC: 797MiB, #params=76,068
    # for c in range(4):
    #     b = tk.dl.conv2d(16 * 4, (1, 1), padding='same', activation='relu', name='conv1_' + str(c))(x)
    #     b = tk.dl.conv2d(16, (3, 3), padding='same', activation='relu', name='conv2_' + str(c))(b)
    #     x = keras.layers.Concatenate()([x, b])
    # x = tk.dl.conv2d(64, (1, 1), padding='same', activation='relu', name='sq')(x)

    x = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Dense(nb_classes, activation='softmax', kernel_regularizer='l2', name='predictions')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile('nadam', 'categorical_crossentropy', ['acc'])
    return model


def _main():
    import matplotlib as mpl
    mpl.use('Agg')

    better_exceptions.MAX_LENGTH = 128

    base_dir = pathlib.Path(os.path.realpath(__file__)).parent
    os.chdir(str(base_dir))
    np.random.seed(1337)  # for reproducibility

    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        _run()


def _run():
    import keras
    import keras.preprocessing.image

    input_shape = (32, 32, 3)
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    model = _create_model(nb_classes, input_shape)
    model.summary()
    print('network depth: %d', tk.dl.count_network_depth(model))

    model.fit(X_train, y_train, epochs=MAX_EPOCH, validation_data=(X_test, y_test))


if __name__ == '__main__':
    _main()
