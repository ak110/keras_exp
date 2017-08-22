"""CIFAR100."""
import pathlib

import numpy as np
import sklearn.metrics

import pytoolkit as tk

BATCH_SIZE = 100
MAX_EPOCH = 300


def create_model(nb_classes: int, input_shape: tuple):
    import keras
    import keras.backend as K

    def conv(x, *args, dropout=None, **kargs):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        if dropout:
            x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Conv2D(*args, **kargs, use_bias=False)(x)
        return x

    def branch(x, filters):
        x = conv(x, filters // 4, (1, 1), padding='same')
        x = conv(x, filters, (3, 3), padding='same', dropout=0.25)
        return x

    def block(x, filters):
        x0 = x
        x1 = x = branch(x, filters)
        x2 = x = branch(x, filters)
        x3 = x = branch(x, filters)
        x = keras.layers.Add()([
            conv(x0, filters, (1, 1)),
            conv(x1, filters, (1, 1)),
            conv(x2, filters, (1, 1)),
            conv(x3, filters, (1, 1)),
        ])
        return x

    def ds(x):
        filters = K.int_shape(x)[-1]
        sq = conv(x, filters // 4, (1, 1))
        return keras.layers.Concatenate()([
            keras.layers.MaxPooling2D()(x),
            conv(sq, filters, (3, 3), strides=(2, 2), padding='same'),
        ])

    x = inp = keras.layers.Input(input_shape)
    x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = conv(x, 128, (3, 3), padding='same')
    x = block(x, 128)
    x = ds(x)
    x = block(x, 256)
    x = ds(x)
    x = block(x, 512)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(nb_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile('nadam', 'categorical_crossentropy', ['acc'])
    return model


def run(logger, result_dir: pathlib.Path):
    import keras
    import keras.preprocessing.image

    input_shape = (32, 32, 3)
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    model = create_model(nb_classes, input_shape)
    model.summary(print_fn=logger.debug)
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

    callbacks = []
    callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=1e-3))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
    # if K.backend() == 'tensorflow':
    #     callbacks.append(keras.callbacks.TensorBoard())
    gen = tk.image.ImageDataGenerator((32, 32))
    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=BATCH_SIZE, data_augmentation=True, shuffle=True),
        steps_per_epoch=gen.steps_per_epoch(X_train.shape[0], BATCH_SIZE),
        epochs=MAX_EPOCH,
        validation_data=gen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        validation_steps=gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE),
        callbacks=callbacks)

    model.save(str(result_dir.joinpath('model.h5')))

    score = model.evaluate_generator(
        gen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE))
    logger.info('Test loss:     {}'.format(score[0]))
    logger.info('Test accuracy: {}'.format(score[1]))
    logger.info('Test error:    {}'.format(1 - score[1]))

    pred = model.predict_generator(
        gen.flow(X_test, batch_size=BATCH_SIZE),
        gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE))
    cm = sklearn.metrics.confusion_matrix(y_test.argmax(axis=-1), pred.argmax(axis=-1))
    tk.ml.plot_cm(cm, result_dir.joinpath('confusion_matrix.png'))
