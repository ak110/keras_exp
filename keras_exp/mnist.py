"""MNIST."""
import pathlib

import numpy as np
import sklearn.metrics

import pytoolkit as tk

BATCH_SIZE = 100
MAX_EPOCH = 300
DATA_AUGMENTATION = True


def _create_model(nb_classes: int, input_shape: tuple):
    import keras

    def _conv(x, *args, **kargs):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(*args, **kargs, use_bias=False)(x)
        return x

    def _ds(x):
        x = keras.layers.MaxPooling2D()(x)
        return x

    x = inp = keras.layers.Input(input_shape)
    x = keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = _conv(x, 32, (3, 3), padding='same')
    x = _conv(x, 32, (3, 3), padding='same')
    x = _ds(x)
    x = _conv(x, 32, (3, 3), padding='same')
    x = _conv(x, 32, (3, 3), padding='same')
    x = _ds(x)
    x = _conv(x, 32, (3, 3), padding='same')
    x = _conv(x, 32, (3, 3), padding='same')
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(nb_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile('nadam', 'categorical_crossentropy', ['acc'])
    return model


def run(logger, result_dir: pathlib.Path):
    """実行。"""
    import keras
    import keras.preprocessing.image

    input_shape = (28, 28, 1)
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(X_train.shape + (1,)).astype(np.float32) / 255
    X_test = X_test.reshape(X_test.shape + (1,)).astype(np.float32) / 255
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    model = _create_model(nb_classes, input_shape)
    model.summary(print_fn=logger.debug)
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

    callbacks = []
    callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=1e-3))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
    # if K.backend() == 'tensorflow':
    #     callbacks.append(keras.callbacks.TensorBoard())
    if DATA_AUGMENTATION:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            channel_shift_range=0,
            horizontal_flip=False,
            vertical_flip=False)
        model.fit_generator(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=X_train.shape[0] // BATCH_SIZE, epochs=MAX_EPOCH,
            validation_data=(X_test, y_test), callbacks=callbacks)
    else:
        model.fit(
            X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH,
            validation_data=(X_test, y_test), callbacks=callbacks)

    model.save(str(result_dir.joinpath('model.h5')))

    score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    logger.info('Test loss:     {}'.format(score[0]))
    logger.info('Test accuracy: {}'.format(score[1]))

    pred = model.predict(X_test, batch_size=BATCH_SIZE)
    cm = sklearn.metrics.confusion_matrix(y_test.argmax(axis=-1), pred.argmax(axis=-1))
    tk.ml.plot_cm(cm, result_dir.joinpath('confusion_matrix.png'))
