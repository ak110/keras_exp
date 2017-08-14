"""CIFAR100."""
import pathlib

import numpy as np
import pytoolkit as tk

BATCH_SIZE = 100
MAX_EPOCH = 300
DATA_AUGMENTATION = True


def create_model(nb_classes: int):
    import keras
    import keras.backend as K

    def conv(*args, **kargs):
        def _t(x):
            x = keras.layers.Conv2D(*args, **kargs, use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ELU()(x)
            return x
        return _t

    def block(x, nb_filter):
        x0 = x = conv(nb_filter // 2, (1, 1), padding='same')(x)
        x1 = x = conv(nb_filter // 2, (3, 3), padding='same')(x)
        x2 = x = conv(nb_filter // 2, (3, 3), padding='same')(x)
        x3 = x = conv(nb_filter // 2, (3, 3), padding='same')(x)
        x = keras.layers.Concatenate()([x0, x1, x2, x3])
        x = conv(nb_filter, (1, 1), padding='same')(x)
        return x

    def ds(x):
        filters = K.int_shape(x)[-1]
        return keras.layers.Concatenate()([
            keras.layers.MaxPooling2D()(x),
            conv(filters, (3, 3), strides=(2, 2), padding='same')(x),
        ])

    x = inp = keras.layers.Input((32, 32, 3))

    x = block(x, 128)
    x = ds(x)  # 16
    x = block(x, 256)
    x = ds(x)  # 8
    x = block(x, 512)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(nb_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile(keras.optimizers.SGD(momentum=0.9, nesterov=True), 'categorical_crossentropy', ['acc'])
    return model


def run(result_dir: pathlib.Path, logger):
    import keras
    import keras.preprocessing.image

    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    model = create_model(nb_classes)
    model.summary(print_fn=logger.debug)
    keras.utils.plot_model(model, to_file=str(result_dir.joinpath('model.png')), show_shapes=True)

    callbacks = []
    callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=0.1))
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
            horizontal_flip=True,
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
