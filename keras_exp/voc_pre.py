"""Object Detectionのためのベースネットワークの事前学習。"""
import pathlib

import sklearn.metrics

import pytoolkit as tk

BATCH_SIZE = 100
MAX_EPOCH = 300


def _create_model(nb_classes: int, input_shape: tuple):
    import keras
    import keras.backend as K

    def _conv(x, filters, kernel_size, name=None, **kargs):
        assert name is not None
        x = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name=name + '_conv', **kargs)(x)
        x = keras.layers.BatchNormalization(name=name + '_bn')(x)
        x = keras.layers.ELU(name=name + '_act')(x)
        return x

    def _branch(x, filters, name):
        x = _conv(x, filters, (3, 3), padding='same', name=name + '_c1')
        x = keras.layers.Dropout(0.25, name=name + '_drop')(x)
        x = _conv(x, filters, (3, 3), padding='same', name=name + '_c2')
        return x

    def _1x1conv(x, filters, name):
        if filters is None:
            filters = K.int_shape(x)[-1] // 2
        x = _conv(x, filters, (1, 1), name=name)
        return x

    def _block(x, name):
        filters = K.int_shape(x)[-1]

        for i in range(4):
            b = _branch(x, filters // 4, name=name + '_b' + str(i))
            x = keras.layers.Concatenate()([x, b])

        x = _1x1conv(x, filters * 2, name=name + '_sq')
        return x

    def _small_block(x, filters, name):
        if K.int_shape(x)[-1] != filters:
            x = _conv(x, filters, (1, 1), padding='same', name=name + '_pre')
        sc = x
        x = keras.layers.Conv2D(filters // 4, (1, 1), padding='same', use_bias=False, name=name + '_b1')(x)
        x = keras.layers.BatchNormalization(name=name + '_b1bn')(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(filters // 4, (3, 3), padding='same', use_bias=False, name=name + '_b2')(x)
        x = keras.layers.BatchNormalization(name=name + '_b2bn')(x)
        x = keras.layers.ELU(name=name + '_b2act')(x)
        x = keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, name=name + '_b3')(x)
        x = keras.layers.Add()([sc, x])
        return x

    def _ds(x, kernel_size, strides, name):
        x = keras.layers.AveragePooling2D(kernel_size, strides=strides, name=name + '_pool')(x)
        return x

    x = inp = keras.layers.Input(input_shape)
    x = _conv(x, 32, (3, 3), padding='same', name='start_cifar100')

    x = _block(x, 'stage1_block')  # ch=64
    x = _ds(x, (2, 2), (2, 2), 'stage1_ds')  # 96
    x = _block(x, 'stage2_block')  # ch=128
    x = _ds(x, (2, 2), (2, 2), 'stage2_ds')
    x = _block(x, 'stage3_block')  # ch=256
    x = _ds(x, (2, 2), (2, 2), 'stage3_ds')

    x = _block(x, 'stage4_block')  # ch=512
    x = _ds(x, (2, 2), (2, 2), 'stage4_ds')

    x = _1x1conv(x, 256, name='last_cifar100')
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(nb_classes, activation='softmax', kernel_regularizer='l2')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile('nadam', 'categorical_crossentropy', ['acc'])
    return model


def run(logger, result_dir: pathlib.Path):
    """実行。"""
    import keras
    import keras.preprocessing.image

    input_shape = (64, 64, 3)
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    model = _create_model(nb_classes, input_shape)
    model.summary(print_fn=logger.debug)
    logger.debug('layer depth: %d', sum(isinstance(l, keras.layers.Conv2D) for l in model.layers))
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

    callbacks = []
    callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=1e-3, max_reduces=0))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
    # if K.backend() == 'tensorflow':
    #     callbacks.append(keras.callbacks.TensorBoard())

    gen = tk.image.ImageDataGenerator(input_shape[:2])
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.125, tk.image.RandomBlur())
    gen.add(0.125, tk.image.RandomBlur(partial=True))
    gen.add(0.125, tk.image.RandomUnsharpMask())
    gen.add(0.125, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.125, tk.image.Sharp())
    gen.add(0.125, tk.image.Sharp(partial=True))
    gen.add(0.125, tk.image.Soft())
    gen.add(0.125, tk.image.Soft(partial=True))
    gen.add(0.125, tk.image.RandomMedian())
    gen.add(0.125, tk.image.RandomMedian(partial=True))
    gen.add(0.125, tk.image.GaussianNoise())
    gen.add(0.125, tk.image.GaussianNoise(partial=True))
    gen.add(0.125, tk.image.RandomSaturation())
    gen.add(0.125, tk.image.RandomBrightness())
    gen.add(0.125, tk.image.RandomContrast())
    gen.add(0.125, tk.image.RandomLighting())

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
