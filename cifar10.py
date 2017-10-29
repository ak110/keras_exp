"""CIFAR10."""
import os
import pathlib
import time

import better_exceptions
import numpy as np
import sklearn.metrics

import pytoolkit as tk

BATCH_SIZE = 100
MAX_EPOCH = 300


def _create_model(nb_classes: int, input_shape: tuple):
    import keras
    import keras.backend as K

    def _conv(x, filters, kernel_size, name=None, **kargs):
        assert name is not None
        x = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name=name, **kargs)(x)
        x = keras.layers.BatchNormalization(name=name + '_bn')(x)
        x = keras.layers.ELU(name=name + '_act')(x)
        return x

    def _branch(x, filters, name):
        x = _conv(x, filters // 2, (1, 1), padding='same', name=name + '_sq')
        x = keras.layers.Dropout(0.25, name=name + '_do')(x)
        x = _conv(x, filters, (3, 3), padding='same', name=name + '_ex')
        return x

    def _block(x, filters, name):
        x0 = x
        x1 = x = _branch(x, filters, name=name + '_b1')
        x2 = x = _branch(x, filters, name=name + '_b2')
        x3 = x = _branch(x, filters, name=name + '_b3')
        x4 = x = _branch(x, filters, name=name + '_b4')
        x = keras.layers.Concatenate()([x1, x2, x3, x4])
        x = _conv(x, filters, (1, 1), name=name + '_mixed')
        x = keras.layers.Add()([x0, x])
        return x

    def _ds(x, name):
        filters = K.int_shape(x)[-1]

        mp = keras.layers.MaxPooling2D()(x)

        cv = _conv(x, filters // 4, (1, 1), name=name + '_sq')
        cv = _conv(cv, filters, (3, 3), strides=(2, 2), padding='same', name=name + '_ds')

        x = keras.layers.Concatenate()([mp, cv])
        return x

    x = inp = keras.layers.Input(input_shape)
    x = _conv(x, 128, (3, 3), padding='same', name='start')
    x = _block(x, 128, name='stage1_block')
    x = _ds(x, name='stage1_ds')
    x = _block(x, 256, name='stage2_block')
    x = _ds(x, name='stage2_ds')
    x = _block(x, 512, name='stage3_block')
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(nb_classes, activation='softmax', kernel_regularizer='l2', name='predictions')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile('nadam', 'categorical_crossentropy', ['acc'])
    return model


def _run2(logger, result_dir: pathlib.Path):
    import keras
    import keras.preprocessing.image

    input_shape = (32, 32, 3)
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    model = _create_model(nb_classes, input_shape)
    model.summary(print_fn=logger.debug)
    logger.debug('network depth: %d', tk.dl.count_network_depth(model))
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

    callbacks = []
    callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=1e-3))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
    # if K.backend() == 'tensorflow':
    #     callbacks.append(keras.callbacks.TensorBoard())

    gen = tk.image.ImageDataGenerator(input_shape[:2], label_encoder=tk.ml.to_categorical(nb_classes))
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.5, tk.image.RandomErasing())
    gen.add(0.125, tk.image.RandomBlur())
    gen.add(0.125, tk.image.RandomBlur(partial=True))
    gen.add(0.125, tk.image.RandomUnsharpMask())
    gen.add(0.125, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.125, tk.image.Sharp())
    gen.add(0.125, tk.image.Soft())
    gen.add(0.125, tk.image.RandomMedian())
    gen.add(0.125, tk.image.GaussianNoise())
    gen.add(0.125, tk.image.GaussianNoise(partial=True))
    gen.add(0.125, tk.image.RandomSaturation())
    gen.add(0.125, tk.image.RandomBrightness())
    gen.add(0.125, tk.image.RandomContrast())
    gen.add(0.125, tk.image.RandomHue())

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
    cm = sklearn.metrics.confusion_matrix(y_test, pred.argmax(axis=-1))
    tk.ml.plot_cm(cm, result_dir.joinpath('confusion_matrix.png'))


def _run(logger, result_dir: pathlib.Path):
    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        _run2(logger, result_dir)


def _main():
    import matplotlib as mpl
    mpl.use('Agg')

    better_exceptions.MAX_LENGTH = 128

    base_dir = pathlib.Path(os.path.realpath(__file__)).parent
    os.chdir(str(base_dir))
    np.random.seed(1337)  # for reproducibility

    result_dir = base_dir.joinpath('results', pathlib.Path(__file__).stem)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = tk.create_tee_logger(result_dir.joinpath('output.log'))

    start_time = time.time()
    _run(logger, result_dir)
    elapsed_time = time.time() - start_time

    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


if __name__ == '__main__':
    _main()
