"""CIFAR100."""
import os
import pathlib
import time

import better_exceptions
import numpy as np
import sklearn.metrics

import pytoolkit as tk

BATCH_SIZE = 50
MAX_EPOCH = 100
USE_NADAM = False


def _create_model(nb_classes: int, input_shape: tuple):
    import keras
    from keras.regularizers import l2

    def _conv2d(*args, **kargs):
        def _l(x):
            activation = kargs.pop('activation')
            x = keras.layers.Conv2D(*args, **kargs, use_bias=False,
                                    kernel_regularizer=l2(1e-4),
                                    bias_regularizer=l2(1e-4))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation)(x)
            return x
        return _l

    x = inp = keras.layers.Input(input_shape)
    x = _conv2d(128, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(128, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(128, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = _conv2d(256, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(256, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(256, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = _conv2d(512, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(512, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(512, (3, 3), padding='same', activation='relu')(x)
    x = _conv2d(512, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(nb_classes, activation='softmax',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4),
                           name='predictions')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    if USE_NADAM:
        model.compile('nadam', 'categorical_crossentropy', ['acc'])
    else:
        model.compile(keras.optimizers.SGD(momentum=0.9, nesterov=True), 'categorical_crossentropy', ['acc'])
    return model


def _run2(logger, result_dir: pathlib.Path):
    import keras
    import keras.preprocessing.image

    input_shape = (32, 32, 3)
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()

    model = _create_model(nb_classes, input_shape)
    model.summary(print_fn=logger.debug)
    logger.debug('network depth: %d', tk.dl.count_network_depth(model))
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

    callbacks = []
    callbacks.append(keras.callbacks.CSVLogger(str(result_dir / 'history.tsv'), separator='\t'))
    callbacks.append(tk.dl.learning_rate_callback(lr=1e-3 if USE_NADAM else 1e-1, epochs=MAX_EPOCH))
    callbacks.append(tk.dl.learning_curve_plotter(result_dir.joinpath('history.{metric}.png'), 'loss'))
    callbacks.append(tk.dl.learning_curve_plotter(result_dir.joinpath('history.{metric}.png'), 'acc'))

    gen = tk.image.ImageDataGenerator(input_shape[:2], label_encoder=tk.ml.to_categorical(nb_classes))
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.5, tk.image.RandomErasing())
    gen.add(0.25, tk.image.RandomBlur())
    gen.add(0.25, tk.image.RandomBlur(partial=True))
    gen.add(0.25, tk.image.RandomUnsharpMask())
    gen.add(0.25, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.25, tk.image.RandomMedian())
    gen.add(0.25, tk.image.GaussianNoise())
    gen.add(0.25, tk.image.GaussianNoise(partial=True))
    gen.add(0.25, tk.image.RandomSaturation())
    gen.add(0.25, tk.image.RandomBrightness())
    gen.add(0.25, tk.image.RandomContrast())
    gen.add(0.25, tk.image.RandomHue())

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
    logger = tk.create_tee_logger(result_dir.joinpath('output.log'), fmt=None)

    start_time = time.time()
    _run(logger, result_dir)
    elapsed_time = time.time() - start_time

    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


if __name__ == '__main__':
    _main()
