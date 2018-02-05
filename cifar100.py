"""CIFAR100."""
import pathlib
import time

import better_exceptions
import horovod.keras as hvd
import numpy as np

import pytoolkit as tk

BATCH_SIZE = 100
MAX_EPOCH = 300


def _create_model(nb_classes: int, input_shape: tuple):
    import keras

    builder = tk.dl.Builder()
    builder.set_default_l2()
    builder.conv_defaults['kernel_initializer'] = 'he_uniform'

    def _block(x, filters, res_count, name):
        for res in range(res_count):
            sc = x
            x = builder.conv2d(filters, (3, 3), name='{}_r{}c1'.format(name, res))(x)
            x = keras.layers.Dropout(0.25)(x)
            x = builder.conv2d(filters, (3, 3), use_act=False, name='{}_r{}c2'.format(name, res))(x)
            x = keras.layers.Add()([sc, x])
        x = builder.bn()(x)
        x = builder.act()(x)
        return x

    def _tran(x, filters, name):
        x = builder.conv2d(filters, (1, 1), name='{}_conv'.format(name))(x)
        x = keras.layers.MaxPooling2D()(x)
        return x

    x = inp = keras.layers.Input(input_shape)
    x = builder.conv2d(128, (3, 3), name='start')(x)
    x = _block(x, 128, 4, name='stage1_block')
    x = _tran(x, 256, name='stage1_tran')
    x = _block(x, 256, 12, name='stage2_block')
    x = _tran(x, 512, name='stage2_tran')
    x = _block(x, 512, 4, name='stage3_block')
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(nb_classes, activation='softmax', name='predictions')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    return model


def _run2(logger, result_dir: pathlib.Path):
    import keras
    import keras.preprocessing.image

    input_shape = (32, 32, 3)
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()

    model = _create_model(nb_classes, input_shape)

    # 学習率：
    # ・lr 0.5、batch size 256くらいが多いのでその辺を基準に
    # ・バッチサイズに比例させるのが良いとのうわさ
    lr = 0.5 * BATCH_SIZE / 256 * hvd.size()
    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(opt, 'categorical_crossentropy', ['acc'])

    if hvd.rank() == 0:
        model.summary(print_fn=logger.info)
        logger.info('network depth: %d', tk.dl.count_network_depth(model))
        # keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)
        # tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

    callbacks = []
    callbacks.append(tk.dl.learning_rate_callback(lr=lr, epochs=MAX_EPOCH))
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
    if hvd.rank() == 0:
        callbacks.append(tk.dl.tsv_log_callback(result_dir / 'history.tsv'))
        # callbacks.append(tk.dl.learning_curve_plot_callback(result_dir.joinpath('history.{metric}.png'), 'loss'))
        # callbacks.append(tk.dl.learning_curve_plot_callback(result_dir.joinpath('history.{metric}.png'), 'acc'))

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
    gen.add(0.5, tk.image.RandomSaturation())
    gen.add(0.5, tk.image.RandomBrightness())
    gen.add(0.5, tk.image.RandomContrast())
    gen.add(0.5, tk.image.RandomHue())

    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=BATCH_SIZE, data_augmentation=True, shuffle=True),
        steps_per_epoch=gen.steps_per_epoch(X_train.shape[0], BATCH_SIZE) // hvd.size(),
        epochs=MAX_EPOCH,
        verbose=1 if hvd.rank() == 0 else 0,
        validation_data=gen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=True),
        validation_steps=gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE) // hvd.size(),  # * 3は省略
        callbacks=callbacks)

    if hvd.rank() == 0:
        model.save(str(result_dir.joinpath('model.h5')))

        score = model.evaluate_generator(
            gen.flow(X_test, y_test, batch_size=BATCH_SIZE),
            gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE))
        logger.info('Test loss:     {}'.format(score[0]))
        logger.info('Test accuracy: {}'.format(score[1]))
        logger.info('Test error:    {}'.format(1 - score[1]))

        # pred = model.predict_generator(
        #     gen.flow(X_test, batch_size=BATCH_SIZE),
        #     gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE))
        # cm = sklearn.metrics.confusion_matrix(y_test, pred.argmax(axis=-1))
        # tk.ml.plot_cm(cm, result_dir.joinpath('confusion_matrix.png'))


def _run(logger, result_dir: pathlib.Path):
    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        _run2(logger, result_dir)


def _main():
    hvd.init()

    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot
    # assert matplotlib.pyplot is not None  # libpngのエラー対策(怪)

    better_exceptions.MAX_LENGTH = 128

    base_dir = pathlib.Path(__file__).resolve().parent
    result_dir = base_dir.joinpath('results', pathlib.Path(__file__).stem)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = tk.create_tee_logger(result_dir.joinpath('output.log'), fmt=None)

    start_time = time.time()
    _run(logger, result_dir)
    elapsed_time = time.time() - start_time

    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


if __name__ == '__main__':
    _main()
