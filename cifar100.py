"""CIFAR100."""
import pathlib
import time

import better_exceptions
import horovod.keras as hvd
import numpy as np

import pytoolkit as tk

BATCH_SIZE = 64
MAX_EPOCH = 300


@tk.log.trace()
def _create_model(num_classes: int, input_shape: tuple):
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
        x = builder.conv2d(filters, (3, 3), strides=(2, 2), use_act=False, name='{}_tran'.format(name))(x)
        return x

    x = inp = keras.layers.Input(input_shape)
    x = builder.conv2d(128, (3, 3), use_act=False, name='start')(x)
    x = _block(x, 128, 4, name='stage1_block')
    x = _tran(x, 256, name='stage1_tran')
    x = _block(x, 256, 12, name='stage2_block')
    x = _tran(x, 512, name='stage2_tran')
    x = _block(x, 512, 4, name='stage3_block')
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(num_classes, activation='softmax',
                      kernel_initializer='zeros', name='predictions')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    return model


@tk.log.trace()
def _run(result_dir: pathlib.Path):
    import keras
    import keras.preprocessing.image

    logger = tk.log.get(__name__)

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    model = _create_model(num_classes, input_shape)

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
    callbacks.append(tk.dl.learning_rate_callback())
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
    if hvd.rank() == 0:
        callbacks.append(tk.dl.tsv_log_callback(result_dir / 'history.tsv'))
        # callbacks.append(tk.dl.learning_curve_plot_callback(result_dir.joinpath('history.{metric}.png'), 'loss'))
        # callbacks.append(tk.dl.learning_curve_plot_callback(result_dir.joinpath('history.{metric}.png'), 'acc'))
    callbacks.append(tk.dl.freeze_bn_callback(0.95))

    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.ProcessOutput(tk.ml.to_categorical(num_classes), batch_axis=True))
    gen.add(tk.image.Mixup(probability=1, num_classes=num_classes))
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize(input_shape[:2]))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.5),
        tk.image.RandomUnsharpMask(probability=0.5),
        tk.image.GaussianNoise(probability=0.5),
        tk.image.RandomSaturation(probability=0.5),
        tk.image.RandomBrightness(probability=0.5),
        tk.image.RandomContrast(probability=0.5),
        tk.image.RandomHue(probability=0.5),
    ]))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.ProcessInput(tk.image.preprocess_input_abs1))

    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=BATCH_SIZE, data_augmentation=True, shuffle=True),
        steps_per_epoch=len(X_train) // BATCH_SIZE // hvd.size(),
        epochs=MAX_EPOCH,
        verbose=1 if hvd.rank() == 0 else 0,
        validation_data=gen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=True),
        validation_steps=len(X_test) // BATCH_SIZE // hvd.size(),  # * 3は省略
        callbacks=callbacks)

    if hvd.rank() == 0:
        model.save(str(result_dir.joinpath('model.h5')))

        score = model.evaluate_generator(
            gen.flow(X_test, y_test, batch_size=BATCH_SIZE),
            gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE))
        logger.info('Test loss:     {}'.format(score[0]))
        logger.info('Test accuracy: {}'.format(score[1]))
        logger.info('Test error:    {}'.format(1 - score[1]))


def _main():
    hvd.init()
    better_exceptions.MAX_LENGTH = 128
    base_dir = pathlib.Path(__file__).resolve().parent
    result_dir = base_dir.joinpath('results', pathlib.Path(__file__).stem)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    if hvd.rank() == 0:
        logger.addHandler(tk.log.file_handler(result_dir / 'output.log', fmt=None))
    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        _run(result_dir)


if __name__ == '__main__':
    _main()
