"""CIFAR."""
import pathlib

import horovod.keras as hvd
import numpy as np

import pytoolkit as tk

BATCH_SIZE = 64
MAX_EPOCH = 300


def _main():
    base_dir = pathlib.Path(__file__).resolve().parent
    result_dir = base_dir / 'results' / pathlib.Path(__file__).stem
    result_dir.mkdir(parents=True, exist_ok=True)
    with tk.dl.session(use_horovod=True):
        tk.log.init(result_dir / 'output.log' if hvd.rank() == 0 else None, file_fmt=None)
        _run(result_dir)


@tk.log.trace()
def _run(result_dir: pathlib.Path):
    import keras
    logger = tk.log.get(__name__)

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    with tk.log.trace_scope('create network'):
        builder = tk.dl.layers.Builder()
        x = inp = keras.layers.Input(input_shape)
        for block, (filters, res_count) in enumerate([(128, 4), (256, 12), (512, 4)]):
            name = f'stage{block + 1}_block'
            strides = (1, 1) if block == 0 else (2, 2)
            x = builder.conv2d(filters, (3, 3), strides=strides, use_act=False, name=f'{name}_start')(x)
            for res in range(res_count):
                sc = x
                x = builder.conv2d(filters, (3, 3), name=f'{name}_r{res}c1')(x)
                x = keras.layers.Dropout(0.25)(x)
                x = builder.conv2d(filters, (3, 3), use_act=False, name=f'{name}_r{res}c2')(x)
                x = keras.layers.add([sc, x])
            x = builder.bn()(x)
            x = builder.act()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = builder.dense(num_classes, activation='softmax',
                          kernel_initializer='zeros', name='predictions')(x)
        model = keras.models.Model(inputs=inp, outputs=x)

    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.ProcessOutput(tk.ml.to_categorical(num_classes), batch_axis=True))
    gen.add(tk.image.Mixup(probability=1, num_classes=num_classes))
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize(input_shape[:2]))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomColorAugmentors(probability=0.5))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.ProcessInput(tk.image.preprocess_input_abs1))

    model = tk.dl.models.Model(model, gen, BATCH_SIZE, use_horovod=True)
    model.compile(sgd_lr=0.5 / 256, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    callbacks = []
    callbacks.append(tk.dl.callbacks.learning_rate())
    callbacks.extend(model.horovod_callbacks())
    if hvd.rank() == 0:
        callbacks.append(tk.dl.callbacks.tsv_logger(result_dir / 'history.tsv'))
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=MAX_EPOCH, callbacks=callbacks)
    model.save(result_dir / 'model.h5')
    if hvd.rank() == 0:
        loss, acc = model.evaluate(X_test, y_test)
        logger.info(f'Test loss:     {loss}')
        logger.info(f'Test accuracy: {acc}')
        logger.info(f'Test error:    {1 - acc}')


if __name__ == '__main__':
    _main()
