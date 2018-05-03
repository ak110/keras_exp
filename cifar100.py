"""CIFAR."""
import pathlib

import numpy as np

import pytoolkit as tk

BATCH_SIZE = 64
MAX_EPOCH = 300


def _main():
    base_dir = pathlib.Path(__file__).resolve().parent
    result_dir = base_dir / 'results' / pathlib.Path(__file__).stem
    result_dir.mkdir(parents=True, exist_ok=True)
    with tk.dl.session(use_horovod=True):
        tk.log.init(result_dir / 'output.log', file_fmt=None)
        _run(result_dir)


@tk.log.trace()
def _run(result_dir: pathlib.Path):
    import keras
    logger = tk.log.get(__name__)

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    with tk.log.trace_scope('create network'):
        builder = tk.dl.layers.Builder()
        x = inp = keras.layers.Input(input_shape)
        for block, filters in enumerate([128, 256, 512]):
            name = f'stage{block + 1}_block'
            strides = 1 if block == 0 else 2
            x = builder.conv2d(filters, strides=strides, use_act=False, name=f'{name}_start')(x)
            for res in range(4):
                sc = x
                x = builder.conv2d(filters // 4, name=f'{name}_r{res}_c1')(x)
                for d in range(8):
                    t = builder.conv2d(filters // 8, name=f'{name}_r{res}_d{d}')(x)
                    x = keras.layers.concatenate([x, t], name=f'{name}_r{res}_d{d}_cat')
                x = builder.conv2d(filters, 1, use_act=False, name=f'{name}_r{res}_c2')(x)
                x = keras.layers.add([sc, x], name=f'{name}_r{res}_add')
            x = builder.bn_act(name=f'{name}')(x)
        x = keras.layers.Dropout(0.5, name='dropout')(x)
        x = keras.layers.GlobalAveragePooling2D(name='pooling')(x)
        x = builder.dense(num_classes, activation='softmax', name='predictions')(x)
        model = keras.models.Model(inputs=inp, outputs=x)

    gen = tk.image.ImageDataGenerator()
    gen.add(tk.generator.ProcessOutput(tk.ml.to_categorical(num_classes), batch_axis=True))
    gen.add(tk.image.Mixup(probability=1, num_classes=num_classes))
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize(input_shape[:2]))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomSaturation(probability=0.5),
        tk.image.RandomBrightness(probability=0.5),
        tk.image.RandomContrast(probability=0.5),
        tk.image.RandomHue(probability=0.5),
    ])
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.generator.ProcessInput(tk.image.preprocess_input_abs1))

    model = tk.dl.models.Model(model, gen, BATCH_SIZE)
    model.compile(sgd_lr=0.5 / 256, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    callbacks = []
    callbacks.append(tk.dl.callbacks.learning_rate())
    callbacks.extend(model.horovod_callbacks())
    callbacks.append(tk.dl.callbacks.tsv_logger(result_dir / 'history.tsv'))
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=MAX_EPOCH, callbacks=callbacks)
    model.save(result_dir / 'model.h5')
    if tk.dl.hvd.is_master():
        loss, acc = model.evaluate(X_test, y_test)
        logger.info(f'Test loss:     {loss:.3f}')
        logger.info(f'Test accuracy: {acc:.3f}')
        logger.info(f'Test error:    {1 - acc:.3f}')


if __name__ == '__main__':
    _main()
