"""MobileNetでCIFAR100してみるコード。"""

import better_exceptions
import keras
import keras.preprocessing.image

BATCH_SIZE = 100
MAX_EPOCH = 300
_MOBILENET_128 = False


def _main():
    better_exceptions.MAX_LENGTH = 128

    input_shape = (32, 32, 3)
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    if _MOBILENET_128:
        outputs = inputs = keras.layers.Input(input_shape)
        outputs = keras.layers.UpSampling2D((4, 4))(outputs)
        base_model = keras.applications.mobilenet.MobileNet(
            input_tensor=outputs, input_shape=(128, 128, 3), include_top=False, pooling='max')
    else:
        outputs = inputs = keras.layers.Input(input_shape)
        weights = keras.applications.mobilenet.MobileNet(input_shape=(128, 128, 3), include_top=False, pooling='max')
        base_model = keras.applications.mobilenet.MobileNet(
            input_tensor=outputs, input_shape=input_shape, include_top=False, pooling='max', weights=None)
        for layer in weights.layers:
            try:
                l = base_model.get_layer(name=layer.name)
                l.set_weights(layer.get_weights())
            except ValueError as e:
                print(e)

    outputs = keras.layers.Dense(nb_classes, activation='softmax')(base_model.outputs[0])
    model = keras.models.Model(inputs, outputs)
    model.compile(keras.optimizers.SGD(1e-2, momentum=0.9, nesterov=True), 'categorical_crossentropy', ['acc'])
    model.summary()

    X_train = keras.applications.mobilenet.preprocess_input(X_train.astype('float32'))
    X_test = keras.applications.mobilenet.preprocess_input(X_test.astype('float32'))

    gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=5 / 32,
        height_shift_range=5 / 32,
        horizontal_flip=True)

    model.fit_generator(
        gen.flow(X_train, y_train, BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=MAX_EPOCH,
        validation_data=(X_test, y_test),
        validation_steps=len(X_test) // BATCH_SIZE,
        callbacks=[
            keras.callbacks.EarlyStopping('loss', patience=1),
        ])

    score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Test loss:     {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))
    print('Test error:    {}'.format(1 - score[1]))


if __name__ == '__main__':
    _main()
