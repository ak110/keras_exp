"""keras-contribのDenseNetでCIFAR10を評価してみるコード。"""

import better_exceptions
import keras
import keras.preprocessing.image
import keras_contrib.applications

BATCH_SIZE = 100
MAX_EPOCH = 300


def _preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017  # scale values

    return x


def _main():
    better_exceptions.MAX_LENGTH = 128

    input_shape = (32, 32, 3)
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    model = keras_contrib.applications.DenseNet(input_shape=input_shape)
    model.compile('nadam', 'categorical_crossentropy', ['acc'])
    model.summary()

    X_train = _preprocess_input(X_train.astype('float32'))
    X_test = _preprocess_input(X_test.astype('float32'))

    score = model.evaluate(X_test, y_test)
    print('Test loss:     {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))
    print('Test error:    {}'.format(1 - score[1]))


if __name__ == '__main__':
    _main()
