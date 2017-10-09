"""初期状態での平均・分散の変化を調査してみるコード。

BNは統計がたぶんうまく動かない。
"""

import numpy as np

import pytoolkit as tk


def _main():
    input_shape = (32, 32, 64)
    X = np.random.normal(0, 1, size=(1000,) + input_shape)
    # X = np.random.uniform(-1, 1, size=(1000,) + input_shape)
    # X = np.random.uniform(0, 1, size=(1000,) + input_shape)

    with tk.dl.session():
        import keras

        def _conv_gl(x):
            x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
            return x

        def _conv_he(x):
            x = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
            return x

        def _sq_gl(x):
            x = keras.layers.Conv2D(8, (1, 1), padding='same')(x)
            return x

        def _sq_he(x):
            x = keras.layers.Conv2D(8, (1, 1), padding='same', kernel_initializer='he_uniform')(x)
            return x

        def _exsq_gl(x):
            x = keras.layers.Conv2D(512, (3, 3), padding='same')(x)
            x = keras.layers.Conv2D(64, (1, 1), padding='same')(x)
            return x

        def _exsq_he(x):
            x = keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
            x = keras.layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(x)
            return x

        def _bn(x):
            x = keras.layers.BatchNormalization()(x)
            return x

        def _ap(x):
            x = keras.layers.AveragePooling2D()(x)
            return x

        def _mp(x):
            x = keras.layers.MaxPooling2D()(x)
            return x

        def _conv_relu(x):
            x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            return x

        def _conv_elu(x):
            x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
            x = keras.layers.ELU()(x)
            return x

        def _conv_elu175(x):
            x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
            x = keras.layers.ELU(alpha=1.75)(x)
            return x

        def _conv_selu(x):
            x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
            x = keras.layers.ELU(alpha=1.6732632423543772848170429916717)(x)
            x = keras.layers.Lambda(lambda x: x * 1.0507009873554804934193349852946)(x)
            return x

        def _l2norm(x):
            x = tk.dl.l2normalization_layer()(np.sqrt(64))(x)
            return x

        def _conv_elu_l2norm(x):
            x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
            x = keras.layers.ELU(alpha=1.75)(x)
            x = tk.dl.l2normalization_layer()(np.sqrt(128))(x)
            return x

        checks = [
            _conv_gl,
            _conv_he,
            _sq_gl,
            _sq_he,
            _exsq_gl,
            _exsq_he,
            _bn,
            _ap,
            _mp,
            _conv_relu,
            _conv_elu,
            _conv_elu175,
            _conv_selu,
            _l2norm,
            _conv_elu_l2norm,
        ]
        for c in checks:
            inputs = keras.layers.Input(input_shape)
            x = c(inputs)
            model = keras.models.Model(inputs, x)
            model.compile('nadam', 'categorical_crossentropy')
            pred = model.predict(X)
            print('{:20s}: mean={:6.3f} std={:6.3f}'.format(c.__name__, np.mean(pred), np.std(pred)))


if __name__ == '__main__':
    _main()
