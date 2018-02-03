"""KerasのMNISTサンプルの魔改造品。"""
import keras

batch_size = 128
epochs = 12

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
num_classes = 10
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def _conv(x, filters, kernel_size=(3, 3)):
    x = keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('elu')(x)
    return x

x = inp = keras.layers.Input((28, 28, 1))
x = _conv(x, 32)
x = _conv(x, 32)
x = keras.layers.MaxPooling2D()(x)
x = _conv(x, 32)
x = _conv(x, 32)
x = keras.layers.MaxPooling2D()(x)
x = _conv(x, 32)
x = _conv(x, 32)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(num_classes, activation='softmax')(x)
model = keras.models.Model(inp, x)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Nadam(),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
