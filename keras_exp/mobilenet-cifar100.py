"""MobileNetでCIFAR100してみるコード。"""
import pathlib

import better_exceptions
import keras
import keras.preprocessing.image

BATCH_SIZE = 100
MAX_EPOCH = 300


def _main():
    better_exceptions.MAX_LENGTH = 128

    input_shape = (32, 32, 3)
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    if False:
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
        validation_steps = len(X_test) // BATCH_SIZE,
        callbacks = [
            keras.callbacks.EarlyStopping('loss', patience=1),
        ])

    score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Test loss:     {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))
    print('Test error:    {}'.format(1 - score[1]))


if __name__ == '__main__':
    _main()


"""

128x128実行結果：

```
Using TensorFlow backend.
2017-09-16 20:21:17.988560: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The T
ensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-16 20:21:17.988868: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The T
ensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-16 20:21:18.327720: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:955] F
ound device 0 with properties:
name: GeForce GTX 1060 6GB
major: 6 minor: 1 memoryClockRate (GHz) 1.759
pciBusID 0000:05:00.0
Total memory: 6.00GiB
Free memory: 5.01GiB
2017-09-16 20:21:18.328075: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:976] D
MA: 0
2017-09-16 20:21:18.328490: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:986] 0
:   Y
2017-09-16 20:21:18.328905: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045]
Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:05:00.0)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 128, 128, 3)       0
_________________________________________________________________
conv1 (Conv2D)               (None, 64, 64, 32)        864
_________________________________________________________________
conv1_bn (BatchNormalization (None, 64, 64, 32)        128
_________________________________________________________________
conv1_relu (Activation)      (None, 64, 64, 32)        0
_________________________________________________________________
conv_dw_1 (DepthwiseConv2D)  (None, 64, 64, 32)        288
_________________________________________________________________
conv_dw_1_bn (BatchNormaliza (None, 64, 64, 32)        128
_________________________________________________________________
conv_dw_1_relu (Activation)  (None, 64, 64, 32)        0
_________________________________________________________________
conv_pw_1 (Conv2D)           (None, 64, 64, 64)        2048
_________________________________________________________________
conv_pw_1_bn (BatchNormaliza (None, 64, 64, 64)        256
_________________________________________________________________
conv_pw_1_relu (Activation)  (None, 64, 64, 64)        0
_________________________________________________________________
conv_dw_2 (DepthwiseConv2D)  (None, 32, 32, 64)        576
_________________________________________________________________
conv_dw_2_bn (BatchNormaliza (None, 32, 32, 64)        256
_________________________________________________________________
conv_dw_2_relu (Activation)  (None, 32, 32, 64)        0
_________________________________________________________________
conv_pw_2 (Conv2D)           (None, 32, 32, 128)       8192
_________________________________________________________________
conv_pw_2_bn (BatchNormaliza (None, 32, 32, 128)       512
_________________________________________________________________
conv_pw_2_relu (Activation)  (None, 32, 32, 128)       0
_________________________________________________________________
conv_dw_3 (DepthwiseConv2D)  (None, 32, 32, 128)       1152
_________________________________________________________________
conv_dw_3_bn (BatchNormaliza (None, 32, 32, 128)       512
_________________________________________________________________
conv_dw_3_relu (Activation)  (None, 32, 32, 128)       0
_________________________________________________________________
conv_pw_3 (Conv2D)           (None, 32, 32, 128)       16384
_________________________________________________________________
conv_pw_3_bn (BatchNormaliza (None, 32, 32, 128)       512
_________________________________________________________________
conv_pw_3_relu (Activation)  (None, 32, 32, 128)       0
_________________________________________________________________
conv_dw_4 (DepthwiseConv2D)  (None, 16, 16, 128)       1152
_________________________________________________________________
conv_dw_4_bn (BatchNormaliza (None, 16, 16, 128)       512
_________________________________________________________________
conv_dw_4_relu (Activation)  (None, 16, 16, 128)       0
_________________________________________________________________
conv_pw_4 (Conv2D)           (None, 16, 16, 256)       32768
_________________________________________________________________
conv_pw_4_bn (BatchNormaliza (None, 16, 16, 256)       1024
_________________________________________________________________
conv_pw_4_relu (Activation)  (None, 16, 16, 256)       0
_________________________________________________________________
conv_dw_5 (DepthwiseConv2D)  (None, 16, 16, 256)       2304
_________________________________________________________________
conv_dw_5_bn (BatchNormaliza (None, 16, 16, 256)       1024
_________________________________________________________________
conv_dw_5_relu (Activation)  (None, 16, 16, 256)       0
_________________________________________________________________
conv_pw_5 (Conv2D)           (None, 16, 16, 256)       65536
_________________________________________________________________
conv_pw_5_bn (BatchNormaliza (None, 16, 16, 256)       1024
_________________________________________________________________
conv_pw_5_relu (Activation)  (None, 16, 16, 256)       0
_________________________________________________________________
conv_dw_6 (DepthwiseConv2D)  (None, 8, 8, 256)         2304
_________________________________________________________________
conv_dw_6_bn (BatchNormaliza (None, 8, 8, 256)         1024
_________________________________________________________________
conv_dw_6_relu (Activation)  (None, 8, 8, 256)         0
_________________________________________________________________
conv_pw_6 (Conv2D)           (None, 8, 8, 512)         131072
_________________________________________________________________
conv_pw_6_bn (BatchNormaliza (None, 8, 8, 512)         2048
_________________________________________________________________
conv_pw_6_relu (Activation)  (None, 8, 8, 512)         0
_________________________________________________________________
conv_dw_7 (DepthwiseConv2D)  (None, 8, 8, 512)         4608
_________________________________________________________________
conv_dw_7_bn (BatchNormaliza (None, 8, 8, 512)         2048
_________________________________________________________________
conv_dw_7_relu (Activation)  (None, 8, 8, 512)         0
_________________________________________________________________
conv_pw_7 (Conv2D)           (None, 8, 8, 512)         262144
_________________________________________________________________
conv_pw_7_bn (BatchNormaliza (None, 8, 8, 512)         2048
_________________________________________________________________
conv_pw_7_relu (Activation)  (None, 8, 8, 512)         0
_________________________________________________________________
conv_dw_8 (DepthwiseConv2D)  (None, 8, 8, 512)         4608
_________________________________________________________________
conv_dw_8_bn (BatchNormaliza (None, 8, 8, 512)         2048
_________________________________________________________________
conv_dw_8_relu (Activation)  (None, 8, 8, 512)         0
_________________________________________________________________
conv_pw_8 (Conv2D)           (None, 8, 8, 512)         262144
_________________________________________________________________
conv_pw_8_bn (BatchNormaliza (None, 8, 8, 512)         2048
_________________________________________________________________
conv_pw_8_relu (Activation)  (None, 8, 8, 512)         0
_________________________________________________________________
conv_dw_9 (DepthwiseConv2D)  (None, 8, 8, 512)         4608
_________________________________________________________________
conv_dw_9_bn (BatchNormaliza (None, 8, 8, 512)         2048
_________________________________________________________________
conv_dw_9_relu (Activation)  (None, 8, 8, 512)         0
_________________________________________________________________
conv_pw_9 (Conv2D)           (None, 8, 8, 512)         262144
_________________________________________________________________
conv_pw_9_bn (BatchNormaliza (None, 8, 8, 512)         2048
_________________________________________________________________
conv_pw_9_relu (Activation)  (None, 8, 8, 512)         0
_________________________________________________________________
conv_dw_10 (DepthwiseConv2D) (None, 8, 8, 512)         4608
_________________________________________________________________
conv_dw_10_bn (BatchNormaliz (None, 8, 8, 512)         2048
_________________________________________________________________
conv_dw_10_relu (Activation) (None, 8, 8, 512)         0
_________________________________________________________________
conv_pw_10 (Conv2D)          (None, 8, 8, 512)         262144
_________________________________________________________________
conv_pw_10_bn (BatchNormaliz (None, 8, 8, 512)         2048
_________________________________________________________________
conv_pw_10_relu (Activation) (None, 8, 8, 512)         0
_________________________________________________________________
conv_dw_11 (DepthwiseConv2D) (None, 8, 8, 512)         4608
_________________________________________________________________
conv_dw_11_bn (BatchNormaliz (None, 8, 8, 512)         2048
_________________________________________________________________
conv_dw_11_relu (Activation) (None, 8, 8, 512)         0
_________________________________________________________________
conv_pw_11 (Conv2D)          (None, 8, 8, 512)         262144
_________________________________________________________________
conv_pw_11_bn (BatchNormaliz (None, 8, 8, 512)         2048
_________________________________________________________________
conv_pw_11_relu (Activation) (None, 8, 8, 512)         0
_________________________________________________________________
conv_dw_12 (DepthwiseConv2D) (None, 4, 4, 512)         4608
_________________________________________________________________
conv_dw_12_bn (BatchNormaliz (None, 4, 4, 512)         2048
_________________________________________________________________
conv_dw_12_relu (Activation) (None, 4, 4, 512)         0
_________________________________________________________________
conv_pw_12 (Conv2D)          (None, 4, 4, 1024)        524288
_________________________________________________________________
conv_pw_12_bn (BatchNormaliz (None, 4, 4, 1024)        4096
_________________________________________________________________
conv_pw_12_relu (Activation) (None, 4, 4, 1024)        0
_________________________________________________________________
conv_dw_13 (DepthwiseConv2D) (None, 4, 4, 1024)        9216
_________________________________________________________________
conv_dw_13_bn (BatchNormaliz (None, 4, 4, 1024)        4096
_________________________________________________________________
conv_dw_13_relu (Activation) (None, 4, 4, 1024)        0
_________________________________________________________________
conv_pw_13 (Conv2D)          (None, 4, 4, 1024)        1048576
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 4, 4, 1024)        4096
_________________________________________________________________
conv_pw_13_relu (Activation) (None, 4, 4, 1024)        0
_________________________________________________________________
global_max_pooling2d_1 (Glob (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               102500
=================================================================
Total params: 3,331,364
Trainable params: 3,309,476
Non-trainable params: 21,888
_________________________________________________________________
Epoch 1/300
500/500 [==============================] - 243s - loss: 3.9806 - acc: 0.3241 - val_loss: 2.0032 - val_acc: 0.4673
Epoch 2/300
500/500 [==============================] - 238s - loss: 1.5644 - acc: 0.5641 - val_loss: 1.3342 - val_acc: 0.6206
Epoch 3/300
500/500 [==============================] - 243s - loss: 1.2258 - acc: 0.6461 - val_loss: 1.2247 - val_acc: 0.6555
Epoch 4/300
500/500 [==============================] - 241s - loss: 1.0403 - acc: 0.6940 - val_loss: 1.1057 - val_acc: 0.6829
Epoch 5/300
500/500 [==============================] - 241s - loss: 0.9085 - acc: 0.7280 - val_loss: 1.0850 - val_acc: 0.6883
Epoch 6/300
500/500 [==============================] - 241s - loss: 0.7979 - acc: 0.7586 - val_loss: 1.0210 - val_acc: 0.7140
Epoch 7/300
500/500 [==============================] - 240s - loss: 0.7192 - acc: 0.7791 - val_loss: 1.0404 - val_acc: 0.7141
Epoch 8/300
500/500 [==============================] - 240s - loss: 0.6460 - acc: 0.8002 - val_loss: 1.0981 - val_acc: 0.7058
Epoch 9/300
500/500 [==============================] - 238s - loss: 0.5891 - acc: 0.8154 - val_loss: 1.0271 - val_acc: 0.7230
Epoch 10/300
500/500 [==============================] - 237s - loss: 0.5322 - acc: 0.8320 - val_loss: 1.0581 - val_acc: 0.7234
Epoch 11/300
500/500 [==============================] - 228s - loss: 0.4896 - acc: 0.8463 - val_loss: 1.0110 - val_acc: 0.7321
Epoch 12/300
500/500 [==============================] - 228s - loss: 0.4473 - acc: 0.8563 - val_loss: 1.0580 - val_acc: 0.7309
Epoch 13/300
500/500 [==============================] - 231s - loss: 0.4171 - acc: 0.8657 - val_loss: 0.9966 - val_acc: 0.7464
Epoch 14/300
500/500 [==============================] - 227s - loss: 0.3719 - acc: 0.8808 - val_loss: 1.0707 - val_acc: 0.7341
Epoch 15/300
500/500 [==============================] - 222s - loss: 0.3463 - acc: 0.8877 - val_loss: 1.0819 - val_acc: 0.7387
Epoch 16/300
500/500 [==============================] - 225s - loss: 0.3221 - acc: 0.8952 - val_loss: 1.0381 - val_acc: 0.7489
Epoch 17/300
500/500 [==============================] - 220s - loss: 0.2936 - acc: 0.9041 - val_loss: 1.0412 - val_acc: 0.7530
Epoch 18/300
500/500 [==============================] - 218s - loss: 0.2709 - acc: 0.9109 - val_loss: 1.1061 - val_acc: 0.7486
Epoch 19/300
500/500 [==============================] - 232s - loss: 0.2521 - acc: 0.9167 - val_loss: 1.1102 - val_acc: 0.7481
Epoch 20/300
500/500 [==============================] - 233s - loss: 0.2414 - acc: 0.9204 - val_loss: 1.1561 - val_acc: 0.7470
Epoch 21/300
500/500 [==============================] - 235s - loss: 0.2194 - acc: 0.9271 - val_loss: 1.1217 - val_acc: 0.7466
Epoch 22/300
500/500 [==============================] - 240s - loss: 0.2108 - acc: 0.9305 - val_loss: 1.1160 - val_acc: 0.7525
Epoch 23/300
500/500 [==============================] - 237s - loss: 0.1962 - acc: 0.9351 - val_loss: 1.1294 - val_acc: 0.7557
Epoch 24/300
500/500 [==============================] - 240s - loss: 0.1832 - acc: 0.9393 - val_loss: 1.2141 - val_acc: 0.7434
Epoch 25/300
500/500 [==============================] - 238s - loss: 0.1702 - acc: 0.9430 - val_loss: 1.1990 - val_acc: 0.7509
Epoch 26/300
500/500 [==============================] - 239s - loss: 0.1665 - acc: 0.9455 - val_loss: 1.2066 - val_acc: 0.7472
Epoch 27/300
500/500 [==============================] - 237s - loss: 0.1563 - acc: 0.9482 - val_loss: 1.1644 - val_acc: 0.7548
Epoch 28/300
500/500 [==============================] - 236s - loss: 0.1453 - acc: 0.9518 - val_loss: 1.1522 - val_acc: 0.7624
Epoch 29/300
500/500 [==============================] - 239s - loss: 0.1338 - acc: 0.9553 - val_loss: 1.1539 - val_acc: 0.7601
Epoch 30/300
500/500 [==============================] - 240s - loss: 0.1307 - acc: 0.9562 - val_loss: 1.1798 - val_acc: 0.7593
Epoch 31/300
500/500 [==============================] - 241s - loss: 0.1241 - acc: 0.9584 - val_loss: 1.1545 - val_acc: 0.7620
Epoch 32/300
500/500 [==============================] - 240s - loss: 0.1258 - acc: 0.9588 - val_loss: 1.1376 - val_acc: 0.7611
Epoch 33/300
500/500 [==============================] - 240s - loss: 0.1182 - acc: 0.9605 - val_loss: 1.2282 - val_acc: 0.7602
Epoch 34/300
500/500 [==============================] - 241s - loss: 0.1108 - acc: 0.9629 - val_loss: 1.2018 - val_acc: 0.7617
Epoch 35/300
500/500 [==============================] - 241s - loss: 0.1048 - acc: 0.9653 - val_loss: 1.2237 - val_acc: 0.7665
Epoch 36/300
500/500 [==============================] - 241s - loss: 0.1005 - acc: 0.9663 - val_loss: 1.2281 - val_acc: 0.7625
Epoch 37/300
500/500 [==============================] - 241s - loss: 0.1011 - acc: 0.9666 - val_loss: 1.2332 - val_acc: 0.7606
Epoch 38/300
500/500 [==============================] - 239s - loss: 0.0964 - acc: 0.9673 - val_loss: 1.2522 - val_acc: 0.7645
Epoch 39/300
500/500 [==============================] - 238s - loss: 0.0942 - acc: 0.9682 - val_loss: 1.2227 - val_acc: 0.7654
Epoch 40/300
500/500 [==============================] - 235s - loss: 0.0843 - acc: 0.9720 - val_loss: 1.2213 - val_acc: 0.7696
Epoch 41/300
500/500 [==============================] - 233s - loss: 0.0815 - acc: 0.9738 - val_loss: 1.2658 - val_acc: 0.7643
Epoch 42/300
500/500 [==============================] - 239s - loss: 0.0812 - acc: 0.9730 - val_loss: 1.2795 - val_acc: 0.7689
Epoch 43/300
500/500 [==============================] - 241s - loss: 0.0770 - acc: 0.9738 - val_loss: 1.2858 - val_acc: 0.7661
Epoch 44/300
500/500 [==============================] - 238s - loss: 0.0752 - acc: 0.9755 - val_loss: 1.2468 - val_acc: 0.7661
Epoch 45/300
500/500 [==============================] - 238s - loss: 0.0688 - acc: 0.9770 - val_loss: 1.2634 - val_acc: 0.7713
Epoch 46/300
500/500 [==============================] - 239s - loss: 0.0689 - acc: 0.9775 - val_loss: 1.2238 - val_acc: 0.7769
Epoch 47/300
500/500 [==============================] - 237s - loss: 0.0721 - acc: 0.9760 - val_loss: 1.3016 - val_acc: 0.7680
10000/10000 [==============================] - 13s
Test loss:     1.3016317409276963
Test accuracy: 0.7679999804496765
Test error:    0.23200001955032346
```
"""
