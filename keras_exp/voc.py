"""ObjectDetectionを適当にやってみるコード。

-train: VOC2007 trainval + VOC2012 trainval
-test:  VOC2007 test
"""
import os
import pathlib
import time

import better_exceptions
import numpy as np
from tqdm import tqdm

import pytoolkit as tk
from _od import ObjectDetector

_DEBUG = True

_BATCH_SIZE = 16
_MAX_EPOCH = 16 if _DEBUG else 300
_BASE_LR = 1e-1  # 1e-3

_CHECK_PRIOR_BOX = True  # 重いので必要なときだけ
_EVALUATE = True  # 重いので必要なときだけ

_CLASS_NAMES = ['bg'] + tk.ml.VOC_CLASS_NAMES
_CLASS_NAME_TO_ID = {n: i for i, n in enumerate(_CLASS_NAMES)}


class Generator(tk.image.ImageDataGenerator):
    """データの読み込みを行うGenerator。"""

    def __init__(self, image_size, od):
        super().__init__(image_size)
        self.od = od

    def _prepare(self, X, y=None, weights=None, parallel=None, data_augmentation=False, rand=None):
        """画像の読み込みとDataAugmentation。"""
        X, y, weights = super()._prepare(X, y, weights, parallel, data_augmentation, rand)
        if y is not None:
            # 学習用に変換
            y = self.od.encode_truth(y)
        return X, y, weights

    def _transform(self, rgb, rand):
        """変形を伴うAugmentation。"""
        # とりあえず殺しておく
        return rgb


def _run(logger, result_dir: pathlib.Path, data_dir: pathlib.Path):
    """実行。"""
    # データの読み込み
    (X_train, y_train), (X_test, y_test) = load_data(data_dir)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # 試しに回答を出力してみる。
    plot_truth(X_test[:_BATCH_SIZE], y_test[:_BATCH_SIZE], result_dir.joinpath('___ground_truth'))
    # for i, y in enumerate(y_test[:3]):
    #     logger.debug('y_test[%d]: %s', i, str(y))

    # 訓練データからパラメータを適当に決める。
    # gridに配置したときのIOUを直接最適化するのは難しそうなので、
    # とりあえず大雑把にKMeansでクラスタ化したりなど。
    od = ObjectDetector.create(len(_CLASS_NAMES), y_train)
    logger.debug('mean objects / image = %f', od.mean_objets)
    logger.debug('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.debug('prior box aspect ratios = %s', str(od.aspect_ratios))
    logger.debug('prior box count = %d', len(od.pb_locs))
    logger.debug('prior box sizes = %s', str(np.unique(od.pb_sizes)))

    # prior boxのカバー度合いのチェック
    if _DEBUG or _CHECK_PRIOR_BOX:
        od.check_prior_boxes(logger, result_dir, y_test, _CLASS_NAMES)

    import keras
    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        model = od.create_model()
        model.summary(print_fn=logger.debug)
        tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))
        keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)

        # 事前学習の読み込み
        # model.load_weights(str(result_dir.parent.joinpath('voc_pre', 'model.h5')), by_name=True)

        gen = Generator(image_size=od.input_size, od=od)
        gen.add(0.5, tk.image.FlipLR())
        gen.add(0.125, tk.image.RandomBlur())
        gen.add(0.125, tk.image.RandomBlur(partial=True))
        gen.add(0.125, tk.image.RandomUnsharpMask())
        gen.add(0.125, tk.image.RandomUnsharpMask(partial=True))
        gen.add(0.125, tk.image.Sharp())
        gen.add(0.125, tk.image.Sharp(partial=True))
        gen.add(0.125, tk.image.Soft())
        gen.add(0.125, tk.image.Soft(partial=True))
        gen.add(0.125, tk.image.RandomMedian())
        gen.add(0.125, tk.image.RandomMedian(partial=True))
        gen.add(0.125, tk.image.GaussianNoise())
        gen.add(0.125, tk.image.GaussianNoise(partial=True))
        gen.add(0.125, tk.image.RandomSaturation())
        gen.add(0.125, tk.image.RandomBrightness())
        gen.add(0.125, tk.image.RandomContrast())
        gen.add(0.125, tk.image.RandomLighting())

        # lr_list = [_BASE_LR] * (_MAX_EPOCH * 6 // 9) + [_BASE_LR / 10] * (_MAX_EPOCH * 2 // 9) + [_BASE_LR / 100] * (_MAX_EPOCH * 1 // 9)

        callbacks = []
        # callbacks.append(tk.dl.my_callback_factory()(result_dir, lr_list=lr_list))
        callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=_BASE_LR, beta1=0.9990, beta2=0.9995))
        callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
        callbacks.append(keras.callbacks.ModelCheckpoint(str(result_dir.joinpath('model.best.h5')), save_best_only=True))
        # callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
        # if K.backend() == 'tensorflow':
        #     callbacks.append(keras.callbacks.TensorBoard())

        # 各epoch毎にmAPを算出して表示してみる
        if _DEBUG or _EVALUATE:
            callbacks.append(keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: evaluate(logger, od, model, gen, X_test, y_test, epoch, result_dir)
            ))

        model.fit_generator(
            gen.flow(X_train, y_train, batch_size=_BATCH_SIZE, data_augmentation=not _DEBUG, shuffle=not _DEBUG),
            steps_per_epoch=gen.steps_per_epoch(len(X_train), _BATCH_SIZE),
            epochs=_MAX_EPOCH,
            validation_data=gen.flow(X_test, y_test, batch_size=_BATCH_SIZE),
            validation_steps=gen.steps_per_epoch(len(X_test), _BATCH_SIZE),
            callbacks=callbacks)

        model.save(str(result_dir.joinpath('model.h5')))

        # 最終結果表示
        evaluate(logger, od, model, gen, X_test, y_test, None, result_dir)


def load_data(data_dir: pathlib.Path):
    """データの読み込み"""
    X_train = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')) +
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'trainval.txt'))
    )
    X_test = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'test.txt'))
    )
    if _DEBUG:
        X_test = X_test[:_BATCH_SIZE]  # 先頭バッチサイズ分だけを使用
        rep = int(np.ceil((len(X_train) / len(X_test)) ** 0.9))  # 個数を減らした分、水増しする。
        X_train = np.tile(X_test, rep)
    annotations = load_annotations(data_dir)
    y_train = np.array([annotations[x] for x in X_train])
    y_test = np.array([annotations[x] for x in X_test])
    # Xのフルパス化
    X_train = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_train])
    X_test = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_test])
    return (X_train, y_train), (X_test, y_test)


def load_annotations(data_dir: pathlib.Path) -> dict:
    """VOC2007,VOC2012のアノテーションデータの読み込み。"""
    data = {}
    for folder in ('VOC2007', 'VOC2012'):
        data.update(tk.ml.ObjectsAnnotation.load_voc(
            data_dir.joinpath('VOCdevkit', folder, 'Annotations'), _CLASS_NAME_TO_ID))
    return data


def evaluate(logger, od, model, gen, X_test, y_test, epoch, result_dir):
    """`mAP`を算出してprintする。"""
    if epoch is not None and not (epoch % 16 == 0 or epoch & (epoch - 1) == 0):
        return  # 重いので16回あたり1回だけ実施
    if epoch is not None:
        print('')

    pred_classes_list = []
    pred_locs_list = []
    steps = gen.steps_per_epoch(len(X_test), _BATCH_SIZE)
    with tqdm(total=steps, desc='evaluate', ascii=True, ncols=100) as pbar:
        for i, X_batch in enumerate(gen.flow(X_test, batch_size=_BATCH_SIZE)):
            pred = model.predict(X_batch)
            pred_classes, pred_confs, pred_locs = od.decode_predictions(pred)
            pred_classes_list += pred_classes
            pred_locs_list += pred_locs
            if i == 0:
                save_dir = result_dir.joinpath('___check')
                for j, (pcl, pcf, pl) in enumerate(zip(pred_classes, pred_confs, pred_locs)):
                    tk.ml.plot_objects(
                        X_test[j], save_dir.joinpath(pathlib.Path(X_test[j]).name + '.png'),
                        pcl, pcf, pl, _CLASS_NAMES)
            pbar.update()
            if i + 1 >= steps:
                break

    gt_classes_list = np.array([y.classes for y in y_test])
    gt_bboxes_list = np.array([y.bboxes for y in y_test])
    gt_difficults_list = np.array([y.difficults for y in y_test])
    map1 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_classes_list, pred_locs_list, use_voc2007_metric=False)
    map2 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_classes_list, pred_locs_list, use_voc2007_metric=True)

    logger.debug('mAP={:.4f} mAP(VOC2007)={:.4f}'.format(map1, map2))
    if epoch is not None:
        print('')


def plot_truth(X_test, y_test, save_dir):
    """正解データの画像化。"""
    for X, y in zip(X_test, y_test):
        tk.ml.plot_objects(
            X, save_dir.joinpath(pathlib.Path(X).name + '.png'),
            y.classes, None, y.bboxes, _CLASS_NAMES)


def plot_result(od, model, gen, X_test, save_dir):
    """結果の画像化。"""
    pred = model.predict_generator(
        gen.flow(X_test, batch_size=_BATCH_SIZE),
        gen.steps_per_epoch(len(X_test), _BATCH_SIZE),
        verbose=1)
    pred_classes_list, pred_confs_list, pred_locs_list = od.decode_predictions(pred)

    for X, pred_classes, pred_confs, pred_locs in zip(X_test, pred_classes_list, pred_confs_list, pred_locs_list):
        tk.ml.plot_objects(
            X, save_dir.joinpath(pathlib.Path(X).name + '.png'),
            pred_classes, pred_confs, pred_locs, _CLASS_NAMES)


def _main():
    import matplotlib as mpl
    mpl.use('Agg')

    better_exceptions.MAX_LENGTH = 128

    base_dir = pathlib.Path(os.path.realpath(__file__)).parent.parent
    os.chdir(str(base_dir))
    np.random.seed(1337)  # for reproducibility

    result_dir = base_dir.joinpath('results', pathlib.Path(__file__).stem)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = tk.create_tee_logger(result_dir.joinpath('output.log'))

    start_time = time.time()
    _run(logger, result_dir, base_dir.joinpath('data'))
    elapsed_time = time.time() - start_time

    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


if __name__ == '__main__':
    _main()
