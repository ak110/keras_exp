"""ObjectDetectionを適当にやってみるコード。

-train: VOC2007 trainval + VOC2012 trainval
-test:  VOC2007 test
"""
import collections
import pathlib

import numpy as np
import sklearn.cluster
import sklearn.metrics
from tqdm import tqdm

import pytoolkit as tk

_BATCH_SIZE = 16
_MAX_EPOCH = 300
_CHECK_PRIOR_BOX = True
_LOSS_FUNC = 'focal'  # 'ce', 'bce', 'focal'
_ASPECT_RATIO_PATTERNS = 5
_PB_SIZE_PATTERNS = 3

_CLASS_NAMES = [
    'bg',  # 背景
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]
_CLASS_NAME_TO_ID = {n: i for i, n in enumerate(_CLASS_NAMES)}

# ground truthなデータを保持する用。
# classes、bboxes、difficultsはそれぞれbounding boxの数分の配列。
ObjectDetectionAnnotation = collections.namedtuple('ObjectDetectionAnnotation', 'width,height,classes,bboxes,difficults')


class PriorBoxes(object):
    """候補として最初に準備するboxの集合。"""

    # 取り出すfeature mapのサイズ
    FM_COUNTS = (48, 24, 12, 6, 3)

    def __init__(self, nb_classes, pb_size_ratios=(1.5, 2.0, 2.5), aspect_ratios=(1, 2, 1 / 2, 3, 1 / 3)):
        self.nb_classes = nb_classes
        self.input_size = (773, 773)
        # 1 / fm_countに対する、prior boxの基準サイズの割合。3なら3倍の大きさのものを用意。
        self.pb_size_ratios = pb_size_ratios
        # アスペクト比のリスト
        self.aspect_ratios = aspect_ratios
        # 基準となるbox(prior box)の座標を事前に算出しておく
        # pb_locs: shape=(box数, 4)で座標
        # pb_indices: shape=(box数,)で、何種類目のprior boxか (集計用)
        self.pb_locs, self.pb_indices, self.config = self._create_prior_boxes(self.pb_size_ratios, self.aspect_ratios)
        # pb_sizes: shape=(box数,)でpb_size
        self.pb_sizes = np.array(sum([[c['pb_size']] * c['box_count'] for c in self.config], []))
        assert len(self.pb_sizes) == len(self.pb_locs)

    @staticmethod
    def _create_prior_boxes(pb_size_ratios, aspect_ratios):
        """基準となるbox(prior box)の座標(x1, y1, x2, y2)の並びを返す。"""

        pb_locs = []
        pb_indices = []
        config = []
        for fm_count in PriorBoxes.FM_COUNTS:
            for pb_size_ratio in pb_size_ratios:
                # 敷き詰める間隔
                tile_size = 1.0 / fm_count
                # 敷き詰めたときの中央の位置のリスト
                lin = np.linspace(0.5 * tile_size, 1 - 0.5 * tile_size, fm_count, dtype=np.float32)
                # 縦横に敷き詰め
                centers_x, centers_y = np.meshgrid(lin, lin)
                centers_x = centers_x.reshape(-1, 1)
                centers_y = centers_y.reshape(-1, 1)
                prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
                # x, y → x1, y1, x2, y2
                prior_boxes = np.tile(prior_boxes, (len(aspect_ratios), 1, 2))

                # prior boxのサイズ
                pb_size = tile_size * pb_size_ratio
                # (x1, y1, x2, y2)の位置を調整。縦横半分ずつ動かす。
                half_box_widths = np.array([[0.5 * pb_size * np.sqrt(ar)] for ar in aspect_ratios])
                half_box_heights = np.array([[0.5 * pb_size / np.sqrt(ar)] for ar in aspect_ratios])
                prior_boxes[:, :, 0] -= half_box_widths
                prior_boxes[:, :, 1] -= half_box_heights
                prior_boxes[:, :, 2] += half_box_widths
                prior_boxes[:, :, 3] += half_box_heights
                # はみ出ているのはclipしておく
                prior_boxes = np.clip(prior_boxes, 0, 1)

                # (アスペクト比, タイル, 4) → (アスペクト比×タイル, 4)
                prior_boxes = prior_boxes.reshape(-1, 4)

                pb_locs.append(prior_boxes)
                pb_indices.extend([len(config)] * len(prior_boxes))

                config.append({'fm_count': fm_count, 'pb_size': pb_size, 'box_count': len(prior_boxes)})  # box個数を記録しとく。

        return np.concatenate(pb_locs), np.array(pb_indices), config

    def check_prior_boxes(self, logger, result_dir, annotations):
        """データに対して`self.pb_locs`がどれくらいマッチしてるか調べる。

        本当はここも学習したほうが良さそうだけどとりあえず。 (cf. YOLO9000)
        """
        y_true = []
        y_pred = []
        match_counts = [0 for _ in range(len(self.config))]
        unrec_widths = []
        unrec_heights = []
        unrec_ars = []
        for _, _, classes, bboxes, _ in tqdm(annotations.values(), desc='check_prior_boxes', ascii=True, ncols=100):
            iou = tk.ml.compute_iou(self.pb_locs, bboxes)
            # クラスごとに再現率を求める
            for i, class_id in enumerate(classes):
                m = iou[:, i] >= 0.5
                success = m.any()
                y_true.append(class_id)
                y_pred.append(class_id if success else 0)  # IOUが0.5以上のboxが存在すれば一致扱いとする
                if success:
                    match_counts[self.pb_indices[iou[:, i].argmax()]] += 1
            # 再現(iou >= 0.5)しなかったboxの情報を集める
            for bbox in bboxes[iou.max(axis=0) < 0.5]:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                unrec_widths.append(w)
                unrec_heights.append(h)
                unrec_ars.append(w / h)
        # 再現率などの表示
        y_true.append(0)  # 警告よけにbgも1個入れておく
        y_pred.append(0)  # 警告よけにbgも1個入れておく
        total_gt_boxes = sum([len(bboxes) for _, _, _, bboxes, _ in annotations.values()])
        cr = sklearn.metrics.classification_report(y_true, y_pred, target_names=_CLASS_NAMES)
        logger.debug('prior boxs = %d', len(self.pb_locs))
        logger.debug(cr)
        logger.debug('match counts:')
        for i, c in enumerate(match_counts):
            logger.debug('  prior_boxes_%d = %d (%.02f%%)',
                         self.config[i]['fm_count'], c,
                         100 * c / self.config[i]['box_count'] / total_gt_boxes)
        # 再現しなかったboxのwidth/height/aspect ratioのヒストグラムを出力
        import matplotlib.pyplot as plt
        plt.hist(unrec_widths, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('unrec_widths.hist.png')))
        plt.close()
        plt.hist(unrec_heights, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('unrec_heights.hist.png')))
        plt.close()
        plt.xscale('log')
        plt.hist(unrec_ars, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('unrec_ars.hist.png')))
        plt.close()

    def loss(self, y_true, y_pred):
        """損失関数。"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_confs, pred_locs = y_pred[:, :, :-4], y_pred[:, :, -4:]

        obj_mask = gt_confs[:, :, 0] <= K.epsilon()  # 背景以外
        obj_mask = K.cast(obj_mask, K.floatx())
        obj_count = K.sum(obj_mask)
        bg_mask = gt_confs[:, :, 0] > K.epsilon()  # 背景
        bg_mask = K.cast(bg_mask, K.floatx())
        bg_count = K.sum(bg_mask)

        # クラス分類のloss
        pred_confs = K.clip(pred_confs, K.epsilon(), 1 - K.epsilon())
        if _LOSS_FUNC == 'ce':
            # cross entropy
            loss_conf = -K.sum(gt_confs * K.log(pred_confs), axis=-1)
            loss_conf = K.mean(loss_conf)
        elif _LOSS_FUNC == 'bce':
            # balanced cross entropy (極端なので上手く動かなそう)
            loss_conf = -K.sum(gt_confs * K.log(pred_confs), axis=-1)
            loss_conf_obj = K.sum(loss_conf * obj_mask)
            loss_conf_bg = K.sum(loss_conf * bg_mask)
            loss_conf_obj = K.switch(obj_count <= K.epsilon(), 0.0, loss_conf_obj / obj_count)
            loss_conf_bg = K.switch(bg_count <= K.epsilon(), 0.0, loss_conf_bg / bg_count)
            loss_conf = loss_conf_obj + loss_conf_bg
        elif _LOSS_FUNC == 'focal':
            # Focal Loss (https://arxiv.org/pdf/1708.02002v1.pdf)
            gamma = 2.0
            alpha = 0.25
            alpha = np.array([[[alpha] + [1 - alpha] * (self.nb_classes - 1)]])
            loss_conf = -K.sum(alpha * gt_confs * K.pow(1 - pred_confs, gamma) * K.log(pred_confs), axis=-1)
            loss_conf = K.mean(loss_conf)
        else:
            assert _LOSS_FUNC in ('ce', 'bce', 'focal')

        # 位置のloss
        loss_loc = K.square(pred_locs - gt_locs)  # squared error (absolute errorでも良いかも?)
        loss_loc = K.sum(loss_loc * K.expand_dims(obj_mask, axis=-1))
        loss_loc = K.switch(obj_count <= K.epsilon(), 0.0, loss_loc / obj_count)

        return loss_conf + loss_loc

    def encode_truth(self, y_gt: [ObjectDetectionAnnotation]):
        """学習用の`y_true`の作成。

        IOUが0.5以上のものを全部割り当てる＆割り当らなかったらIOUが一番大きいものに割り当てる。
        1つのprior boxに複数割り当たってしまっていたらIOUが一番大きいものを採用。
        """
        confs = np.zeros((len(y_gt), len(self.pb_locs), self.nb_classes), dtype=np.float32)
        confs[:, :, 0] = 1
        locs = np.zeros((len(y_gt), len(self.pb_locs), 4), dtype=np.float32)
        # 画像ごとのループ
        for i, (_, _, classes, bboxes, difficults) in enumerate(y_gt):
            # prior_boxesとbboxesで重なっているものを探す
            iou = tk.ml.compute_iou(self.pb_locs, bboxes)

            pb_allocs = np.zeros((len(self.pb_locs),), dtype=np.int_)  # 割り当たっている個数のリスト
            pb_candidates = np.empty((len(self.pb_locs),), dtype=np.int_)  # 割り当てようとしているbb_ix

            for bb_ix, (class_id, difficult) in enumerate(zip(classes, difficults)):
                if difficult:
                    continue  # difficult == 1はスキップししておく
                targets = iou[:, bb_ix] >= 0.5
                if targets.any():
                    # IOUが0.5以上のものがあるなら全部割り当てる
                    pb_allocs[targets] += 1
                    pb_candidates[targets] = bb_ix
                else:
                    # 0.5以上のものが無ければIOUが一番大きいものに割り当てる
                    targets = iou[:, bb_ix].argmax()
                    if pb_candidates[targets] >= 0:  # 割り当て済みな場合
                        # assert iou[targets, bb_ix] >= 0.5  # 0.5未満のものが同じprior boxに割り当たるようだと困る..
                        pass  # TODO: あるのでなんとかしないと
                    pb_allocs[targets] += 1
                    pb_candidates[targets] = bb_ix

            for pb_ix in pb_allocs.nonzero()[0]:
                bb_ix = pb_candidates[pb_ix]
                class_id = classes[bb_ix]
                assert 0 < class_id < self.nb_classes
                # confs: 該当のクラスだけ1にする。
                confs[i, pb_ix, 0] = 0  # bg
                confs[i, pb_ix, class_id] = 1
                # locs: xmin, ymin, xmax, ymaxそれぞれのoffsetをそのまま回帰する。(prior_boxのサイズでスケーリングもする)
                locs[i, pb_ix, :] = (bboxes[bb_ix, :] - self.pb_locs[pb_ix, :]) / np.expand_dims(self.pb_sizes[pb_ix], axis=-1)

        # いったんくっつける (損失関数の中で分割して使う)
        return np.concatenate([confs, locs], axis=-1)

    def decode_predictions(self, predictions, confidence_threshold=0.3):
        """予測結果をデコードする。

        出力は以下の3つの値。画像ごとにconfidenceの降順。
        - class_id×検出数×画像数
        - confidence×検出数×画像数
        - (xmin, ymin, xmax, ymax)×検出数×画像数

        """
        assert predictions.shape[1] == len(self.pb_locs)
        assert predictions.shape[2] == self.nb_classes + 4
        confs_list, locs_list = predictions[:, :, :-4], predictions[:, :, -4:]
        classes, confs, locs = zip(*[
            self._decode_prediction(confs, locs, confidence_threshold)
            for confs, locs
            in zip(confs_list, locs_list)])
        return np.array(classes), np.array(confs), np.array(locs)

    def _decode_prediction(self, pred_confs, pred_locs, top_k=16):
        """予測結果のデコード。(1枚分)"""
        assert len(pred_confs) == len(pred_locs)
        assert pred_confs.shape == (len(self.pb_locs), self.nb_classes)
        assert pred_locs.shape == (len(self.pb_locs), 4)
        max_nonbg_confs = pred_confs[:, 1:].max(axis=-1)  # prior box数分の、背景以外のconfidenceの最大値

        # confidenceの上位top_k個のindexを降順に取得
        pb_ix = np.argpartition(max_nonbg_confs, -top_k)[-top_k:][::-1]

        # 該当するものを取得
        confs = max_nonbg_confs[pb_ix]
        classes = pred_confs[pb_ix, :].argmax(axis=-1)
        locs = (self.pb_locs[pb_ix, :] + pred_locs[pb_ix, :]) * np.expand_dims(self.pb_sizes[pb_ix], axis=-1)
        locs = np.clip(locs, 0, 1)  # はみ出ている分はクリッピング

        return classes, confs, locs


def _create_model(input_shape: tuple, pbox: PriorBoxes):
    import keras
    import keras.backend as K

    def _conv(x, filters, kernel_size, name=None, **kargs):
        assert name is not None
        x = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name=name + '_conv', **kargs)(x)
        x = keras.layers.BatchNormalization(name=name + '_bn')(x)
        x = keras.layers.ELU(name=name + '_act')(x)
        return x

    def _branch(x, filters, name):
        x = _conv(x, filters, (3, 3), padding='same', name=name + '_c1')
        x = keras.layers.Dropout(0.25, name=name + '_drop')(x)
        x = _conv(x, filters, (3, 3), padding='same', name=name + '_c2')
        return x

    def _1x1conv(x, filters, name):
        if filters is None:
            filters = K.int_shape(x)[-1] // 2
        x = _conv(x, filters, (1, 1), name=name)
        return x

    def _block(x, name):
        filters = K.int_shape(x)[-1]

        for i in range(4):
            b = _branch(x, filters // 4, name=name + '_b' + str(i))
            x = keras.layers.Concatenate()([x, b])

        x = _1x1conv(x, filters * 2, name=name + '_sq')
        return x

    def _small_block(x, filters, name):
        if K.int_shape(x)[-1] != filters:
            x = _conv(x, filters, (1, 1), padding='same', name=name + '_pre')
        sc = x
        x = keras.layers.Conv2D(filters // 4, (1, 1), padding='same', use_bias=False, name=name + '_b1')(x)
        x = keras.layers.BatchNormalization(name=name + '_b1bn')(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(filters // 4, (3, 3), padding='same', use_bias=False, name=name + '_b2')(x)
        x = keras.layers.BatchNormalization(name=name + '_b2bn')(x)
        x = keras.layers.ELU(name=name + '_b2act')(x)
        x = keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, name=name + '_b3')(x)
        x = keras.layers.Add()([sc, x])
        return x

    def _ds(x, kernel_size, strides, name):
        x = keras.layers.AveragePooling2D(kernel_size, strides=strides, name=name + '_pool')(x)
        return x

    net = {}
    x = inputs = keras.layers.Input(input_shape)
    x = _conv(x, 32, (7, 7), strides=(2, 2), padding='valid', name='start_conv')  # 384
    assert K.int_shape(x)[1] == 384
    x = keras.layers.MaxPooling2D(name='start_pool')(x)  # 192
    assert K.int_shape(x)[1] == 192
    x = _block(x, 'stage1_block')  # ch=64
    x = _ds(x, (2, 2), (2, 2), 'stage1_ds')  # 96
    assert K.int_shape(x)[1] == 96
    x = _block(x, 'stage2_block')  # ch=128
    x = _ds(x, (2, 2), (2, 2), 'stage2_ds')
    assert K.int_shape(x)[1] == 48
    net['out48'] = x

    x = _block(x, 'stage3_block')  # ch=256
    x = _ds(x, (2, 2), (2, 2), 'stage3_ds')
    assert K.int_shape(x)[1] == 24
    net['out24'] = x

    x = _block(x, 'stage4_block')  # ch=512
    x = _ds(x, (2, 2), (2, 2), 'stage4_ds')
    assert K.int_shape(x)[1] == 12
    net['out12'] = x

    x = _small_block(x, 512, 'stage5_conv')
    x = _ds(x, (2, 2), (2, 2), 'stage5_ds')
    assert K.int_shape(x)[1] == 6
    net['out6'] = x

    x = _small_block(x, 512, 'stage6_conv')
    x = _ds(x, (2, 2), (2, 2), 'stage6_ds')
    assert K.int_shape(x)[1] == 3
    net['out3'] = x

    x = keras.layers.MaxPooling2D((3, 3))(x)
    x = keras.layers.UpSampling2D((3, 3))(x)
    x = keras.layers.Concatenate()([x, net['out3']])
    x = _small_block(x, 512, 'stage6u_conv')
    net['out3'] = x

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([x, net['out6']])
    x = _small_block(x, 512, 'stage5u_conv')
    net['out6'] = x

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([x, net['out12']])
    x = _small_block(x, 512, 'stage4u_conv')
    net['out12'] = x

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([x, net['out24']])
    x = _small_block(x, 512, 'stage3u_conv')
    net['out24'] = x

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([x, net['out48']])
    x = _small_block(x, 512, 'stage2u_conv')
    net['out48'] = x

    # prediction moduleの重み共有用レイヤー
    pm_layers_dict = {}
    for pb_size_ratio in pbox.pb_size_ratios:
        pm_layers = [
            keras.layers.Conv2D(512, (1, 1), use_bias=False, name='pm-{}-conv'.format(pb_size_ratio)),
            keras.layers.BatchNormalization(name='pm-{}-bn'.format(pb_size_ratio)),
            keras.layers.Conv2D(len(pbox.aspect_ratios) * pbox.nb_classes, (1, 1), name='pm-{}-conf'.format(pb_size_ratio)),
            keras.layers.Conv2D(len(pbox.aspect_ratios) * 4, (1, 1), name='pm-{}-loc'.format(pb_size_ratio)),
        ]
        pm_layers_dict[pb_size_ratio] = pm_layers

    confs, locs = [], []
    for fm_count in PriorBoxes.FM_COUNTS:
        for pb_size_ratio in pbox.pb_size_ratios:
            pm_layers = pm_layers_dict[pb_size_ratio]
            x = net['out{}'.format(fm_count)]

            x = pm_layers[0](x)  # conv
            x = pm_layers[1](x)  # bn
            x = keras.layers.ELU()(x)

            conf = pm_layers[2](x)  # conv
            conf = keras.layers.Reshape((-1, pbox.nb_classes))(conf)

            loc = pm_layers[3](x)  # conv
            loc = keras.layers.Reshape((-1, 4))(loc)

            confs.append(conf)
            locs.append(loc)

    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    confs = keras.layers.Activation('softmax')(confs)

    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs])

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile('nadam', pbox.loss)
    # model.compile(keras.optimizers.SGD(momentum=0.9, nesterov=True), pbox.loss)
    return model


class Generator(tk.image.ImageDataGenerator):
    """データの読み込みを行うGenerator。"""

    def __init__(self, image_size, pbox):
        super().__init__(image_size)
        self.pbox = pbox

    def _prepare(self, X, y=None, weights=None, parallel=None, data_augmentation=False, rand=None):
        """画像の読み込みとDataAugmentation。"""
        X, y, weights = super()._prepare(X, y, weights, parallel, data_augmentation, rand)
        if y is not None:
            # 学習用に変換
            y = self.pbox.encode_truth(y)
        return X, y, weights

    def _transform(self, rgb, rand):
        """変形を伴うAugmentation。"""
        # とりあえず殺しておく
        return rgb


def print_map(model, gen, X_test, y_test, pbox, epoch):
    """`mAP`を算出してprintする。"""
    if epoch is not None and epoch % 16 != 0:
        return  # とりあえず重いので16回あたり1回だけ実施
    if epoch is not None:
        print('')

    pred = model.predict_generator(
        gen.flow(X_test, batch_size=_BATCH_SIZE),
        gen.steps_per_epoch(X_test.shape[0], _BATCH_SIZE),
        verbose=1)

    gt_classes_list = np.array([y.classes for y in y_test])
    gt_bboxes_list = np.array([y.bboxes for y in y_test])
    gt_difficults_list = np.array([y.difficults for y in y_test])
    pred_classes_list, _, pred_locs_list = pbox.decode_predictions(pred)
    map1 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_classes_list, pred_locs_list, use_voc2007_metric=False)
    map2 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_classes_list, pred_locs_list, use_voc2007_metric=True)

    print('mAP={:.4f} mAP(VOC2007)={:.4f}'.format(map1, map2))
    if epoch is not None:
        print('')


def run(logger, result_dir: pathlib.Path, data_dir: pathlib.Path):
    """実行。"""
    # データの読み込み
    annotations, nb_classes = load_annotations(data_dir)
    X_train = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')) +
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'trainval.txt'))
    )
    X_test = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'test.txt'))
    )
    y_train = np.array([annotations[x] for x in X_train], dtype=object)  # TODO: ここなんとかしたい…
    y_test = [annotations[x] for x in X_test]
    # Xのフルパス化
    image_dir1 = str(data_dir.joinpath('VOCdevkit', 'VOC2007', 'JPEGImages')) + '/'
    image_dir2 = str(data_dir.joinpath('VOCdevkit', 'VOC2012', 'JPEGImages')) + '/'
    X_train = np.array([(image_dir1 if len(x) == 6 else image_dir2) + x + '.jpg' for x in X_train])
    X_test = np.array([(image_dir1 if len(x) == 6 else image_dir2) + x + '.jpg' for x in X_test])
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # 訓練データからアスペクト比とサイズを適当に決める。
    # gridに配置したときのIOUを直接最適化するのは難しそうなので、
    # とりあえず大雑把にKMeansでクラスタ化するだけ。
    bboxes = np.concatenate([bboxes for _, _, _, bboxes, _ in y_train])
    # サイズ(feature mapのサイズからの相対値)
    sizes = np.sqrt((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
    fm_sizes = 1 / np.array(PriorBoxes.FM_COUNTS)
    pb_size_ratios = np.concatenate([sizes / fm_size for fm_size in fm_sizes])
    pb_size_ratios = pb_size_ratios[np.logical_and(pb_size_ratios >= 1.5, pb_size_ratios <= 4.5)]
    cluster = sklearn.cluster.KMeans(n_clusters=_PB_SIZE_PATTERNS, n_jobs=-1)
    cluster.fit(np.expand_dims(pb_size_ratios, axis=-1))
    pb_size_ratios = np.sort(cluster.cluster_centers_[:, 0])
    logger.debug('prior box size ratios = %s', str(pb_size_ratios))
    # アスペクト比
    log_ars = np.log((bboxes[:, 2] - bboxes[:, 0]) / (bboxes[:, 3] - bboxes[:, 1]))
    cluster = sklearn.cluster.KMeans(n_clusters=_ASPECT_RATIO_PATTERNS, n_jobs=-1)
    cluster.fit(np.expand_dims(log_ars, axis=-1))
    aspect_ratios = np.sort(np.exp(cluster.cluster_centers_[:, 0]))
    logger.debug('aspect ratios = %s', str(aspect_ratios))

    pbox = PriorBoxes(nb_classes, pb_size_ratios, aspect_ratios)
    logger.debug('prior box count = %d', len(pbox.pb_locs))
    logger.debug('prior box sizes = %s', str(np.unique(pbox.pb_sizes)))

    # prior boxのカバー度合いのチェック
    if _CHECK_PRIOR_BOX:  # 重いので必要なときだけ
        pbox.check_prior_boxes(logger, result_dir, annotations)

    import keras
    input_shape = pbox.input_size + (3,)
    model = _create_model(input_shape, pbox)
    model.summary(print_fn=logger.debug)
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

    # 事前学習の読み込み
    model.load_weights(str(result_dir.parent.joinpath('voc_pre', 'model.h5')), by_name=True)

    gen = Generator(image_size=input_shape[:2], pbox=pbox)
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

    callbacks = []
    callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=1e-3))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
    # callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
    # if K.backend() == 'tensorflow':
    #     callbacks.append(keras.callbacks.TensorBoard())

    # 各epoch毎にmAPを算出して表示してみる
    callbacks.append(keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print_map(model, gen, X_test, y_test, pbox, epoch)
    ))

    # lossの確認用コード
    if False:
        callbacks.append(keras.callbacks.LambdaCallback(
            on_batch_end=lambda batch, logs: print('\nloss=%f\n' % logs['loss'])
        ))

    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=_BATCH_SIZE),
        steps_per_epoch=gen.steps_per_epoch(X_train.shape[0], _BATCH_SIZE),
        epochs=_MAX_EPOCH,
        # validation_data=gen.flow(X_test, y_test, batch_size=_BATCH_SIZE),
        # validation_steps=gen.steps_per_epoch(X_test.shape[0], _BATCH_SIZE),
        callbacks=callbacks)

    # 最終結果
    print_map(model, gen, X_test, y_test, pbox, None)

    model.save(str(result_dir.joinpath('model.h5')))


def load_annotations(data_dir: pathlib.Path) -> (dict, int):
    """VOC2007,VOC2012のアノテーションデータの読み込み。

    結果は「ファイル名拡張子なし」とObjectDetectionAnnotationのdictと、クラス数。
    """
    import xml.etree

    data = {}
    for folder in ('VOC2007', 'VOC2012'):
        for f in data_dir.joinpath('VOCdevkit', folder, 'Annotations').iterdir():
            root = xml.etree.ElementTree.parse(str(f)).getroot()
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            classes = []
            bboxes = []
            difficults = []
            for object_tree in root.findall('object'):
                class_id = _CLASS_NAME_TO_ID[object_tree.find('name').text]
                bndbox = object_tree.find('bndbox')
                xmin = float(bndbox.find('xmin').text) / width
                ymin = float(bndbox.find('ymin').text) / height
                xmax = float(bndbox.find('xmax').text) / width
                ymax = float(bndbox.find('ymax').text) / height
                difficult = object_tree.find('difficult').text == '1'
                classes.append(class_id)
                bboxes.append([xmin, ymin, xmax, ymax])
                difficults.append(difficult)
            data[f.stem] = ObjectDetectionAnnotation(
                width=width,
                height=height,
                classes=np.array(classes),
                bboxes=np.array(bboxes),
                difficults=np.array(difficults))
    return data, len(_CLASS_NAMES)
