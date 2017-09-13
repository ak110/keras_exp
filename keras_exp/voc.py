"""ObjectDetectionを適当にやってみるコード。

-train: VOC2007 trainval + VOC2012 trainval
-test:  VOC2007 test
"""
import pathlib

import numpy as np
import sklearn.cluster
import sklearn.metrics
from tqdm import tqdm

import pytoolkit as tk

_DEBUG = False
_USE_BN = True  # not _DEBUG
_USE_DO = True  # not _DEBUG

_BATCH_SIZE = 16
_MAX_EPOCH = 300
_LARGE_IMAGE_SIZE = False
_TRAIN_DIFFICULT = True
_ASPECT_RATIO_PATTERNS = 5
_PB_SIZE_PATTERNS = 3

_CHECK_PRIOR_BOX = False  # 重いので必要なときだけ
_PRINT_MAP = False  # 重いので必要なときだけ

_CLASS_NAMES = ['bg'] + tk.ml.VOC_CLASS_NAMES
_CLASS_NAME_TO_ID = {n: i for i, n in enumerate(_CLASS_NAMES)}


class PriorBoxes(object):
    """候補として最初に準備するboxの集合。"""

    # 取り出すfeature mapのサイズ
    FM_COUNTS = (48, 24, 12, 6, 3)

    def __init__(self, nb_classes, mean_objets, pb_size_ratios=(1.5, 2.0, 2.5), aspect_ratios=(1, 2, 1 / 2, 3, 1 / 3)):
        self.nb_classes = nb_classes
        self.input_size = (773, 773) if _LARGE_IMAGE_SIZE else (390, 390)
        # 1枚の画像あたりの平均オブジェクト数
        self.mean_objets = mean_objets
        # 1 / fm_countに対する、prior boxの基準サイズの割合。3なら3倍の大きさのものを用意。
        self.pb_size_ratios = pb_size_ratios
        # アスペクト比のリスト
        self.aspect_ratios = aspect_ratios

        # shape=(box数, 4)で座標
        self.pb_locs = None
        # shape=(box数,)で、何種類目のprior boxか (集計用)
        self.pb_indices = None
        # shape=(box数,4)でpred_locsのスケール
        self.pb_scales = None
        # shape=(config数,)でpb_size
        self.pb_sizes = None
        # 各prior boxの情報をdictで保持
        self.config = None

        self._create_prior_boxes()

        nb_pboxes = len(self.pb_locs)
        assert self.pb_locs.shape == (nb_pboxes, 4)
        assert self.pb_indices.shape == (nb_pboxes,)
        assert self.pb_scales.shape == (nb_pboxes, 4)
        assert self.pb_sizes.shape == (len(PriorBoxes.FM_COUNTS) * len(self.pb_size_ratios),)
        assert len(self.config) == len(PriorBoxes.FM_COUNTS) * len(self.pb_size_ratios)

    def _create_prior_boxes(self):
        """基準となるbox(prior box)の座標(x1, y1, x2, y2)の並びを返す。"""
        self.pb_locs = []
        self.pb_indices = []
        self.pb_scales = []
        self.pb_sizes = []
        self.config = []
        for fm_count in PriorBoxes.FM_COUNTS:
            for pb_size_ratio in self.pb_size_ratios:
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
                prior_boxes = np.tile(prior_boxes, (len(self.aspect_ratios), 1, 2))

                # prior boxのサイズ
                pb_size = tile_size * pb_size_ratio
                # (x1, y1, x2, y2)の位置を調整。縦横半分ずつ動かす。
                half_box_widths = np.array([[0.5 * pb_size * np.sqrt(ar)] for ar in self.aspect_ratios])
                half_box_heights = np.array([[0.5 * pb_size / np.sqrt(ar)] for ar in self.aspect_ratios])
                prior_boxes[:, :, 0] -= half_box_widths
                prior_boxes[:, :, 1] -= half_box_heights
                prior_boxes[:, :, 2] += half_box_widths
                prior_boxes[:, :, 3] += half_box_heights
                # はみ出ているのはclipしておく
                prior_boxes = np.clip(prior_boxes, 0, 1)

                # (アスペクト比, タイル, 4) → (アスペクト比×タイル, 4)
                prior_boxes = prior_boxes.reshape(-1, 4)

                self.pb_locs.extend(prior_boxes)
                self.pb_indices.extend([len(self.config)] * len(prior_boxes))
                self.pb_scales.extend(np.tile(prior_boxes[:, 2:] - prior_boxes[:, :2], 2))
                self.pb_sizes.append(pb_size)
                self.config.append({'fm_count': fm_count, 'pb_size': pb_size, 'box_count': len(prior_boxes)})

        self.pb_locs = np.array(self.pb_locs)
        self.pb_indices = np.array(self.pb_indices)
        self.pb_sizes = np.array(self.pb_sizes)
        self.pb_scales = np.array(self.pb_scales)

    def check_prior_boxes(self, logger, result_dir, y_test: [tk.ml.ObjectsAnnotation]):
        """データに対して`self.pb_locs`がどれくらいマッチしてるか調べる。"""
        y_true = []
        y_pred = []
        match_counts = [0 for _ in range(len(self.config))]
        rec_mean_abs_delta = []
        unrec_widths = []
        unrec_heights = []
        unrec_ars = []
        for y in tqdm(y_test, desc='check_prior_boxes', ascii=True, ncols=100):
            iou = tk.ml.compute_iou(self.pb_locs, y.bboxes)
            # クラスごとに再現率を求める
            for gt_ix, class_id in enumerate(y.classes):
                assert 1 <= class_id < self.nb_classes
                m = iou[:, gt_ix] >= 0.5
                success = m.any()
                y_true.append(class_id)
                y_pred.append(class_id if success else 0)  # IOUが0.5以上のboxが存在すれば一致扱いとする
                if success:
                    pb_ix = self.pb_indices[iou[:, gt_ix].argmax()]
                    mean_abs_delta = np.mean(np.abs((y.bboxes[gt_ix, :] - self.pb_locs[pb_ix, :]) / self.pb_scales[pb_ix, :]))
                    match_counts[pb_ix] += 1
                    rec_mean_abs_delta.append(mean_abs_delta)
            # 再現(iou >= 0.5)しなかったboxの情報を集める
            for bbox in y.bboxes[iou.max(axis=0) < 0.5]:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                unrec_widths.append(w)
                unrec_heights.append(h)
                unrec_ars.append(w / h)
        # 再現率などの表示
        y_true.append(0)  # 警告よけにbgも1個入れておく
        y_pred.append(0)  # 警告よけにbgも1個入れておく
        total_gt_boxes = sum([len(y.bboxes) for y in y_test])
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
        plt.hist(rec_mean_abs_delta, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('rec_mean_abs_delta.hist.png')))
        plt.close()
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

        obj_mask = K.less(gt_confs[:, :, 0], 0.5)  # 背景以外
        obj_mask = K.cast(obj_mask, K.floatx())
        obj_count = K.sum(obj_mask, axis=-1)  # 各batch毎のobj数。
        obj_div = K.maximum(obj_count, K.ones_like(obj_count))

        loss_conf = self._loss_conf(gt_confs, pred_confs, obj_div)
        loss_loc = self._loss_loc(gt_locs, pred_locs, obj_mask, obj_div)
        return loss_conf + loss_loc

    def _loss_conf(self, gt_confs, pred_confs, obj_div):
        """分類のloss。"""
        import keras.backend as K
        loss_conf = tk.dl.categorical_focal_loss(gt_confs, pred_confs, self.nb_classes)
        loss_conf = K.sum(loss_conf, axis=-1) / obj_div  # normalized by the number of anchors assigned to a ground-truth box
        return loss_conf

    @staticmethod
    def _loss_loc(gt_locs, pred_locs, obj_mask, obj_div):
        """位置のloss。"""
        import keras.backend as K
        loss_loc = tk.dl.l1_smooth_loss(gt_locs, pred_locs)
        loss_loc = K.sum(loss_loc * obj_mask, axis=-1) / obj_div  # mean
        return loss_loc

    @staticmethod
    def acc_bg(y_true, y_pred):
        """背景の正解率。"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-4]
        pred_confs = y_pred[:, :, :-4]

        bg_mask = gt_confs[:, :, 0] >= 0.5  # 背景
        bg_mask = K.cast(bg_mask, K.floatx())
        bg_count = K.sum(bg_mask, axis=-1)

        acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
        return K.sum(acc * bg_mask, axis=-1) / bg_count

    @staticmethod
    def acc_obj(y_true, y_pred):
        """物体の正解率。"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-4]
        pred_confs = y_pred[:, :, :-4]

        obj_mask = gt_confs[:, :, 0] < 0.5  # 背景以外
        obj_mask = K.cast(obj_mask, K.floatx())
        obj_count = K.sum(obj_mask, axis=-1)

        acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
        return K.sum(acc * obj_mask, axis=-1) / obj_count

    @staticmethod
    def loss_loc(y_true, y_pred):
        """位置の損失項。"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_locs = y_pred[:, :, -4:]

        obj_mask = gt_confs[:, :, 0] < 0.5  # 背景以外
        obj_mask = K.cast(obj_mask, K.floatx())
        obj_count = K.sum(obj_mask, axis=-1)
        obj_div = K.maximum(obj_count, K.ones_like(obj_count))

        return PriorBoxes._loss_loc(gt_locs, pred_locs, obj_mask, obj_div)

    @staticmethod
    def mae_loc(y_true, y_pred):
        """位置のずれの絶対値の平均。"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_locs = y_pred[:, :, -4:]

        obj_mask = gt_confs[:, :, 0] < 0.5  # 背景以外
        obj_mask = K.cast(obj_mask, K.floatx())
        obj_count = K.sum(obj_mask, axis=-1)
        obj_div = K.maximum(obj_count, K.ones_like(obj_count))

        mae = K.mean(K.abs(gt_locs - pred_locs), axis=-1)
        mae *= obj_mask
        mae = K.sum(mae, axis=-1) / obj_div
        return mae

    def encode_truth(self, y_gt):
        """学習用の`y_true`の作成。

        IOUが0.5以上のものを全部割り当てる＆割り当らなかったらIOUが一番大きいものに割り当てる。
        1つのprior boxに複数割り当たってしまっていたらIOUが一番大きいものを採用。
        """
        confs = np.zeros((len(y_gt), len(self.pb_locs), self.nb_classes), dtype=np.float32)
        confs[:, :, 0] = 1
        locs = np.zeros((len(y_gt), len(self.pb_locs), 4), dtype=np.float32)
        # 画像ごとのループ
        for i, y in enumerate(y_gt):
            y = tk.ml.ObjectsAnnotation(*y)
            # prior_boxesとbboxesで重なっているものを探す
            iou = tk.ml.compute_iou(self.pb_locs, y.bboxes)

            pb_confs = np.zeros((len(self.pb_locs),), dtype=bool)  # 割り当てようとしているconfidence
            pb_candidates = np.empty((len(self.pb_locs),), dtype=int)  # 割り当てようとしているgt_ix

            for gt_ix, (class_id, difficult) in enumerate(zip(y.classes, y.difficults)):
                if not _TRAIN_DIFFICULT and difficult:
                    continue
                pb_ixs = np.where(iou[:, gt_ix] >= 0.5)[0]
                if pb_ixs.any():
                    # IOUが0.5以上のものがあるなら全部割り当てる
                    iou_list = iou[pb_ixs, gt_ix]
                    obj_confs = iou_list / iou_list.max()  # iouが最大のもの以外はconfが低めになるように割合で割り当てる
                    assert (obj_confs > 0.5).all()
                    pb_confs[pb_ixs] = obj_confs
                    pb_candidates[pb_ixs] = gt_ix
                else:
                    # 0.5以上のものが無ければIOUが一番大きいものに割り当てる
                    pb_ixs = iou[:, gt_ix].argmax()
                    if pb_candidates[pb_ixs] >= 0:  # 割り当て済みな場合
                        # assert iou[pb_ixs, gt_ix] >= 0.5  # 0.5未満のものが同じprior boxに割り当たるようだと困る..
                        pass
                    pb_confs[pb_ixs] = 1
                    pb_candidates[pb_ixs] = gt_ix

            for pb_ix in pb_confs.nonzero()[0]:
                gt_ix = pb_candidates[pb_ix]
                class_id = y.classes[gt_ix]
                conf = pb_confs[pb_ix]
                assert 0 < class_id < self.nb_classes
                assert 0 < conf <= 1
                # confs: 該当のクラスだけ1にする。
                confs[i, pb_ix, 0] = 1 - conf  # bg
                confs[i, pb_ix, class_id] = conf
                # locs: xmin, ymin, xmax, ymaxそれぞれのoffsetをそのまま回帰する。(prior_boxのサイズでスケーリングもする)
                locs[i, pb_ix, :] = (y.bboxes[gt_ix, :] - self.pb_locs[pb_ix, :]) / self.pb_scales[pb_ix, :]

        # 分類の教師が合計≒1になっていることの確認
        assert (np.abs(confs.sum(axis=-1) - 1) < 1e-7).all()

        # いったんくっつける (損失関数の中で分割して使う)
        return np.concatenate([confs, locs], axis=-1)

    def decode_predictions(self, predictions, top_k=16, detect_least_conf=0.9, detect_min_conf=0.6, collision_iou=0.5):
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
            self._decode_prediction(confs, locs, top_k, detect_least_conf, detect_min_conf, collision_iou)
            for confs, locs
            in zip(confs_list, locs_list)])
        return classes, confs, locs

    def _decode_prediction(self, pred_confs, pred_locs, top_k, detect_least_conf, detect_min_conf, collision_iou):
        """予測結果のデコード。(1枚分)"""
        assert len(pred_confs) == len(pred_locs)
        assert pred_confs.shape == (len(self.pb_locs), self.nb_classes)
        assert pred_locs.shape == (len(self.pb_locs), 4)
        classes = []
        confs = []
        locs = []
        # confidenceの上位を降順にループ
        obj_confs = pred_confs[:, 1:].max(axis=-1)  # prior box数分の、背景以外のconfidenceの最大値
        for pb_ix in np.argsort(obj_confs)[::-1]:
            loc = self.pb_locs[pb_ix, :] + pred_locs[pb_ix, :] * self.pb_scales[pb_ix, :]
            loc = np.clip(loc, 0, 1)  # はみ出ている分はクリッピング
            assert loc.shape == (4,)
            # 幅や高さが0以下になってしまっているものはスキップする
            if (loc[2:] <= loc[:2]).any():
                continue
            # 既に出現済みのものと大きく重なっているものはスキップする
            if len(locs) >= 1 and tk.ml.compute_iou(np.array(locs), np.array([loc])).max() >= collision_iou:
                continue

            conf = obj_confs[pb_ix]
            # 充分高いconfidenceのものがあるなら、ある程度以下のconfidenceのものは無視
            if len(confs) >= 1 and detect_least_conf <= confs[0] and conf < detect_min_conf:
                break

            classes.append(pred_confs[pb_ix, 1:].argmax(axis=-1) + 1)
            confs.append(conf)
            locs.append(loc)
            # 最大top_k個まで検知扱いにする
            if top_k <= len(locs):
                break

        return np.array(classes), np.array(confs), np.array(locs)


def _create_model(input_shape: tuple, pbox: PriorBoxes):
    import keras
    import keras.backend as K
    from keras.regularizers import l2

    def _conv(x, filters, kernel_size, use_bias, name, **kargs):
        x = keras.layers.Conv2D(filters, kernel_size, use_bias=use_bias, name=name, **kargs)(x)
        return x

    def _conv_bn(x, filters, kernel_size, name, **kargs):
        x = _conv(x, filters, kernel_size, use_bias=not _USE_BN, name=name, **kargs)
        if _USE_BN:
            x = keras.layers.BatchNormalization(name=name + '_bn')(x)
        return x

    def _conv_bn_act(x, filters, kernel_size, name=None, **kargs):
        assert name is not None
        x = _conv_bn(x, filters, kernel_size, name, **kargs)
        x = keras.layers.ELU(name=name + '_act')(x)
        return x

    def _branch(x, filters, name):
        x = _conv_bn_act(x, filters, (3, 3), padding='same', name=name + '_c1')
        if _USE_DO:
            x = keras.layers.Dropout(0.25, name=name + '_drop')(x)
        x = _conv_bn_act(x, filters, (3, 3), padding='same', name=name + '_c2')
        return x

    def _1x1conv(x, filters, name):
        if filters is None:
            filters = K.int_shape(x)[-1] // 2
        x = _conv_bn_act(x, filters, (1, 1), name=name)
        return x

    def _block(x, name):
        filters = K.int_shape(x)[-1]

        for i in range(4):
            b = _branch(x, filters // 4, name=name + '_b' + str(i))
            x = keras.layers.Concatenate()([x, b])

        x = _1x1conv(x, filters * 2, name=name + '_sq')
        return x

    def _small_block(x, filters, name):
        x = _conv_bn(x, filters // 2, (1, 1), padding='same', name=name + '_b1')
        if _USE_DO:
            x = keras.layers.Dropout(0.25)(x)
        x = _conv_bn_act(x, filters, (3, 3), padding='same', name=name + '_b2')
        return x

    def _ds(x, kernel_size, strides, name):
        x = keras.layers.AveragePooling2D(kernel_size, strides=strides, name=name + '_pool')(x)
        return x

    net = {}
    x = inputs = keras.layers.Input(input_shape)
    x = _conv_bn_act(x, 32, (7, 7), strides=(2, 2) if _LARGE_IMAGE_SIZE else (1, 1), padding='valid', name='start_conv')  # 384
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

    x = _small_block(x, 512, 'stage5_block')
    x = _ds(x, (2, 2), (2, 2), 'stage5_ds')
    assert K.int_shape(x)[1] == 6
    net['out6'] = x

    x = _small_block(x, 512, 'stage6_block')
    x = _ds(x, (2, 2), (2, 2), 'stage6_ds')
    assert K.int_shape(x)[1] == 3
    net['out3'] = x

    x = keras.layers.MaxPooling2D((3, 3))(x)
    x = keras.layers.UpSampling2D((3, 3))(x)
    x = keras.layers.Concatenate()([x, net['out3']])
    x = _small_block(x, 512, 'stage6u_block')
    net['out3'] = x

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([x, net['out6']])
    x = _small_block(x, 512, 'stage5u_block')
    net['out6'] = x

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([x, net['out12']])
    x = _small_block(x, 512, 'stage4u_block')
    net['out12'] = x

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([x, net['out24']])
    x = _small_block(x, 512, 'stage3u_block')
    net['out24'] = x

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([x, net['out48']])
    x = _small_block(x, 512, 'stage2u_block')
    net['out48'] = x

    # prediction moduleの重み共有用レイヤー
    pm_layers = {}
    for pb_size_ratio in pbox.pb_size_ratios:
        for target, filters in (('conf', 256), ('loc', 64)):
            layers = []
            for i in range(2):
                kernel_size = (1, 1)
                layers.append(keras.layers.Conv2D(
                    filters, kernel_size, padding='same', use_bias=True,
                    name='pm-{}-{}-conv{}'.format(pb_size_ratio, target, i)))
                layers.append(keras.layers.ELU(name='pm-{}-{}-act{}'.format(pb_size_ratio, target, i)))
            if target == 'conf':
                layers.append(keras.layers.Conv2D(
                    len(pbox.aspect_ratios) * pbox.nb_classes, (1, 1), use_bias=True,
                    kernel_initializer=tk.dl.focal_loss_kernel_initializer(),
                    bias_initializer=tk.dl.focal_loss_bias_initializer(pbox.nb_classes),
                    bias_regularizer=l2(1e-4),  # 初期値が±3.8とかなので、徐々に減らしたい
                    name='pm-{}-conf'.format(pb_size_ratio)))
            else:
                layers.append(keras.layers.Conv2D(
                    len(pbox.aspect_ratios) * 4, (1, 1), use_bias=True,
                    name='pm-{}-loc'.format(pb_size_ratio)))
            pm_layers['{}-{}'.format(pb_size_ratio, target)] = layers

    confs, locs = [], []
    for fm_count in PriorBoxes.FM_COUNTS:
        for pb_size_ratio in pbox.pb_size_ratios:
            x = net['out{}'.format(fm_count)]

            conf = x
            for layer in pm_layers['{}-{}'.format(pb_size_ratio, 'conf')]:
                conf = layer(conf)
            conf = keras.layers.Reshape((-1, pbox.nb_classes))(conf)
            confs.append(conf)

            loc = x
            for layer in pm_layers['{}-{}'.format(pb_size_ratio, 'loc')]:
                loc = layer(loc)
            loc = keras.layers.Reshape((-1, 4))(loc)
            locs.append(loc)

    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    confs = keras.layers.Activation('softmax')(confs)

    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs])

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile('nadam', pbox.loss, [pbox.loss_loc, pbox.acc_bg, pbox.acc_obj, pbox.mae_loc])
    # model.compile(keras.optimizers.SGD(momentum=0.9, nesterov=True), pbox.loss, [pbox.loss_loc, pbox.acc_bg, pbox.acc_obj, pbox.mae_loc])
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


def print_map(logger, pbox, model, gen, X_test, y_test, epoch):
    """`mAP`を算出してprintする。"""
    if epoch is not None and not (epoch % 16 == 0 or epoch & (epoch - 1) == 0):
        return  # 重いので16回あたり1回だけ実施
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

    logger.debug('mAP={:.4f} mAP(VOC2007)={:.4f}'.format(map1, map2))
    if epoch is not None:
        print('')


def plot_truth(X_test, y_test, save_dir):
    """正解データの画像化。"""
    for X, y in zip(X_test, y_test):
        tk.ml.plot_objects(
            X, save_dir.joinpath(pathlib.Path(X).name + '.png'),
            y.classes, None, y.bboxes, _CLASS_NAMES)


def plot_result(pbox, model, gen, X_test, save_dir):
    """結果の画像化。"""
    pred = model.predict_generator(
        gen.flow(X_test, batch_size=_BATCH_SIZE),
        gen.steps_per_epoch(X_test.shape[0], _BATCH_SIZE),
        verbose=1)
    pred_classes_list, pred_confs_list, pred_locs_list = pbox.decode_predictions(pred)

    for X, pred_classes, pred_confs, pred_locs in zip(X_test, pred_classes_list, pred_confs_list, pred_locs_list):
        tk.ml.plot_objects(
            X, save_dir.joinpath(pathlib.Path(X).name + '.png'),
            pred_classes, pred_confs, pred_locs, _CLASS_NAMES)


def run(logger, result_dir: pathlib.Path, data_dir: pathlib.Path):
    """実行。"""
    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        _run(logger, result_dir, data_dir)


def _run(logger, result_dir: pathlib.Path, data_dir: pathlib.Path):
    """実行。"""
    # データの読み込み
    nb_classes = len(_CLASS_NAMES)
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
    y_train = [annotations[x] for x in X_train]
    y_test = [annotations[x] for x in X_test]
    y_train_arr = np.array(y_train, dtype=object)  # TODO: ここなんとかしたい…
    y_test_arr = np.array(y_test, dtype=object)  # TODO: ここなんとかしたい…
    # Xのフルパス化
    X_train = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_train])
    X_test = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_test])
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # 試しに回答を出力してみる。
    plot_truth(X_test[:_BATCH_SIZE], y_test[:_BATCH_SIZE], result_dir.joinpath('___ground_truth'))
    # for i, y in enumerate(y_test[:3]):
    #     logger.debug('y_test[%d]: %s', i, str(y))

    # 訓練データからパラメータを適当に決める。
    # gridに配置したときのIOUを直接最適化するのは難しそうなので、
    # とりあえず大雑把にKMeansでクラスタ化したりなど。
    bboxes = np.concatenate([y.bboxes for y in y_train])
    # 平均オブジェクト数
    mean_objets = np.mean([len(y.bboxes) for y in y_train])
    logger.debug('mean objects / image = %f', mean_objets)
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

    pbox = PriorBoxes(nb_classes, mean_objets, pb_size_ratios, aspect_ratios)
    logger.debug('prior box count = %d', len(pbox.pb_locs))
    logger.debug('prior box sizes = %s', str(np.unique(pbox.pb_sizes)))

    # prior boxのカバー度合いのチェック
    if _CHECK_PRIOR_BOX:
        # pbox.check_prior_boxes(logger, result_dir, y_train)
        pbox.check_prior_boxes(logger, result_dir, y_test)

    import keras
    input_shape = pbox.input_size + (3,)
    model = _create_model(input_shape, pbox)
    model.summary(print_fn=logger.debug)
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)

    # 事前学習の読み込み
    if _USE_BN:
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
    callbacks.append(keras.callbacks.ModelCheckpoint(str(result_dir.joinpath('model.best.h5')), save_best_only=True))
    # callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
    # if K.backend() == 'tensorflow':
    #     callbacks.append(keras.callbacks.TensorBoard())

    # 各epoch毎にmAPを算出して表示してみる
    if _DEBUG or _PRINT_MAP:
        callbacks.append(keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print_map(logger, pbox, model, gen, X_test, y_test, epoch)
        ))

    model.fit_generator(
        gen.flow(X_train, y_train_arr, batch_size=_BATCH_SIZE, data_augmentation=not _DEBUG, shuffle=not _DEBUG),
        steps_per_epoch=gen.steps_per_epoch(X_train.shape[0], _BATCH_SIZE),
        epochs=_MAX_EPOCH,
        validation_data=gen.flow(X_test, y_test_arr, batch_size=_BATCH_SIZE),
        validation_steps=gen.steps_per_epoch(X_test.shape[0], _BATCH_SIZE),
        callbacks=callbacks)

    model.save(str(result_dir.joinpath('model.h5')))

    # 最終結果表示
    print_map(logger, pbox, model, gen, X_test, y_test, None)
    plot_result(pbox, model, gen, X_test[:_BATCH_SIZE], result_dir.joinpath('___check'))


def load_annotations(data_dir: pathlib.Path) -> dict:
    """VOC2007,VOC2012のアノテーションデータの読み込み。"""
    data = {}
    for folder in ('VOC2007', 'VOC2012'):
        data.update(tk.ml.load_voc_annotations(
            data_dir.joinpath('VOCdevkit', folder, 'Annotations'), _CLASS_NAME_TO_ID))
    return data
