"""VOC2007のObjectDetectionを適当にやってみるコード。"""
import pathlib
import collections

import numpy as np
import sklearn.metrics

import pytoolkit as tk

BATCH_SIZE = 8
MAX_EPOCH = 300

# ground truthなデータを保持する用。
# classes、bboxes、difficultsはそれぞれbounding boxの数分の配列。
ObjectDetectionAnnotation = collections.namedtuple('ObjectDetectionAnnotation', 'width,height,classes,bboxes,difficults')


class PriorBoxes(object):
    """候補として最初に準備するbounding boxの集合。"""

    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
        self.input_size = (512, 512)
        self.config = [
            {'tile_count': 63, 'aspect_ratios': [1]},
            {'tile_count': 31, 'aspect_ratios': [1, 2, 1 / 2, 4, 1 / 4]},
            {'tile_count': 15, 'aspect_ratios': [1, 2, 1 / 2, 4, 1 / 4]},
            {'tile_count': 7, 'aspect_ratios': [1, 1.4, 1 / 1.4, 2, 1 / 2, 2.8, 1 / 2.8, 4, 1 / 4]},
            {'tile_count': 3, 'aspect_ratios': [1, 1.4, 1 / 1.4, 2, 1 / 2, 2.8, 1 / 2.8, 4, 1 / 4]},
            {'tile_count': 1, 'aspect_ratios': [1, 1.4, 1 / 1.4, 2, 1 / 2, 2.8, 1 / 2.8, 4, 1 / 4]},
        ]
        # 各tileで扱うboxのサイズを適当に決める。指数的に増やすよりは線形で増やすほうが良さそう。
        # (大きめのboxを沢山用意しておくイメージ)
        box_size_diff = 1 / (len(self.config) - 1)
        for i, c in enumerate(self.config):
            c['min_size'] = box_size_diff / 2 if i == 0 else self.config[i - 1]['max_size']
            c['max_size'] = c['min_size'] + box_size_diff
        # 基準となるbox(prior box)の座標を事前に算出しておく
        # prior_boxes: shape=(box数, 4)で座標
        # pb_indices: shape=(box数,)で、self.configのindex
        self.prior_boxes, self.pb_indices = self._create_prior_boxes(self.config)

    @staticmethod
    def _create_prior_boxes(config):
        """基準となるbox(prior box)の座標(x1, y1, x2, y2)の並びを返す。"""

        boxes = []
        pb_indices = []
        for i, c in enumerate(config):
            tile_count = c['tile_count']
            min_size = c['min_size']
            max_size = c['max_size']
            aspect_ratios = c['aspect_ratios']

            # 敷き詰める間隔
            tile_size = 1.0 / tile_count
            # 敷き詰めたときの中央の位置のリスト
            lin = np.linspace(0.5 * tile_size, 1 - 0.5 * tile_size, tile_count, dtype=np.float32)
            # 縦横に敷き詰め
            centers_x, centers_y = np.meshgrid(lin, lin)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)
            prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
            # x, y → x1, y1, x2, y2
            prior_boxes = np.tile(prior_boxes, (len(aspect_ratios), 1, 2))

            # (x1, y1, x2, y2)の位置を調整。縦横半分ずつ動かす。
            box_default_size = np.sqrt(min_size * max_size) if min_size else np.sqrt(max_size)
            half_box_widths = np.array([[0.5 * box_default_size * np.sqrt(ar)] for ar in aspect_ratios])
            half_box_heights = np.array([[0.5 * box_default_size / np.sqrt(ar)] for ar in aspect_ratios])
            prior_boxes[:, :, 0] -= half_box_widths
            prior_boxes[:, :, 1] -= half_box_heights
            prior_boxes[:, :, 2] += half_box_widths
            prior_boxes[:, :, 3] += half_box_heights
            # はみ出ているのはclipしておく
            prior_boxes = np.clip(prior_boxes, 0, 1)

            # (アスペクト比, タイル, 4) → (アスペクト比×タイル, 4)
            prior_boxes = prior_boxes.reshape(-1, 4)
            c['box_count'] = len(prior_boxes)  # box個数を記録しとく。

            boxes.append(prior_boxes)
            pb_indices.extend([i] * len(prior_boxes))
        return np.concatenate(boxes), np.array(pb_indices)

    def check_prior_boxes(self, logger, gt):
        """データに対して`self.prior_boxes`がどれくらいマッチしてるか調べる。

        本当はここも学習したほうが良さそうだけどとりあえず。 (cf. YOLO9000)
        """
        y_true = []
        y_pred = []
        match_counts = [0 for _ in range(len(self.config))]
        for _, _, classes, bboxes, _ in gt.values():
            for class_id, bbox in zip(classes, bboxes):
                iou = tk.ml.compute_iou(self.prior_boxes, np.expand_dims(bbox, 0))
                m = iou >= 0.5
                success = m.any()
                y_true.append(class_id)
                y_pred.append(class_id if success else 0)  # IOUが0.5以上のboxが存在すれば一致扱いとする
                if success:
                    match_counts[self.pb_indices[iou.argmax()]] += 1
        y_true.append(0)  # 警告よけにbgも1個入れておく
        y_pred.append(0)  # 警告よけにbgも1個入れておく
        total_gt_boxes = sum([len(bboxes) for _, _, _, bboxes, _ in gt.values()])
        cr = sklearn.metrics.classification_report(y_true, y_pred)
        logger.debug('prior boxs = %d', len(self.prior_boxes))
        logger.debug(cr)
        logger.debug('match counts:')
        for i, c in enumerate(match_counts):
            logger.debug('  prior_boxes_%d = %d (%.02f%%)',
                         self.config[i]['tile_count'], c,
                         100 * c / self.config[i]['box_count'] / total_gt_boxes)

    def loss(self, y_true, y_pred):
        """損失関数。"""
        import keras
        # import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_confs, pred_locs = y_pred[:, :, :-4], y_pred[:, :, -4:]

        # クラス分類のloss。
        loss_conf = keras.losses.categorical_crossentropy(gt_confs, pred_confs)
        # 位置のloss。
        loss_loc = keras.losses.mean_squared_error(gt_locs, pred_locs)

        # TODO: 補正？
        # mask = gt_confs[:, :, 0] <= K.epsilon()
        # mask_size = K.sum(K.cast(mask, K.floatx()))  # 背景のconfidenceが0なものがオブジェクトなはず
        # loss_loc = loss_loc * K.prod(K.shape(gt_locs)[:2]) / mask_size  # 補正

        return loss_conf + loss_loc

    def encode_truth(self, y_gt: [ObjectDetectionAnnotation], nb_classes: int):
        """学習用の`y_true`の作成。

        IOUが0.5以上のものを全部割り当てる＆割り当らなかったらIOUが一番大きいものに割り当てる。
        1つのprior boxに複数割り当たってしまっていたらIOUが一番大きいものを採用。
        """
        confs = np.zeros((len(y_gt), len(self.prior_boxes), nb_classes), dtype=np.float32)
        locs = np.zeros((len(y_gt), len(self.prior_boxes), 4), dtype=np.float32)
        for i, (_, _, classes, bboxes, _) in enumerate(y_gt):
            allocs = np.zeros((len(self.prior_boxes),), dtype=np.int_)  # 割り当たっている個数のリスト
            iou_list = np.zeros((len(self.prior_boxes), len(bboxes)))
            for bb_ix, bbox in enumerate(bboxes):
                iou = tk.ml.compute_iou(self.prior_boxes, np.expand_dims(bbox, 0))[:, 0]
                matches = iou >= 0.5
                if matches.any():
                    # IOUが0.5以上のものを全部割り当てる
                    allocs[matches] += 1
                    iou_list[matches, bb_ix] = iou[matches]
                else:
                    # 割り当らなかったらIOUが一番大きいものに割り当てる
                    iou_max_ix = iou.argmax()
                    allocs[iou_max_ix] += 1
                    iou_list[iou_max_ix, bb_ix] = 100  # 強制的に割り当てるので最優先
            for pb_ix in allocs.nonzero()[0]:
                assert np.sum(iou_list[pb_ix] == 100) <= 1  # 同じ箇所に2個の割り当てが発生するとつらい
                bb_ix = iou_list[pb_ix].argmax()
                class_id = classes[bb_ix]
                assert 0 < class_id and class_id < self.nb_classes
                # confs: 該当のクラスだけ1にする。
                confs[i, pb_ix, class_id] = 1
                # locs: xmin, ymin, xmax, ymaxそれぞれのoffsetをそのまま回帰する。(TODO: スケールが怪しい)
                locs[i, pb_ix, :] = bboxes[bb_ix] - self.prior_boxes[pb_ix]

        return np.concatenate([confs, locs], axis=-1)

    def decode_predictions(self, predictions, confidence_threshold=0.5):
        """予測結果をデコードする。

        出力は `(conf, class_id, xmin, ymin, xmax, ymax)×検出数×画像数` のような形式。画像毎にconfの降順。
        """
        confs, locs = predictions
        # 0番目(bg)以外でthresholdを超えたindexリストを作る
        max_nonbg_confs = confs[:, :, 1:].max(axis=-1)
        pb_ix_list = np.where(max_nonbg_confs >= confidence_threshold)
        if len(pb_ix_list) <= 0:
            # thresholdを超えたのが1つも無ければとりあえず1個だけ最大のを取り出す。
            pb_ix_list = [max_nonbg_confs.argmax()]
        # conf, locを集めてくっつける
        conf_list = max_nonbg_confs[pb_ix_list]
        class_id_list = confs[pb_ix_list].argmax(axis=-1)
        loc_list = self.prior_boxes[pb_ix_list] + locs[pb_ix_list, :]
        results = np.concatenate([
            np.expand_dims(conf_list, axis=-1),
            np.expand_dims(class_id_list, axis=-1),
            loc_list
        ], axis=1)
        # confの降順にソート
        results.sort(axis=0, order='')
        results = results[::-1, :]
        return results


def _create_model(nb_classes: int, input_shape: tuple, pbox: PriorBoxes):
    import keras

    def _conv(x, *args, name=None, **kargs):
        # x = keras.layers.BatchNormalization(name=name + 'bn')(x)
        x = keras.layers.ELU(name=name + 'act')(x)
        x = keras.layers.Conv2D(*args, use_bias=False, name=name, **kargs)(x)
        return x

    def _conv2(x, nb_filter, name):
        x = _conv(x, nb_filter // 4, (1, 1), name=name + 'sq')
        x = _conv(x, nb_filter, (3, 3), padding='same', name=name + 'ex')
        return x

    def _block(x, nb_filter, name):
        x0 = x
        x1 = x = _conv2(x, nb_filter, name + '_c1')
        x2 = x = _conv2(x, nb_filter, name + '_c2')
        x3 = x = _conv2(x, nb_filter, name + '_c3')
        x = keras.layers.Concatenate()([x0, x1, x2, x3])
        x = _conv(x, nb_filter, (1, 1), name=name + '_sq')
        return x

    def _ds(x, pool_size, strides, name):
        return keras.layers.Concatenate(name=name + '_mixed')([
            keras.layers.MaxPooling2D(pool_size, strides, name=name + '_max')(x),
            keras.layers.AveragePooling2D(pool_size, strides, name=name + '_avg')(x),
        ])

    def _us(x, *args, name=None, **kargs):
        x = keras.layers.BatchNormalization(name=name + 'bn')(x)
        x = keras.layers.ELU(name=name + 'act')(x)
        x = keras.layers.Conv2DTranspose(*args, use_bias=False, name=name, **kargs)(x)
        return x

    net = {}
    x = inputs = keras.layers.Input(input_shape)
    x = keras.layers.Conv2D(32, (3, 3), padding='valid', name='stage1_conv')(x)  # 510
    x = _ds(x, (2, 2), (2, 2), 'stage1_ds')  # 255
    x = _block(x, 64, 'stage2_block')
    x = _ds(x, (3, 3), (2, 2), 'stage2_ds')  # 127
    x = _block(x, 128, 'stage3_block')
    x = _ds(x, (3, 3), (2, 2), 'stage3_ds')  # 63
    net['out63'] = x

    x = _block(x, 256, 'stage4_block')
    x = _ds(x, (3, 3), (2, 2), 'stage4_ds')  # 31
    net['out31'] = x

    x = _block(x, 512, 'stage5_block')
    x = _ds(x, (3, 3), (2, 2), 'stage5_ds')  # 15
    net['out15'] = x

    x = _conv2(x, 512, 'stage6_conv')
    x = _ds(x, (3, 3), (2, 2), 'stage6_ds')  # 7
    net['out7'] = x

    x = _conv2(net['out7'], 512, 'stage7_conv')
    x = _ds(x, (3, 3), (2, 2), 'stage7_ds')  # 3
    net['out3'] = x

    x = _conv2(x, 512, 'stage8_conv')
    x = _ds(x, (3, 3), (1, 1), 'stage8_ds')  # 1
    net['out1'] = x

    x = _us(x, 128, (3, 3), strides=(1, 1), name='stage8u_us')  # 3
    x = keras.layers.Concatenate()([x, net['out3']])
    x = _conv2(x, 512, 'stage8u_conv')
    net['out3'] = x

    x = _us(x, 128, (3, 3), strides=(2, 2), name='stage7u_us')  # 7
    x = keras.layers.Concatenate()([x, net['out7']])
    x = _conv2(x, 512, 'stage7u_conv')
    net['out7'] = x

    x = _us(x, 128, (3, 3), strides=(2, 2), name='stage6u_us')  # 15
    x = keras.layers.Concatenate()([x, net['out15']])
    x = _conv2(x, 512, 'stage6u_conv')
    net['out15'] = x

    x = _us(x, 128, (3, 3), strides=(2, 2), name='stage5u_us')  # 31
    x = keras.layers.Concatenate()([x, net['out31']])
    x = _conv2(x, 512, 'stage5u_conv')
    net['out31'] = x

    x = _us(x, 128, (3, 3), strides=(2, 2), name='stage4u_us')  # 63
    x = keras.layers.Concatenate()([x, net['out63']])
    x = _conv2(x, 512, 'stage4u_conv')
    net['out63'] = x

    confs, locs = [], []
    for c in pbox.config:
        tile_count = c['tile_count']
        priors = len(c['aspect_ratios'])

        x = net['out{}'.format(tile_count)]

        conf = _conv(x, priors * nb_classes, (1, 1), name='out{}conf_{}x{}'.format(tile_count, priors, nb_classes))
        conf = keras.layers.Reshape((-1, nb_classes))(conf)

        # locには全体的な傾向がある気がするのでGlobalAveragePoolingしたやつを足し込むとかしてみたい。
        # (w/hだけとかがよいかも)

        loc = _conv(x, priors * 4, (1, 1), name='out{}loc_{}'.format(tile_count, priors))
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
    return model


class Generator(tk.image.ImageDataGenerator):
    """データの読み込みを行うGenerator。"""

    def __init__(self, data_dir: pathlib.Path, gt: dict, nb_classes: int, pbox: PriorBoxes, input_shape: tuple):
        self.image_dir = str(data_dir.joinpath('VOCdevkit', 'VOC2007', 'JPEGImages'))
        self.gt = gt
        self.nb_classes = nb_classes
        self.pbox = pbox
        super().__init__(image_size=input_shape[:2])

    def _prepare(self, X, y=None, weights=None, **kargs):
        if y is not None:
            y = [self.gt[x] for x in X]
            y = self.pbox.encode_truth(y, self.nb_classes)
        import os
        X = [self.image_dir + os.sep + x + '.jpg' for x in X]
        return super()._prepare(X, y, weights, **kargs)

    def _transform(self, rgb, rand):
        """変形を伴うAugmentation。"""
        # とりあえず殺しておく
        return rgb


def run(logger, result_dir: pathlib.Path, data_dir: pathlib.Path):
    gt, nb_classes = load_gt(data_dir)
    X_train = np.array(tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')))
    X_test = np.array(tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'test.txt')))
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    pbox = PriorBoxes(nb_classes)
    if False:
        pbox.check_prior_boxes(logger, gt)

    import keras
    input_shape = (512, 512, 3)
    model = _create_model(nb_classes, input_shape, pbox)
    model.summary(print_fn=logger.debug)
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

    callbacks = []
    # callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=1e-3))
    # callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
    # callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
    # if K.backend() == 'tensorflow':
    #     callbacks.append(keras.callbacks.TensorBoard())

    gen = Generator(data_dir, gt, nb_classes, pbox, input_shape)
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.125, tk.image.RandomBlur())
    gen.add(0.125, tk.image.RandomBlur(partial=True))
    gen.add(0.125, tk.image.RandomUnsharpMask())
    gen.add(0.125, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.125, tk.image.RandomMedian())
    gen.add(0.125, tk.image.RandomMedian(partial=True))
    gen.add(0.125, tk.image.GaussianNoise())
    gen.add(0.125, tk.image.GaussianNoise(partial=True))
    gen.add(0.125, tk.image.RandomSaturation())
    gen.add(0.125, tk.image.RandomBrightness())
    gen.add(0.125, tk.image.RandomContrast())
    gen.add(0.125, tk.image.RandomLighting())

    model.fit_generator(
        gen.flow(X_train, X_train, batch_size=BATCH_SIZE),
        steps_per_epoch=gen.steps_per_epoch(X_train.shape[0], BATCH_SIZE),
        epochs=MAX_EPOCH,
        validation_data=gen.flow(X_test, X_test, batch_size=BATCH_SIZE),
        validation_steps=gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE),
        callbacks=callbacks)

    model.save(str(result_dir.joinpath('model.h5')))

    pred = model.predict_generator(
        gen.flow(X_test, batch_size=BATCH_SIZE),
        gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE))

    # TODO: mAP


def load_gt(data_dir: pathlib.Path) -> (dict, int):
    """VOC2007のアノテーションデータの読み込み。

    結果は「ファイル名拡張子なし」とObjectDetectionAnnotationのdictと、クラス数。
    """
    import xml.etree

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

    data = {}
    for f in data_dir.joinpath('VOCdevkit', 'VOC2007', 'Annotations').iterdir():
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
