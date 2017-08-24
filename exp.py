#!/usr/bin/env python
"""Keras実験用コード置き場。"""
import argparse
import os
import pathlib
import time

import better_exceptions
import numpy as np

import keras_exp.cifar10
import keras_exp.cifar100
import keras_exp.mnist
import pytoolkit as tk


def _main():
    import matplotlib as mpl
    mpl.use('Agg')

    better_exceptions.MAX_LENGTH = None

    parser = argparse.ArgumentParser(description='Keras実験用コード置き場')
    parser.add_argument('mode', help='動作モード',
                        nargs='?',
                        default='cifar100',
                        choices=['mnist', 'cifar10', 'cifar100'])
    args = parser.parse_args()

    base_dir = pathlib.Path(os.path.realpath(__file__)).parent
    os.chdir(str(base_dir))
    np.random.seed(1337)  # for reproducibility

    result_dir = base_dir.joinpath('results', args.mode)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = tk.create_tee_logger(result_dir.joinpath('output.log'))

    import keras.backend as K
    K.set_image_dim_ordering('tf')

    with tk.dl.session():
        start_time = time.time()
        if args.mode == 'mnist':
            keras_exp.mnist.run(logger, result_dir)
        elif args.mode == 'cifar10':
            keras_exp.cifar10.run(logger, result_dir)
        elif args.mode == 'cifar100':
            keras_exp.cifar100.run(logger, result_dir)
        else:
            assert False
        elapsed_time = time.time() - start_time

    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


if __name__ == '__main__':
    _main()
