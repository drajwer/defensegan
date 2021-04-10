# Copyright 2018 The Defense-GAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Testing white-box attacks Defense-GAN models. This module is based on MNIST
tutorial of cleverhans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths

import argparse
import cPickle
import logging
import os
import sys

import keras.backend as K
import numpy as np
import tensorflow as tf

from blackbox import dataset_gan_dict, get_cached_gan_data
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval
from models.gan import MnistDefenseGAN, FmnistDefenseDefenseGAN, CelebADefenseGAN
from utils.config import load_config
from utils.gan_defense import model_eval_gan
from utils.misc import ensure_dir
from utils.network_builder import model_a, model_b, model_c, model_d, model_e, model_f
from tflib.inception_score import get_inception_score
import tflib.fid as fid

ds_gan = {
    'mnist': MnistDefenseGAN,
    'f-mnist': FmnistDefenseDefenseGAN,
    'celeba': CelebADefenseGAN,
}
orig_data_paths = {k: 'data/cache/{}_pkl'.format(k) for k in ds_gan.keys()}


def measure_gan(gan, rec_data_path=None, probe_size=10000, calc_real_data_is=True):
    """Based on MNIST tutorial from cleverhans.
    
    Args:
         gan: A `GAN` model.
         rec_data_path: A string to the directory.
    """
    FLAGS = tf.flags.FLAGS

    # Set logging level to see debug information.
    set_log_level(logging.WARNING)
    sess = gan.sess

    # FID init
    stats_path = 'data/fid_stats_celeba.npz' # training set statistics
    inception_path = fid.check_or_download_inception(None) # download inception network

    train_images, train_labels, test_images, test_labels = \
        get_cached_gan_data(gan, False)

    images = train_images[0:probe_size] * 255 # np.concatenate(train_images, test_images)

    # Inception Score for real data
    is_orig_mean, is_orig_stddev = (-1, -1)
    if calc_real_data_is:
        is_orig_mean, is_orig_stddev = get_inception_score(images)
        print('\n[#] Inception Score for original data: mean = %f, stddev = %f\n' % (is_orig_mean, is_orig_stddev))

    rng = np.random.RandomState([11, 24, 1990])
    tf.set_random_seed(11241990)

    # Calculate Inception Score for GAN
    gan.batch_size = probe_size
    generated_images_tensor = gan.generator_fn()
    generated_images = sess.run(generated_images_tensor)
    generated_images = 255*((generated_images + 1) / 2)
    is_gen_mean, is_gen_stddev = get_inception_score(generated_images)
    print('\n[#] Inception Score for generated data: mean = %f, stddev = %f\n'% (is_gen_mean, is_gen_stddev))

    # load precalculated training set statistics
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(generated_images, sess, batch_size=100)

    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print("FID: %s" % fid_value)
    
    return is_gen_mean, is_gen_stddev, is_orig_mean, is_orig_stddev, fid_value


import re


def main(cfg, argv=None):
    FLAGS = tf.app.flags.FLAGS
    GAN = dataset_gan_dict[FLAGS.dataset_name]

    gan = GAN(cfg=cfg, test_mode=True)
    gan.load_generator()
    # Setting test time reconstruction hyper parameters.
    [tr_rr, tr_lr, tr_iters] = [FLAGS.rec_rr, FLAGS.rec_lr, FLAGS.rec_iters]
    if FLAGS.defense_type.lower() != 'none':
        if FLAGS.rec_path and FLAGS.defense_type == 'defense_gan':

            # Extract hyperparameters from reconstruction path.
            if FLAGS.rec_path:
                train_param_re = re.compile('recs_rr(.*)_lr(.*)_iters(.*)')
                [tr_rr, tr_lr, tr_iters] = \
                    train_param_re.findall(FLAGS.rec_path)[0]
                gan.rec_rr = int(tr_rr)
                gan.rec_lr = float(tr_lr)
                gan.rec_iters = int(tr_iters)
        elif FLAGS.defense_type == 'defense_gan':
            assert FLAGS.online_training or not FLAGS.train_on_recs

    if FLAGS.override:
        gan.rec_rr = int(tr_rr)
        gan.rec_lr = float(tr_lr)
        gan.rec_iters = int(tr_iters)

    # Setting the results directory.
    results_dir, result_file_name = _get_results_dir_filename(gan)

    # Result file name. The counter ensures we are not overwriting the
    # results.
    counter = 0
    temp_fp = str(counter) + '_' + result_file_name
    results_dir = os.path.join(results_dir, FLAGS.results_dir)
    temp_final_fp = os.path.join(results_dir, temp_fp)
    while os.path.exists(temp_final_fp):
        counter += 1
        temp_fp = str(counter) + '_' + result_file_name
        temp_final_fp = os.path.join(results_dir, temp_fp)
    result_file_name = temp_fp
    sub_result_path = os.path.join(results_dir, result_file_name)

    accuracies = measure_gan(
        gan, rec_data_path=FLAGS.rec_path, probe_size=FLAGS.probe_size,
        calc_real_data_is=FLAGS.calc_real_data_is)

    ensure_dir(results_dir)

    with open(sub_result_path, 'a') as f:
        f.writelines([str(acc) + ' ' for acc in accuracies])
        f.write('\n')
        print('[*] saved accuracy in {}'.format(sub_result_path))


def _get_results_dir_filename(gan):
    FLAGS = tf.flags.FLAGS

    results_dir = os.path.join('results')
    result_file_name = \
            'Iter{}.txt'.format(
                FLAGS.iter,
            )
    
    return results_dir, result_file_name


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python whitebox.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    flags.DEFINE_integer('nb_classes', 10, 'Number of classes.')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training.')
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model.')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697.')
    flags.DEFINE_float('fgsm_eps', 0.3, 'FGSM epsilon.')
    flags.DEFINE_string('rec_path', None, 'Path to reconstructions.')
    flags.DEFINE_integer('num_tests', -1, 'Number of test samples.')
    flags.DEFINE_integer('random_test_iter', -1,
                         'Number of random sampling for testing the classifier.')
    flags.DEFINE_boolean("online_training", False,
                         "Train the base classifier on reconstructions.")
    flags.DEFINE_string("defense_type", "none", "Type of defense [none|defense_gan|adv_tr]")
    flags.DEFINE_string("attack_type", "none", "Type of attack [fgsm|cw|rand_fgsm]")
    flags.DEFINE_string("results_dir", None, "The final subdirectory of the results.")
    flags.DEFINE_integer("iter", 0, "Iteration identifier output filename.")
    flags.DEFINE_integer("probe_size", 10000,
                         "Size of probes used for calculating GAN performance metrics")
    flags.DEFINE_boolean("calc_real_data_is", False,
                         "Boolean if Inception Score should be calculated for real distribution")
    flags.DEFINE_boolean("same_init", False, "Same initialization for z_hats.")
    flags.DEFINE_string("model", "F", "The classifier model.")
    flags.DEFINE_string("debug_dir", "temp", "The debug directory.")
    flags.DEFINE_integer("num_train", -1, 'Number of training data to load.')
    flags.DEFINE_boolean("debug", False, "True for saving reconstructions [False]")
    flags.DEFINE_boolean("override", False, "Overriding the config values of reconstruction "
                                            "hyperparameters. It has to be true if either "
                                            "`--rec_rr`, `--rec_lr`, or `--rec_iters` is passed "
                                            "from command line.")
    flags.DEFINE_boolean("train_on_recs", False,
                         "Train the classifier on the reconstructed samples "
                         "using Defense-GAN.")

                         

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
