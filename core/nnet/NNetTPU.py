import os
from abc import ABC, abstractclassmethod

from core.nnet.NNet import NNet

import tensorflow as tf

import numpy as np

import time

from tensorflow.python.keras._impl.keras.estimator import _save_first_checkpoint, _clone_and_build_model
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

from distutils.dir_util import copy_tree

import six
from tensorflow.python.estimator.estimator import _check_hooks_type
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.training import saver
from tensorflow.python.training import training
from tensorflow.python.eager import context
from tensorflow.python.training import warm_starting_util
from tensorflow.contrib import predictor

import shutil


class NNetTPU(NNet):
    def __init__(self, observation_size_x, observation_size_y, observation_size_z, action_size):
        super().__init__(observation_size_x, observation_size_y, observation_size_z, action_size)

        if 'TPU_NAME' in os.environ:
            self.tpu = [os.environ['TPU_NAME']]
        else:
            self.tpu = ["demo-tpu"]
        self.tpu_zone = "us-central1-b"
        self.gcp_project = "alpha-zero-arvi"

        self.batch_size = 1024
        self.learning_rate = 0.001
        self.use_tpu = False
        self.iterations_per_loop = 10
        self.num_shards = 8

        self.temp_dir = "temp/tpu_estimator_data"
        os.makedirs(self.temp_dir, exist_ok=True)

        if self.use_tpu:
            print("TPU config: \n"
                  " tpu name: %s \n"
                  " project: %s \n"
                  " tpu_zone: %s" % (self.tpu, self.gcp_project, self.tpu_zone))

            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                tpu=self.tpu,
                zone=self.tpu_zone,
                project=self.gcp_project)

            self.run_config = tpu_config.RunConfig(
                cluster=tpu_cluster_resolver,
                save_checkpoints_secs=1200,
                session_config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True),
                tpu_config=tpu_config.TPUConfig(
                    iterations_per_loop=self.iterations_per_loop,
                    num_shards=self.num_shards),
            )

        self.cur_model_dir = self.temp_dir

        self._build_estimator(self.cur_model_dir)
        self.fast_estimator = None

        self.saved = False

        self.export_dir = None
        self.predict_fn = None

    def enable_training_capability(self):
        pass

    def disable_training_capability(self, temp_dir=None, optimize=True):
        pass

    def save(self, dir_path):
        if not self.saved:
            self.cur_model_dir = dir_path
            self._build_estimator(self.cur_model_dir)
            _save_first_checkpoint(self.get_model(), self.estimator, None, self.get_model().get_weights())
        else:
            copy_tree(self.cur_model_dir, dir_path)
            self.cur_model_dir = dir_path
            self._build_estimator(self.cur_model_dir)

            if self.export_dir:
                shutil.rmtree(self.export_dir)
                self.export_dir = None
                self.predict_fn = None

    def load(self, dir_path):
        self.cur_model_dir = dir_path
        self._build_estimator(self.cur_model_dir)
        self.saved = True

        if self.export_dir:
            shutil.rmtree(self.export_dir)
            self.export_dir = None
            self.predict_fn = None

    def train(self, input_boards, target_pis, target_vs,
              batch_size=None,
              epochs=10,
              verbose=False):

        input_identifiers = self._INPUT_NODE_UNIQUE_IDENTIFIER
        out_identifiers = self._OUTPUT_NODE_UNIQUE_IDENTIFIER

        def input_function(observations,
                           labels_probs=None, labels_values=None,
                           shuffle=False):
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={input_identifiers[0]: observations},
                y={out_identifiers[0]: labels_probs, out_identifiers[1]: labels_values},
                shuffle=shuffle
            )
            return input_fn

        if not self.use_tpu:
            steps = len(input_boards) * epochs
        else:
            steps = len(input_boards)

        tf.logging.set_verbosity(tf.logging.INFO)
        self.estimator.train(input_fn=input_function(input_boards,
                                                     target_pis,
                                                     target_vs,
                                                     True),
                             max_steps=steps)

        if self.export_dir:
            shutil.rmtree(self.export_dir)
            self.export_dir = None
            self.predict_fn = None

        self.saved = True

    def predict(self, observation):
        input_identifiers = self._INPUT_NODE_UNIQUE_IDENTIFIER
        out_identifiers = self._OUTPUT_NODE_UNIQUE_IDENTIFIER

        features_batch = np.array(observation)

        input_shape = [None,
                       self.observation_size_x,
                       self.observation_size_y,
                       self.observation_size_z]

        def serving_input_receiver_fn():
            inputs = {input_identifiers[0]: tf.placeholder(shape=input_shape, dtype=tf.int32)}
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        if not self.export_dir:
            self.export_dir = self.estimator.export_savedmodel(
                export_dir_base=self.estimator.model_dir,
                serving_input_receiver_fn=serving_input_receiver_fn)
            self.predict_fn = predictor.from_saved_model(self.export_dir)

        prediction = self.predict_fn(
            {input_identifiers[0]: features_batch})

        policy = prediction[out_identifiers[0]][0]
        value = prediction[out_identifiers[1]][0][0]

        return policy, value

    def _build_estimator(self, dir_path):
        if self.use_tpu:
            self.estimator = self._model_to_tpu_estimator(self.get_model(),
                                                          model_dir=dir_path,
                                                          config=self.run_config)
        else:
            self.estimator = tf.keras.estimator.model_to_estimator(self.get_model(),
                                                                   model_dir=dir_path)

    def _model_to_tpu_estimator(self, keras_model,
                                keras_model_path=None,
                                custom_objects=None,
                                model_dir=None,
                                config=None):
        keras_model_fn = self._create_keras_tpu_model_fn(keras_model, custom_objects)
        est = tf.contrib.tpu.TPUEstimator(keras_model_fn,
                                          model_dir=model_dir,
                                          config=config,
                                          use_tpu=self.use_tpu,
                                          train_batch_size=self.batch_size)
        return est

    def _create_keras_tpu_model_fn(self, keras_model, custom_objects=None):

        def model_fn(features, labels, mode):
            """model_fn for keras Estimator."""
            model = _clone_and_build_model(mode, keras_model, custom_objects, features,
                                           labels)
            predictions = dict(zip(model.output_names, model.outputs))

            loss = None
            train_op = None
            eval_metric_ops = None

            # Set loss and metric only during train and evaluate.
            if mode is not tf.estimator.ModeKeys.PREDICT:
                model.optimizer.optimizer = tf.contrib.tpu.CrossShardOptimizer(model.optimizer.optimizer)

                model._make_train_function()  # pylint: disable=protected-access
                loss = model.total_loss

                if model.metrics:
                    eval_metric_ops = {}
                    # When each metric maps to an output
                    if isinstance(model.metrics, dict):
                        for i, output_name in enumerate(model.metrics.keys()):
                            metric_name = model.metrics[output_name]
                            if callable(metric_name):
                                metric_name = metric_name.__name__
                            # When some outputs use the same metric
                            if list(model.metrics.values()).count(metric_name) > 1:
                                metric_name += '_' + output_name
                            eval_metric_ops[metric_name] = tf.metrics.mean(
                                model.metrics_tensors[i - len(model.metrics)])
                    else:
                        for i, metric_name in enumerate(model.metrics):
                            if callable(metric_name):
                                metric_name = metric_name.__name__
                            eval_metric_ops[metric_name] = tf.metrics.mean(
                                model.metrics_tensors[i])

            if mode is tf.estimator.ModeKeys.TRAIN:
                train_op = model.train_function.updates_op

            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops)

        return model_fn
