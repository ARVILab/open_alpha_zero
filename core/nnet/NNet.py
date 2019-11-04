import os
from abc import ABC, abstractclassmethod

import tensorflow as tf

from core.utils.keras_to_tensorflow import freeze_keras_model_simplified
from tensorflow.python.framework import graph_io
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

import pdb
import os

import numpy as np

import time


class NNet(ABC):
    EXTENSION_KERAS = ".h5"
    EXTENSION_FROZEN = ".pb"

    OUTPUT_NODE_NAME_FROZEN = "output_node"

    def __init__(self, observation_size_x, observation_size_y, observation_size_z, action_size,
                 native_multi_gpu=False, multi_gpu_n=2, horovod_distributed=False, kungfu_distributed=False):

        self.observation_size_x = observation_size_x
        self.observation_size_y = observation_size_y
        self.observation_size_z = observation_size_z
        self.action_size = action_size

        if native_multi_gpu and horovod_distributed or native_multi_gpu and kungfu_distributed:
            assert False, "Native and Horovod multi gpu training " \
                          "modes should not be enabled at the same time!"

        self.multi_gpu = native_multi_gpu
        self.multi_gpu_n = multi_gpu_n

        self.horovod_distributed = horovod_distributed
        self.kungfu_distributed = kungfu_distributed

        self._INPUT_NODE_UNIQUE_IDENTIFIER = ["input_1"]
        self._OUTPUT_NODE_UNIQUE_IDENTIFIER = ["out_prob", "out_value"]

        self.model, self.multi_gpu_model = self.build_model()

        self.model_optimized = False
        self.graph = tf.get_default_graph()
        self.sess = None

    def enable_training_capability(self):
        self.model_optimized = False

    def disable_training_capability(self, temp_dir=None, optimize=True):

        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        input_model_file = temp_dir + "/" + "_temp" + self.EXTENSION_KERAS
        output_model_file = temp_dir + "/" + "_temp" + self.EXTENSION_FROZEN
        self._save_model(input_model_file)

        print("Freezing model ... ")
        freezed_model = freeze_keras_model_simplified(input_model_file, 2, self.OUTPUT_NODE_NAME_FROZEN,
                                                      output_model_file)

        with tf.Graph().as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(output_model_file, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            list_of_tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]

            input_names = [list_of_tensors[0]]
            output_names = [list_of_tensors[-2], list_of_tensors[-1]]

        print("Loading model ... ")
        if optimize:

            optimized_graph = optimize_for_inference(freezed_model, input_names, output_names,
                                                     tf.float32.as_datatype_enum)

            tf.train.write_graph(optimized_graph, temp_dir + "/", "_optimized" + self.EXTENSION_FROZEN, as_text=False)

            self.load(temp_dir + "/" + "_optimized" + self.EXTENSION_FROZEN)
            print("Optimized model loaded successfully ... ")
        else:
            self.load(output_model_file)
            print("Frozen model loaded successfully ... ")

    def save(self, filepath):
        if self.model_optimized or filepath.endswith(self.EXTENSION_FROZEN):
            filepath = filepath if filepath.endswith(self.EXTENSION_FROZEN) else filepath + self.EXTENSION_FROZEN
            folder, name = os.path.split(filepath)
            tf.train.write_graph(self.graph, folder, name, as_text=False)
        else:
            filepath = filepath if filepath.endswith(self.EXTENSION_KERAS) else filepath + self.EXTENSION_KERAS
            self.model.save_weights(filepath)

    def load(self, filepath):
        if filepath.endswith(self.EXTENSION_FROZEN):
            self.model_optimized = True

            if not filepath.endswith(self.EXTENSION_FROZEN):
                filepath = filepath + self.EXTENSION_FROZEN

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = True

            self.graph = tf.Graph()

            with self.graph.as_default():
                od_graph_def = tf.GraphDef()

                with tf.gfile.GFile(filepath, 'rb') as fid:
                    od_graph_def.ParseFromString(fid.read())
                    tf.import_graph_def(od_graph_def, name='')

                list_of_tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]

            self.sess = tf.Session(graph=self.graph, config=config)

            input_names = [list_of_tensors[0]]
            output_names = [list_of_tensors[-2], list_of_tensors[-1]]

            self.input_x = self.graph.get_tensor_by_name(input_names[0] + ':0')
            self.out_prob = self.graph.get_tensor_by_name(output_names[0] + ':0')
            self.out_v = self.graph.get_tensor_by_name(output_names[1] + ':0')

        else:
            filepath = filepath if filepath.endswith(self.EXTENSION_KERAS) else filepath + self.EXTENSION_KERAS

            self.model.load_weights(filepath)

    def train(self, input_boards, target_pis, target_vs, batch_size=2048, epochs=10, verbose=1):
        self.model_optimized = False

        if self.is_multi_gpu():
            model = self.get_multi_gpu_model()
        else:
            model = self.get_model()

        if self.horovod_distributed:
            import horovod.keras as hvd

            callbacks = [
                # Horovod: broadcast initial variable states from rank 0 to all other processes.
                # This is necessary to ensure consistent initialization of all workers when
                # training is started with random weights or restored from a checkpoint.
                hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            ]

            model.fit(x=input_boards,
                      y=[target_pis, target_vs],
                      batch_size=batch_size,
                      callbacks=callbacks,
                      epochs=epochs,
                      verbose=verbose)
        if self.kungfu_distributed:
            from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback

             callbacks = [
                # KungFu: broadcast initial variable states from rank 0 to all other processes.
                # This is necessary to ensure consistent initialization of all workers when
                # training is started with random weights or restored from a checkpoint.
                BroadcastGlobalVariablesCallback(),
            ]

            model.fit(x=input_boards,
                      y=[target_pis, target_vs],
                      batch_size=batch_size,
                      callbacks=callbacks,
                      epochs=epochs,
                      verbose=verbose)
        else:
            model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=batch_size, epochs=epochs, verbose=verbose)

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_multi_gpu_model(self):
        return self.multi_gpu_model

    def is_multi_gpu(self):
        return self.multi_gpu

    def clone(self):
        return NNet(self.observation_size_x, self.observation_size_y, self.observation_size_z, self.action_size,
                    native_multi_gpu=self.native_multi_gpu, multi_gpu_n=self.multi_gpu_n,
                    horovod_distributed=self.horovod_distributed, kungfu_distributed=self.kungfu_distributed)

    def predict(self, observation):
        if self.model_optimized:
            with self.graph.as_default():
                pi, v = self.sess.run(
                    [self.out_prob, self.out_v],
                    feed_dict={self.input_x: observation})

            return pi[0], v[0]
        else:
            with self.graph.as_default():
                self.get_model()._make_predict_function()
                pi, v = self.get_model().predict(observation)

                if np.isscalar(v[0]):
                    return pi[0], v[0]
                else:
                    return pi[0], v[0][0]

    def _save_model(self, filepath):
        filepath = filepath if filepath.endswith(self.EXTENSION_KERAS) else filepath + self.EXTENSION_KERAS
        self.model.save(filepath)

    @abstractclassmethod
    def build_model(self):
        pass
