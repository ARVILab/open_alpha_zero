import sys

sys.path.append('..')

from core.nnet.NNet import NNet

from keras.models import *
from keras.layers import *
from keras.optimizers import *

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.losses import mean_squared_error
from keras.regularizers import l2

from keras.utils import multi_gpu_model as make_multi_gpu

import kungfu as kf
from keras import backend as K

import keras as keras


class CheckersResNNetKungFuDistributed(NNet):
    learning_rate = 0.0001

    cnn_filter_num = 256
    cnn_first_filter_size = 5
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4
    value_fc_size = 256

    def init_horovod(self):
        # TODO (may be bad for KungFu) pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(kf.current_rank())
        K.set_session(tf.Session(config=config))

    def build_model(self):

        self.init_horovod()

        input_boards = Input(
            shape=(self.observation_size_x, self.observation_size_y, self.observation_size_z))  # s: batch_size x board_x x board_y

        x = Reshape((self.observation_size_x, self.observation_size_y, self.observation_size_z))(
            input_boards)  # batch_size  x board_x x board_y x 1

        # (batch, channels, height, width)
        x = Conv2D(filters=CheckersResNNetKungFuDistributed.cnn_filter_num, kernel_size=CheckersResNNetKungFuDistributed.cnn_first_filter_size,
                   padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(CheckersResNNetKungFuDistributed.l2_reg),
                   name="input_conv-" + str(CheckersResNNetKungFuDistributed.cnn_first_filter_size) + "-" + str(
                       CheckersResNNetKungFuDistributed.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(CheckersResNNetKungFuDistributed.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(CheckersResNNetKungFuDistributed.l2_reg),
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(self.action_size, kernel_regularizer=l2(CheckersResNNetKungFuDistributed.l2_reg), activation="softmax",
                           name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(CheckersResNNetKungFuDistributed.l2_reg),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(CheckersResNNetKungFuDistributed.value_fc_size, kernel_regularizer=l2(CheckersResNNetKungFuDistributed.l2_reg), activation="relu",
                  name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(CheckersResNNetKungFuDistributed.l2_reg), activation="tanh", name="value_out")(x)

        model = Model(input_boards, [policy_out, value_out])

        # KungFu: adjust learning rate based on number of GPUs.=
        opt = keras.optimizers.Adadelta(1.0 * kf.current_cluster_size())

        # KungFu: add KungFu Distributed Optimizer.
        from kungfu.tensorflow.optimizers import SynchronousAveragingOptimizer
        opt = SynchronousAveragingOptimizer(opt)

        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                      optimizer=opt)

        return model, None

    def _build_residual_block(self, x, index):
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=CheckersResNNetKungFuDistributed.cnn_filter_num, kernel_size=CheckersResNNetKungFuDistributed.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(CheckersResNNetKungFuDistributed.l2_reg),
                   name=res_name + "_conv1-" + str(CheckersResNNetKungFuDistributed.cnn_filter_size) + "-" + str(
                       CheckersResNNetKungFuDistributed.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=CheckersResNNetKungFuDistributed.cnn_filter_num, kernel_size=CheckersResNNetKungFuDistributed.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(CheckersResNNetKungFuDistributed.l2_reg),
                   name=res_name + "_conv2-" + str(CheckersResNNetKungFuDistributed.cnn_filter_size) + "-" + str(
                       CheckersResNNetKungFuDistributed.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    def clone(self):
        return CheckersResNNetKungFuDistributed(self.observation_size_x, self.observation_size_y, self.observation_size_z, self.action_size,
                               multi_gpu=self.multi_gpu, multi_gpu_n=self.multi_gpu_n)
