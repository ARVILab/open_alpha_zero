import sys
sys.path.append('../')

# Hello there!

from core.nnet.NNet import NNet

from keras.models import *
from keras.layers import *
from keras.layers import concatenate
from keras.optimizers import *
import keras

from keras.utils import multi_gpu_model as make_multi_gpu

class TicTacToeNNet(NNet):

    def build_model(self):

        learning_rate = 0.01

        input_boards = Input(shape=(self.observation_size_x, self.observation_size_y, self.observation_size_z))  # s: batch_size x board_x x board_y

        x_image = Reshape((self.observation_size_x, self.observation_size_y, self.observation_size_z))(
            input_boards)  # batch_size  x board_x x board_y x 1

        h_conv1_1 = (BatchNormalization(axis=1)(Conv2D(512, 1, padding='same', activation='relu')(x_image)))
        h_conv1_3 = (BatchNormalization(axis=1)(Conv2D(512, 3, padding='same', activation='relu')(x_image)))
        h_conv1_5 = (BatchNormalization(axis=1)(Conv2D(512, 5, padding='same', activation='relu')(x_image)))

        merged_h_conv1 = keras.layers.concatenate([h_conv1_1, h_conv1_3, h_conv1_5], axis=2)
        merged_h_conv1 = Flatten()(merged_h_conv1)

        h_fc1 = Dropout(0.3)(BatchNormalization(axis=1)(Dense(1024, activation='relu')(merged_h_conv1)))
        h_fc2 = Dropout(0.3)(BatchNormalization(axis=1)(Dense(512, activation='relu')(h_fc1)))

        pi = Dense(self.action_size, activation='softmax', name='pi')(h_fc2)  # batch_size x self.action_size
        v = Dense(1, activation='tanh', name='v')(h_fc2)  # batch_size x 1

        model = Model(inputs=input_boards, outputs=[pi, v])

        if self.multi_gpu:
            _multi_gpu_model = make_multi_gpu(self.model, gpus=self.multi_gpu_n)
            _multi_gpu_model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                                    optimizer=Adam(learning_rate))
        else:
            _multi_gpu_model = None

        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate))

        return model, _multi_gpu_model

    def clone(self):
        return TicTacToeNNet(self.observation_size_x, self.observation_size_y, self.observation_size_z, self.action_size,
                            multi_gpu=self.multi_gpu, multi_gpu_n=self.multi_gpu_n)
