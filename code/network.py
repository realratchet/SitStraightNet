from model import Model
from easydict import EasyDict
from utils import config as cfg
import tensorflow.keras.optimizers as optimizers


class Network:
    def __init__(self, *, res_image, man_labels, checkpoint=None):
        self.net = Model(res_image)
    
    def predict(self, inputs):
        return self.net.model.predict(inputs)

    def train(self, *, dat_train, dat_test, initial_learning_rate: float, batch_size: int, epochs: int):
        optimizer = optimizers.Adam(learning_rate=initial_learning_rate)
        model = self.net.model

        for epoch in range(0, epochs):
            result = model.fit(dat_train, epochs=1)
            test_loss, test_accuracy = model.evaluate(dat_test)