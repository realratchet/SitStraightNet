from data import dataset
from utils import config as cfg
from network import Network

data_training, data_testing = get_dataset(cfg.DIRS.INPUTS, cfg.TRAINING.DATA_SPLOT_RATIO)

network = Network(res_image=cfg.RESOLUTION.IMAGE)

network.train(
    dat_train=data_training, dat_test=data_testing,
    initial_learning_rate=cfg.learning_rate,
    batch_size=cfg.batch_size,
    epochs=cfg.epochs
)
