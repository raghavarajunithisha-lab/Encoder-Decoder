# main.py
# Minimal launcher. Edit below to choose which config to load.

# from configs import config_bart as cfg
# from configs import config_lstm as cfg
# from configs import config_gru as cfg
from configs import config_t4 as cfg

from train import prepare_and_train


if __name__ == "__main__":
    # Pass the config object directly to training
    prepare_and_train(cfg)
