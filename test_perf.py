import time
from configs import config_lstm as cfg

cfg.CSV_PATH = "e:/TDA/code/Encoder-Decoder/data/preprocessed_mental_health_chatbot.csv"
cfg.USE_TDA = True
cfg.MAX_EXAMPLES = 100

from train import prepare_and_train

t0 = time.time()
prepare_and_train(cfg)
print(f"Total time: {time.time() - t0:.2f}s")
