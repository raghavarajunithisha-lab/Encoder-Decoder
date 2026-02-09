import os
import copy
from train import prepare_and_train
from configs import config_bart, config_lstm, config_gru, config_t4
import torch

# List of datasets to iterate through
datasets = [
    "data/preprocessed_MentalChat_df.csv",
    "data/preprocessed_counselchat_data_df.csv",
    "data/preprocessed_mental_health_chatbot.csv",
    "data/preprocessed_nlp_mental_health_df.csv"
]

# Map of model names to their respective config objects
configs = {
    "BART": config_bart,
    "LSTM": config_lstm,
    "GRU": config_gru,
    "T4": config_t4
}

def run_experiments():
    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path)
        
        for model_name, cfg in configs.items():
            for tda_status in [True, False]:
                
                cfg.MODEL_NAME = model_name
                cfg.CSV_PATH = dataset_path
                cfg.USE_TDA = tda_status
                
                # --- FIXED COLUMN MAPPING ---
                if "mental_health_chatbot" in dataset_name:
                    cfg.INPUT_COL = "human_input"
                    cfg.OUTPUT_COL = "assistant_output"
                elif "MentalChat_df" in dataset_name:
                    cfg.INPUT_COL = "input"
                    cfg.OUTPUT_COL = "output"
                elif "nlp_mental_health" in dataset_name:
                    cfg.INPUT_COL = "Context"
                    cfg.OUTPUT_COL = "Response"
                else:
                    # Default for counselchat and others
                    cfg.INPUT_COL = "questionText"
                    cfg.OUTPUT_COL = "answerText"

                print("\n" + "="*80)
                print(f"STARTING RUN:")
                print(f"  DATASET: {dataset_name}")
                print(f"  MODEL:   {model_name}")
                print(f"  TDA:     {tda_status}")
                print(f"  COLUMNS: {cfg.INPUT_COL} -> {cfg.OUTPUT_COL}")
                print("="*80 + "\n")

                try:
                    prepare_and_train(cfg)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"!!! ERROR in {model_name} on {dataset_name}: {e}")

if __name__ == "__main__":
    run_experiments()