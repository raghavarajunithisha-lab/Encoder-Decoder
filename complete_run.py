import os
import types
import time
import gc
import torch
from train import prepare_and_train
from configs import config_bart, config_lstm, config_gru, config_t4

# ---- GLOBAL SETTINGS ----
# Set MAX_EXAMPLES to limit dataset size for faster CPU runs.
# Set to None for full dataset (recommended only with GPU).
MAX_EXAMPLES = 1000

# List of datasets to iterate through
datasets = [
    "data/preprocessed_MentalChat_df.csv",
    "data/preprocessed_counselchat_data_df.csv",
    "data/preprocessed_mental_health_chatbot.csv",
    "data/preprocessed_nlp_mental_health_df.csv"
]

# Map of model names to their respective config modules
configs = {
    "BART": config_bart,
    "LSTM": config_lstm,
    "GRU": config_gru,
    "T4": config_t4
}

# Column mapping per dataset
COLUMN_MAP = {
    "mental_health_chatbot": ("human_input", "assistant_output"),
    "MentalChat_df":         ("input", "output"),
    "nlp_mental_health":     ("Context", "Response"),
}
DEFAULT_COLUMNS = ("questionText", "answerText")

# Output CSV — written after every single run so results survive crashes
RESULTS_CSV = "enc_dec_results.csv"

def _make_cfg_copy(source_module):
    """Create an independent SimpleNamespace copy of a config module so
    mutations in one run never bleed into the next."""
    ns = types.SimpleNamespace()
    for key in dir(source_module):
        if key.startswith("_"):
            continue
        setattr(ns, key, getattr(source_module, key))
    return ns

def _append_result_to_csv(row: dict):
    """Append a single result row to the CSV immediately after each run."""
    import csv
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def run_experiments():
    total_t0 = time.time()
    all_results = []

    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path)

        # Determine columns once per dataset
        input_col, output_col = DEFAULT_COLUMNS
        for key, cols in COLUMN_MAP.items():
            if key in dataset_name:
                input_col, output_col = cols
                break

        for model_name, cfg_module in configs.items():
            for tda_status in [True, False]:

                # Fresh copy — no cross-contamination
                cfg = _make_cfg_copy(cfg_module)
                cfg.MODEL_NAME = model_name
                cfg.CSV_PATH = dataset_path
                cfg.USE_TDA = tda_status
                cfg.INPUT_COL = input_col
                cfg.OUTPUT_COL = output_col
                cfg.MAX_EXAMPLES = MAX_EXAMPLES

                print("\n" + "=" * 80)
                print(f"STARTING RUN:")
                print(f"  DATASET: {dataset_name}")
                print(f"  MODEL:   {model_name}")
                print(f"  TDA:     {tda_status}")
                print(f"  COLUMNS: {cfg.INPUT_COL} -> {cfg.OUTPUT_COL}")
                print("=" * 80 + "\n")

                t0 = time.time()
                metrics = {}
                try:
                    result = prepare_and_train(cfg)
                    if result:
                        metrics = result
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"!!! ERROR in {model_name} on {dataset_name} (TDA={tda_status}): {e}")
                    metrics = {"test_loss": None, "test_bleu": None, "error": str(e)}
                finally:
                    elapsed = time.time() - t0
                    print(f"\n--- Run finished in {elapsed:.1f}s ---")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Build result row and save immediately
                row = {
                    "model":      model_name,
                    "dataset":    dataset_name,
                    "tda":        tda_status,
                    "test_loss":  metrics.get("test_loss"),
                    "test_bleu":  metrics.get("test_bleu"),
                    "time_s":     round(elapsed, 1),
                }
                all_results.append(row)
                _append_result_to_csv(row)
                print(f"[SAVED] Result appended to {RESULTS_CSV}")

                # --- Validation: flag potential issues ---
                if row["test_loss"] is None:
                    print(f"[⚠ WARNING] test_loss is None for {model_name} on {dataset_name} (TDA={tda_status})")
                if row["test_bleu"] is None:
                    print(f"[⚠ WARNING] test_bleu is None for {model_name} on {dataset_name} (TDA={tda_status})")

                # Check for identical results between TDA=True and TDA=False
                if not tda_status:  # just finished TDA=False, compare with TDA=True
                    prev = [r for r in all_results
                            if r["model"] == model_name and r["dataset"] == dataset_name and r["tda"] == True]
                    if prev:
                        prev_row = prev[-1]
                        if prev_row["test_loss"] == row["test_loss"] and prev_row["test_bleu"] == row["test_bleu"]:
                            print(f"[❌ BUG DETECTED] TDA=True and TDA=False gave IDENTICAL results for {model_name}!")
                            print(f"    TDA=True:  loss={prev_row['test_loss']}, bleu={prev_row['test_bleu']}")
                            print(f"    TDA=False: loss={row['test_loss']}, bleu={row['test_bleu']}")
                        else:
                            print(f"[✅ OK] TDA=True and TDA=False gave DIFFERENT results for {model_name}")
                            print(f"    TDA=True:  loss={prev_row['test_loss']}, bleu={prev_row['test_bleu']}")
                            print(f"    TDA=False: loss={row['test_loss']}, bleu={row['test_bleu']}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE.  Total time: {total_elapsed:.1f}s")
    print(f"{'='*80}")

    # Final summary table
    try:
        import pandas as pd
        df = pd.read_csv(RESULTS_CSV)
        print("\nFINAL RESULTS SUMMARY:")
        print(df.to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    run_experiments()