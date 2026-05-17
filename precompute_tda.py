"""
Pre-compute TDA feature caches for all datasets.
Run this ONCE before complete_run.py to avoid recomputing TDA during training.

After this finishes, complete_run.py will load cached TDA features instantly.
"""
import os, sys, io, time, random, pickle
import numpy as np
from tqdm import tqdm

# Fix Windows encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from utils.data_utils import load_csv
from utils.tda_utils import (
    train_fasttext,
    diagrams_to_landscape_vectors,
    sentence_diagram_from_embeddings,
)
from train import _ensure_tda_matrix

# ---- SETTINGS (must match complete_run.py) ----
MAX_EXAMPLES = 1000
SEED = 42
PCA_DIM = 10
FASTTEXT_DIM = 50
LANDSCAPE_RESOLUTION = 100

DATASETS = [
    ("data/preprocessed_MentalChat_df.csv",            "input",       "output"),
    ("data/preprocessed_counselchat_data_df.csv",       "questionText","answerText"),
    ("data/preprocessed_mental_health_chatbot.csv",     "human_input", "assistant_output"),
    ("data/preprocessed_nlp_mental_health_df.csv",      "Context",     "Response"),
]

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def precompute_tda(csv_path, input_col, output_col):
    # Build cache filename (must match train.py format)
    cache_filename = (
        f"{csv_path}.max{MAX_EXAMPLES}.seed{SEED}.pca{PCA_DIM}.tda.pkl"
        .replace('/', '_').replace('\\', '_')
    )
    cache_path = os.path.join("data", cache_filename)

    if os.path.exists(cache_path):
        print(f"  SKIP - cache already exists: {cache_path}")
        return

    set_seed(SEED)

    # Load and split data (same logic as train.py)
    df = load_csv(csv_path, input_col, output_col, max_examples=MAX_EXAMPLES)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n = len(df)
    val_n = max(1, int(0.1 * n))
    test_n = max(1, int(0.1 * n))
    train_df = df[:-(val_n + test_n)].reset_index(drop=True)
    val_df = df[-(val_n + test_n):-test_n].reset_index(drop=True)
    test_df = df[-test_n:].reset_index(drop=True)

    print(f"  Data: {n} total -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Train FastText on training data only
    tokenized_train = [s.strip().split()[:100] for s in train_df[input_col].astype(str).tolist()]
    print(f"  Training FastText (dim={FASTTEXT_DIM})...")
    ft = train_fasttext(tokenized_train, vec_size=FASTTEXT_DIM)

    # Build embeddings
    def build_embeddings(data):
        embeddings = []
        for s in data:
            tokens = str(s).strip().split()[:100]
            vecs = []
            for tok in tokens:
                try:
                    vecs.append(ft.wv[tok])
                except Exception:
                    vecs.append(np.zeros(ft.vector_size))
            if len(vecs) == 0:
                vecs = np.zeros((1, ft.vector_size))
            embeddings.append(np.array(vecs))
        return embeddings

    # Compute persistence diagrams using thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def process_diagrams(embeddings_list, desc="Diagrams"):
        results = [None] * len(embeddings_list)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(sentence_diagram_from_embeddings, emb): i
                for i, emb in enumerate(embeddings_list)
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"  {desc}"):
                results[futures[f]] = f.result()
        return results

    # Train
    print(f"  Computing train TDA ({len(train_df)} samples)...")
    train_diag = process_diagrams(build_embeddings(train_df[input_col].tolist()), "Train diagrams")
    tda_train_raw, pca_model, scaler = diagrams_to_landscape_vectors(
        train_diag, resolution=LANDSCAPE_RESOLUTION, pca_dim=PCA_DIM
    )

    # Val (transform only, using train's PCA/scaler)
    print(f"  Computing val TDA ({len(val_df)} samples)...")
    val_diag = process_diagrams(build_embeddings(val_df[input_col].tolist()), "Val diagrams")
    tda_val_raw, _, _ = diagrams_to_landscape_vectors(
        val_diag, resolution=LANDSCAPE_RESOLUTION, pca_model=pca_model, scaler=scaler
    )

    # Test
    print(f"  Computing test TDA ({len(test_df)} samples)...")
    test_diag = process_diagrams(build_embeddings(test_df[input_col].tolist()), "Test diagrams")
    tda_test_raw, _, _ = diagrams_to_landscape_vectors(
        test_diag, resolution=LANDSCAPE_RESOLUTION, pca_model=pca_model, scaler=scaler
    )

    # Ensure proper shape
    tda_train = _ensure_tda_matrix(tda_train_raw, PCA_DIM)
    tda_val   = _ensure_tda_matrix(tda_val_raw, PCA_DIM)
    tda_test  = _ensure_tda_matrix(tda_test_raw, PCA_DIM)

    print(f"  Final shapes: train={tda_train.shape}, val={tda_val.shape}, test={tda_test.shape}")

    # Save cache
    os.makedirs("data", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((tda_train, tda_val, tda_test), f)
    print(f"  SAVED: {cache_path}")


if __name__ == "__main__":
    total_t0 = time.time()
    print("=" * 60)
    print("PRE-COMPUTING TDA FEATURES FOR ALL DATASETS")
    print(f"Settings: MAX_EXAMPLES={MAX_EXAMPLES}, SEED={SEED}, PCA={PCA_DIM}")
    print("=" * 60)

    for csv_path, input_col, output_col in DATASETS:
        dataset_name = os.path.basename(csv_path)
        print(f"\n--- {dataset_name} ---")
        t0 = time.time()
        try:
            precompute_tda(csv_path, input_col, output_col)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
        print(f"  Time: {time.time()-t0:.1f}s")

    print(f"\n{'='*60}")
    print(f"ALL DONE in {time.time()-total_t0:.1f}s")
    print(f"{'='*60}")
    print("\nNow run: python complete_run.py")
