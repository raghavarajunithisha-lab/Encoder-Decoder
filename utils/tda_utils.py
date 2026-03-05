# utils/tda_utils.py
# Reusable TDA utilities. These are optional and only required when USE_TDA=True.

import numpy as np

def safe_imports():
    # Lazy imports so package isn't required unless USE_TDA is True
    import gudhi
    from gudhi.representations import DiagramSelector, DiagramScaler, Clamping, Landscape
    from gensim.models import FastText
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import nltk
    from nltk.tokenize import word_tokenize

    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        nltk.download("punkt", quiet=True)

    return {
        "gudhi": gudhi,
        "DiagramSelector": DiagramSelector,
        "DiagramScaler": DiagramScaler,
        "Clamping": Clamping,
        "Landscape": Landscape,
        "FastText": FastText,
        "PCA": PCA,
        "StandardScaler": StandardScaler,
        "word_tokenize": word_tokenize,
    }


# -------------------------
# FASTTEXT
# -------------------------
def train_fasttext(tokenized_sentences, vec_size=100, epochs=10):
    libs = safe_imports()
    FastText = libs["FastText"]

    model = FastText(
        sentences=tokenized_sentences,
        vector_size=vec_size,
        window=5,
        min_count=1,
        workers=4,
        epochs=epochs,
        bucket=50000,
    )
    return model


# -------------------------
# PERSISTENCE DIAGRAM
# -------------------------
def sentence_diagram_from_embeddings(embeddings, quantile=0.9, max_tokens=100):
    # embeddings: (n_tokens, dim)
    if embeddings.size == 0:
        return np.zeros((0, 2))

    # Cap number of tokens to prevent OOM in Gudhi for very long texts
    if embeddings.shape[0] > max_tokens:
        embeddings = embeddings[:max_tokens]

    try:
        dists = np.linalg.norm(
            embeddings[:, None, :] - embeddings[None, :, :], axis=-1
        )
        flat = dists[np.triu_indices_from(dists, k=1)]
        max_edge = np.quantile(flat, quantile) if flat.size > 0 else 0.1
    except Exception:
        max_edge = 0.1

    libs = safe_imports()
    gudhi = libs["gudhi"]

    try:
        rips = gudhi.RipsComplex(points=embeddings, max_edge_length=float(max_edge))
        st = rips.create_simplex_tree(max_dimension=2)
        diag = st.persistence()

        if not diag:
            return np.zeros((0, 2))

        return np.array([p[1] for p in diag])
    except Exception:
        return np.zeros((0, 2))


# -------------------------
# LANDSCAPE + PCA
# -------------------------
def diagrams_to_landscape_vectors(
    diagrams,
    resolution=100,
    pca_dim=10,
    pca_model=None,
    scaler=None,
    texts=None
):
    libs = safe_imports()
    DiagramSelector = libs["DiagramSelector"]
    DiagramScaler = libs["DiagramScaler"]
    Clamping = libs["Clamping"]
    Landscape = libs["Landscape"]
    PCA = libs["PCA"]
    StandardScaler = libs["StandardScaler"]

    selector = DiagramSelector(use=True, point_type="finite")
    scaler_diag = DiagramScaler()
    clamper = Clamping()
    landscape = Landscape(resolution=resolution)

    import warnings

    vectors = []
    
    # We first try to get an exact length for zero arrays based on a non-empty diagram.
    # The length depends on landscape configuration, scaling, etc.
    # Often it is (number_of_landscapes * resolution) which gudhi dynamically creates.
    # If all fail, we will fallback to resolution and hope the downstream handles it.
    target_length = None
    for diag in diagrams:
        if len(diag) > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    D = selector.fit_transform([diag])
                    D = scaler_diag.fit_transform(D)
                    D = clamper.fit_transform(D)
                    L = landscape.fit_transform(D)[0]
                    target_length = len(np.array(L).flatten())
                    break
            except Exception:
                pass
    
    if target_length is None:
        target_length = resolution

    for idx, diag in enumerate(diagrams):
        if len(diag) == 0:
            if texts is not None and idx < len(texts):
                print(f"Empty diagram for: {repr(texts[idx])}")
            vectors.append(np.zeros(target_length))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                D = selector.fit_transform([diag])
                D = scaler_diag.fit_transform(D)
                D = clamper.fit_transform(D)
                L = landscape.fit_transform(D)[0]
                
                arr = np.array(L).flatten()
                if len(arr) != target_length:
                    arr = pad_or_truncate(arr, target_length)
                
                vectors.append(arr)
        except Exception:
            vectors.append(np.zeros(target_length))

    tda_features = np.vstack(vectors).astype(float)

    # PCA
    if pca_model is None:
        pca_model = PCA(n_components=min(pca_dim, tda_features.shape[1]))
        tda_pca = pca_model.fit_transform(tda_features)
    else:
        tda_pca = pca_model.transform(tda_features)

    # Scaling
    if scaler is None:
        scaler = StandardScaler()
        tda_scaled = scaler.fit_transform(tda_pca)
    else:
        tda_scaled = scaler.transform(tda_pca)

    return tda_scaled, pca_model, scaler

def pad_or_truncate(vec, target_dim):
    vec = np.asarray(vec).flatten()
    if len(vec) > target_dim:
        return vec[:target_dim]
    if len(vec) < target_dim:
        return np.concatenate([vec, np.zeros(target_dim - len(vec))])
    return vec
