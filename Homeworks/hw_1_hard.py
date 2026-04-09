import numpy as np
from collections import Counter

def train_word2vec(data: str) -> dict:
    words = data.strip().split()
    if not words:
        return {}

    vocab_counts = Counter(words)
    vocab = list(vocab_counts.keys())
    vocab_size = len(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}
    indices = np.array([word2idx[w] for w in words], dtype=np.int32)
    n = len(indices)

    vector_size = 100
    window = 5

    row_list, col_list, val_list = [], [], []

    for offset in range(1, window + 1):
        weight = 1.0 / offset

        row_list.append(indices[:-offset])
        col_list.append(indices[offset:])
        val_list.append(np.full(n - offset, weight, dtype=np.float32))

        row_list.append(indices[offset:])
        col_list.append(indices[:-offset])
        val_list.append(np.full(n - offset, weight, dtype=np.float32))

    all_rows = np.concatenate(row_list)
    all_cols = np.concatenate(col_list)
    all_vals = np.concatenate(val_list)

    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import svds

        cooc = coo_matrix(
            (all_vals, (all_rows, all_cols)),
            shape=(vocab_size, vocab_size)
        ).tocsr()

        total = float(cooc.sum())
        row_sums = np.asarray(cooc.sum(axis=1)).ravel() + 1e-10
        col_sums = np.asarray(cooc.sum(axis=0)).ravel() + 1e-10

        coo = cooc.tocoo()
        r, c, v = coo.row, coo.col, coo.data.astype(np.float64)

        pmi = np.log(v * total / (row_sums[r] * col_sums[c]))
        ppmi = np.maximum(0.0, pmi).astype(np.float32)

        ppmi_mat = coo_matrix(
            (ppmi, (r, c)),
            shape=(vocab_size, vocab_size)
        ).tocsr()

        k = min(vector_size, vocab_size - 2, 300)
        U, S, Vt = svds(ppmi_mat, k=k)

        order = np.argsort(S)[::-1]
        U, S = U[:, order], S[order]

        embeddings = (U * np.sqrt(np.maximum(S, 0))).astype(np.float32)

        if embeddings.shape[1] < vector_size:
            pad = np.zeros((vocab_size, vector_size - embeddings.shape[1]), dtype=np.float32)
            embeddings = np.hstack([embeddings, pad])

    except ImportError:
        cooc_dense = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        np.add.at(cooc_dense, (all_rows, all_cols), all_vals)

        total = cooc_dense.sum() + 1e-10
        row_sums = cooc_dense.sum(axis=1, keepdims=True) + 1e-10
        col_sums = cooc_dense.sum(axis=0, keepdims=True) + 1e-10

        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(cooc_dense * total / (row_sums * col_sums) + 1e-10)
        ppmi = np.maximum(0, pmi)

        U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
        k = min(vector_size, len(S))
        embeddings = (U[:, :k] * np.sqrt(S[:k])).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    embeddings = embeddings / norms

    return {word: embeddings[word2idx[word]] for word in vocab}