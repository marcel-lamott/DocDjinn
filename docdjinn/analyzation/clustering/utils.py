

import h5py
import numpy as np
from tqdm import tqdm

from docdjinn import ENV


def read_embeddings_numpy(dataset_name: str, embeddings_type: str, kernel_size: int = None) -> np.ndarray:
    all_embeddings = []
    fname = f'{dataset_name}_{embeddings_type}'
    if embeddings_type == 'paper':
        fname += f'_kernel={kernel_size}'
    
    fpath = ENV.EMBEDDINGS_DIR / f'{fname}.h5'
    with h5py.File(fpath, "r") as f:
        for id_ in tqdm(sorted(f.keys())):
            emb = f[id_][:]  # load tensor in numpy format
            all_embeddings.append(emb)

    # Vertically stack along the first dimension
    X = np.vstack(all_embeddings)
    return X