import pathlib
import h5py
import numpy as np
from tqdm import tqdm


def read_h5_numpy(path: pathlib.Path) -> np.ndarray:
    all_embeddings = []
    all_ids = []
    with h5py.File(path, "r") as f:
        for id_ in tqdm(sorted(f.keys())):
            emb = f[id_][:]  # load tensor in numpy format
            all_embeddings.append(emb)
            all_ids.append(id_)
    
    return all_embeddings, all_ids