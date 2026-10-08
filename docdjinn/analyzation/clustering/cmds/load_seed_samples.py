from __future__ import annotations

from pathlib import Path
import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn import ENV
from docdjinn.analyzation.clustering.core._embeddings import (
    _load_sample_ids_from_embeddings,
)
from docdjinn.analyzation.clustering.core._utilities import (
    EmbeddingType,
    _get_clustering_output_path,
)
from docdjinn.logging import get_logger


logger = get_logger(__name__)


def main(cfg: LoadSeedSamples):
    import pandas as pd
    from docdjinn.data import load_dataset

    for embedding_type in EmbeddingType.__members__.values():
        output_dir = Path(cfg.output_dir) / cfg.dataset_name / embedding_type.value
        embeddings_path = (
            Path(cfg.embeddings_dir) / cfg.dataset_name / f"{embedding_type.value}.h5"
        )
        sample_ids = _load_sample_ids_from_embeddings(embeddings_path)
        clusters_path = _get_clustering_output_path(
            output_dir=output_dir,
            intermediate_num_dims=cfg.intermediate_num_dims,
            hdbscan_min_cluster_size=cfg.hdbscan_min_cluster_size,
            hdbscan_metric=cfg.hdbscan_metric,
            k_nn_n_neighbors=cfg.k_nn_n_neighbors,
            do_knn=cfg.do_knn,
            method=cfg.method,
        )
        seeds_output_path = clusters_path.parent / clusters_path.name.replace(
            ".npy", f"_strategy={cfg.sampling_strategy}_seeds.csv"
        )

        # load the sampled seeds
        dataset = load_dataset(cfg.dataset_name, split="train")
        seed_sample_indices = pd.read_csv(seeds_output_path)
        for _, row in seed_sample_indices.iterrows():
            # get seed samples from first row
            sampled_seeds = row.tolist()
            seed_sample_ids = [sample_ids[int(i)] for i in sampled_seeds]
            samples = [dataset.train.get_by_id(sid) for sid in seed_sample_ids]
            print(f"Loaded {len(samples)} seed samples from {seeds_output_path}")
            print("Example sample: ", samples[0])
            break


class LoadSeedSamples(pydantic.BaseModel):
    # same as clustering config
    dataset_name: str
    seed: int = 42
    hdbscan_min_cluster_size: int = 10
    intermediate_num_dims: int = 100
    hdbscan_metric: str = "euclidean"
    do_knn: bool = True
    k_nn_n_neighbors: int = 5
    embeddings_dir: str | Path = ENV.EMBEDDINGS_DIR
    output_dir: str | Path = ENV.CLUSTERS_DIR
    method: str = "hdbscan"  # or "kmeans"
    sampling_strategy: str = "uniform_cluster_sampling"


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=LoadSeedSamples,
    )
    main(parser.parse_typed_args())
