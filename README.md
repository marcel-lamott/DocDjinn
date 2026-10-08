# DocDjinn

Code for the paper **[DocDjinn: Controllable Synthetic Document Generation with VLMs and Handwriting Diffusion](https://doi.org/10.1007/s10032-026-00604-7)** (IJDAR 2026, [arXiv:2602.21824](https://arxiv.org/abs/2602.21824)).

DocDjinn generates annotated synthetic documents with a VLM. Seed images are sampled from clusters of a real dataset, the VLM (Claude) writes new documents as HTML together with their ground truth, and these are rendered to PDF and enriched with diffusion-generated handwriting and visual elements. This repository contains the generation pipeline and the code to train and evaluate models on the generated datasets.

## Data

| File | Content | Size | SHA-256 |
|---|---|---|---|
| [docdjinn_data.tar](https://www.cs.hs-rm.de/forschungsdaten/lamott/docdjinn/docdjinn_data.tar) | Synthetic datasets | 112 G | `b57eaa3ad9757dd9079c6cc80af4c042538f796e4e8bf8f2ee30d728d9510223` |
| [document_hw_model.tar](https://www.cs.hs-rm.de/forschungsdaten/lamott/docdjinn/document_hw_model.tar) | Handwriting diffusion model files; see the included README.md for usage | 4,6G | `282d3f90acf3612e31ed98b5262086057a0a485336b9ab9e76cd4d8d47640464` |
| [document_hw.jsonl](https://www.cs.hs-rm.de/forschungsdaten/lamott/docdjinn/document_hw.jsonl) | Reference file for the DocVQA-HW evaluation | 56K | `b67c2515a0cf18e12f29928ffe3c0886bf988ab8a7a64a44caa9fa9f8e956c17` |

Verify a download with `sha256sum <file>` and compare the output with the table.

`document_hw.jsonl` restricts the DocVQA test split to the 103 documents and 277 questions that concern handwritten content. After preparing the base datasets (see below), place it in `data/datasets/base_v2/due_benchmark/datasets/DocVQA/aws_neurips_time/DocVQA/test/`.

## Setup

Linux x86-64, Python 3.11.12, an NVIDIA GPU (PyTorch is pinned to CUDA 12.1), [uv](https://docs.astral.sh/uv/) and `poppler-utils` (the system package behind `pdf2image`, which renders the page images).

```bash
uv sync
source .venv/bin/activate
playwright install chromium
```

The real datasets are prepared with [docdjinn-base-datasets](https://github.com/saifullah3396/docdjinn-base-datasets). Give it this repository's `data/datasets/base_v2/` as the output directory:

```bash
# in a checkout of docdjinn-base-datasets; its README covers preparing all datasets at once
uv run python -m atria_datasets.prepare_dataset cord/default /path/to/docdjinn/data/datasets/base_v2
```

Dataset names used here (e.g. `ex_docvqa`) are mapped to its `<dataset>/<config>` entries (e.g. `due_benchmark/ExDocVQA`) in `docdjinn/data/constants.py`.

Also required to run the full pipeline, and not included:

- `ANTHROPIC_API_KEY` for Claude (Message Batches API).
- The handwriting diffusion model files in `data/models/handwriting/`. See [Data](#data).
- An OCR service, which is not part of this repository. Step 15 sends page images to `http://localhost:$DOCDJINN_OCR_PORT/v1/sync/ocr/microsoft_di` (default port `8000`), i.e. Microsoft Document Intelligence behind a local HTTP service. It is only called for documents that contain handwriting or visual elements; results already in the dataset's `ocr_results/` are reused, and all other documents take their bounding boxes from the PDF.

## Usage

Run from the repository root.

```bash
# 1. Embed and cluster a real dataset; seed images are sampled from the clusters.
#    --hdbscan-min-cluster-size (default 10) must match the dataset definition.
python docdjinn/analyzation/clustering/cmds/generate_embeddings.py --dataset-name cord
python docdjinn/analyzation/clustering/cmds/generate_clusters.py --dataset-name cord

# 2. Generate data/syn_dataset_definitions/cord_alpha=1.0.yaml
#    into data/datasets/synthesized_datasets/cord_alpha=1.0/
python docdjinn/generation/main.py cord_alpha=1.0

# 3. Convert the result for training.
python docdjinn/data/cmds/prepare_synth_datasets.py --dataset-name cord_alpha=1.0

# 4. Train and evaluate:
#    <real dataset> <synthetic dataset> <model> <seed> <#real samples> <#synthetic samples>, -1 = all
bash scripts/experiments/03_mixed_training/train_entity_labeling.sh cord cord_alpha=1.0 layoutlmv3 42 100 -1
```

`main.py` runs the 19 steps `docdjinn/generation/pipeline_01_*.py` to `pipeline_19_*.py` in order. `--entry N` resumes at step `N`; `--reset` clears earlier outputs but keeps the VLM responses, seed images and OCR results.

`scripts/` holds the remaining experiments; `scripts/experiments/schedule_jobs.py` distributes a job set over GPUs. `data/docvqa_hw/` lists the documents and question IDs of DocVQA-HW, the handwriting subset of DocVQA introduced in the paper.

## Known issue: seed logs in the released datasets

Each dataset records which seed images a prompt received in `logs/prompt_batches/<batch_id>.json`, under `message_id_to_seed_docids`. A logging bug, fixed in this code, stored the seed lists of the **whole batch** under every message ID instead of that message's own list. Only the log is wrong: every prompt was sent the correct seeds, so the documents are unaffected.

In an affected log the entry is a list of lists. The seeds of a message are the element at that message's position in `message_ids`:

```python
seeds = log["message_id_to_seed_docids"][message_id]
if isinstance(seeds[0], list):  # affected log
    seeds = seeds[log["message_ids"].index(message_id)]
```

A document named `<message_id>_<i>` was generated by the message `<message_id>`.

## License

Copyright (c) 2026 RheinMain University of Applied Sciences, German Research
Center for Artificial Intelligence (DFKI), DeepReader GmbH, Insiders
Technologies GmbH, National University of Sciences and Technology (NUST).

The code and the synthetic datasets are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0); see
`LICENSE`.

## Contact

For questions and problems, please open an issue in this repository.

For commercial use of the code or the data, contact Marcel Lamott
(marcel.lamott@hs-rm.de), RheinMain University of Applied Sciences.

## Citation

```bibtex
@article{lamott2026docdjinn,
  title   = {DocDjinn: Controllable Synthetic Document Generation with VLMs and Handwriting Diffusion},
  author  = {Lamott, Marcel and Saifullah, Saifullah and Riaz, Nauman and Weweler, Yves-Noel and Alt-Veit, Tobias and Ali, Ahmad Sarmad and Shakir, Muhammad Armaghan and Kalwa, Adrian and Moetesum, Momina and Dengel, Andreas and Ahmed, Sheraz and Shafait, Faisal and Schwanecke, Ulrich and Ulges, Adrian},
  journal = {International Journal on Document Analysis and Recognition (IJDAR)},
  year    = {2026},
  doi     = {10.1007/s10032-026-00604-7}
}
```
