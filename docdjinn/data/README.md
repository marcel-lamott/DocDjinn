# DocDjinn
## Setup environment
```bash
uv sync
source .venv/bin/activate
```

## Run visualizations scripts for datasets for sanity check
```bash
# classification
uv run python docdjinn/data/cmds/visualize.py --dataset-name tobacco3482
uv run python docdjinn/data/cmds/visualize.py --dataset-name rvlcdip

# entity labeling
uv run python docdjinn/data/cmds/visualize.py --dataset-name cord
uv run python docdjinn/data/cmds/visualize.py --dataset-name sroie
uv run python docdjinn/data/cmds/visualize.py --dataset-name funsd
uv run python docdjinn/data/cmds/visualize.py --dataset-name wild_receipts
uv run python docdjinn/data/cmds/visualize.py --dataset-name docile

# extractive qa
uv run python docdjinn/data/cmds/visualize.py --dataset-name ex_docvqa  # avg pages ~1
uv run python docdjinn/data/cmds/visualize.py --dataset-name ex_deepform  # avg pages ~5
uv run python docdjinn/data/cmds/visualize.py --dataset-name ex_tabfact  # avg pages ~1
uv run python docdjinn/data/cmds/visualize.py --dataset-name ex_wiki  # avg pages ~1
uv run python docdjinn/data/cmds/visualize.py --dataset-name ex_infographics  # avg pages ~1
uv run python docdjinn/data/cmds/visualize.py --dataset-name ex_klc  # avg pages ~23
```

## How to load a specific dataset without transforms
This script assumes that datasets are already prepared in `data/datasets/base_v2/` in msgpack format 
The dataset preparation itself is managed using a separate atria_datasets library ([docdjinn-base-datasets](https://github.com/saifullah3396/docdjinn-base-datasets)).
To keep docdjinn code clean the two are separated.
```python
from docdjinn.data import load_dataset
dataset = load_dataset(dataset_name)

# read samples or use dataset.train[0]
train_dataset = dataset.train # could be None, check for actual use
for sample in dataset.train:
    print("Sample: ", sample)

validation_dataset = dataset.validation # could be None, check for actual use
for sample in dataset.validation:
    print("Sample: ", sample)

test_dataset = dataset.test # could be None, check for actual use
for sample in dataset.test:
    print("Sample: ", sample)
```

## How to load a specific dataset with task-specific transforms
This script assumes that datasets are already prepared in `data/datasets/base_v2/` in msgpack format 
The dataset preparation itself is managed using a separate atria_datasets library ([docdjinn-base-datasets](https://github.com/saifullah3396/docdjinn-base-datasets)).
To keep docdjinn code clean the two are separated.
```python
from docdjinn.data import load_data_pipeline

# load sequence classification dataset pipeline 
data_pipeline = load_data_pipeline(
    dataset_name=dataset_name, 
)

# load tokenized batch from train dataloader 
for batch in data_pipeline.train_dataloader:
    print(batch)

# load tokenized batch from validation dataloader 
for batch in data_pipeline.validation_dataloader:
    print(batch)

# load tokenized batch from test dataloader 
for batch in data_pipeline.test_dataloader:
    print(batch)
```
