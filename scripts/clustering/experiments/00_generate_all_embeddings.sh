declare -a datasets=(
    # classification datasets
    tobacco3482
    rvlcdip
    # entity labeling datasets
    cord
    funsd
    sroie
    # extractive QA,
    ex_docvqa
    ex_wiki
    ex_klc
    # layout analysis
    publaynet
    doclaynet_4k
    icdar2019
)

for dataset in "${datasets[@]}"; do
    uv run docdjinn/analyzation/clustering/cmds/generate_embeddings.py --dataset-name ${dataset}
done
