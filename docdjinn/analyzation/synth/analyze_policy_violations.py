import argparse
from collections import Counter
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from docdjinn import ENV
from docdjinn.generation.models._file import SyntheticDatasetFileStructure
from docdjinn.generation.models._syndatadef import SynDatasetDefinition


def parse_args():
    parser = argparse.ArgumentParser(
        description="DocDjinn Synthetic Document Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "SynDatasetDefinition",
        type=str,
        help="Filename without extension of the SynDatasetDefinition in data/syn_dataset_definitions",
    )

    args = parser.parse_args()
    assert args.SynDatasetDefinition
    print(args)
    return args


def search_for_refusals(dsname):
    deffile = ENV.SYN_DATA_DEFINITIONS_DIR / f"{dsname}.yaml"
    dsdef: SynDatasetDefinition = SynDatasetDefinition.from_file(deffile)
    dsfiles: SyntheticDatasetFileStructure = dsdef.get_file_structure()

    msg_ids_from_all_batches = set()
    msg_id_to_batch_id = dict()
    for prompt_batch_log_path in dsfiles.prompt_batches_directory.iterdir():
        prompt_batch_log = json.loads(prompt_batch_log_path.read_text(encoding="utf-8"))
        for msg_id in prompt_batch_log["message_ids"]:
            msg_ids_from_all_batches.add(msg_id)
            msg_id_to_batch_id[msg_id] = prompt_batch_log["id"]

    missing_message_results = set()
    refusals = set()
    for msg_id in msg_ids_from_all_batches:
        # Look for missing message results, as previously we didn't save them
        msg_res_path = dsfiles.message_results_directory / f"{msg_id}.json"
        if not msg_res_path.exists():
            missing_message_results.add(msg_id)
        else:
            msg_res = json.loads(msg_res_path.read_text(encoding="utf-8"))
            if msg_res["error"] == "refusal":
                refusals.add(msg_id)

    # Search seed images
    all_refusals = missing_message_results | refusals
    prompt_batch_log_lookup = dict()
    problematic_seeds = list()
    for msg_id in all_refusals:
        batch_id = msg_id_to_batch_id[msg_id]
        if batch_id not in prompt_batch_log_lookup:
            prompt_batch_log_path = (
                dsfiles.prompt_batches_directory / f"{batch_id}.json"
            )
            prompt_batch_log = json.loads(
                prompt_batch_log_path.read_text(encoding="utf-8")
            )
            prompt_batch_log_lookup[batch_id] = prompt_batch_log

        prompt_batch_log = prompt_batch_log_lookup[batch_id]
        msg_seeds = prompt_batch_log["message_id_to_seed_docids"][msg_id]

        # Previously there was a bug, such that every message got ALL seeds of the batch saved as list of lists in message_id_to_seed_docids
        # In newer versions, this is just a single list
        is_buggy_lookup = all(isinstance(elem, list) for elem in msg_seeds)
        if is_buggy_lookup:
            # we need to retrive the correct sublist via index
            i = prompt_batch_log["message_ids"].index(msg_id)
            msg_seeds = msg_seeds[i]
        else:
            # msg_seeds is already in correct format
            ...
        problematic_seeds.extend(msg_seeds)
    c = Counter(problematic_seeds)
    sc = sorted(c.items(), key=lambda item: item[1], reverse=True)
    for seed, cnt in sc[:3]:
        print(f"{cnt=} {dsfiles.preprocessed_seed_images_directory / f'{seed}.jpg'}")

    return all_refusals


if __name__ == "__main__":
    dsnames = [
        "cord_alpha=0.75",
        "cord_alpha=1.0_v1",
        "docvqa_alpha=0.5",
        "docvqa_alpha=0.5_v1",
        "docvqa_alpha=0.75",
        "docvqa_alpha=0.75_v1",
        "docvqa_alpha=1.0",
        "docvqa_alpha=1.0_v1",
        "publaynet_alpha=0.75",
        "rvlcdip_alpha=0.5",
        "rvlcdip_alpha=0.5_v1",
        "rvlcdip_alpha=0.75",
        "rvlcdip_alpha=0.75_v1",
        "rvlcdip_alpha=1.0",
        "rvlcdip_alpha=1.0_v1",
    ]
    for n in dsnames:
        refusals = search_for_refusals(n)
        print(f"{n} {len(refusals)=}")
