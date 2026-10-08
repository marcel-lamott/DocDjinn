import json
import pathlib
from docdjinn import GENERATION
from docdjinn.generation.constants import WRITER_STYLES
from docdjinn.generation.handwriting_diffusion.add_handwriting_blur import (
    blur_handwriting,
)
from docdjinn.generation.handwriting_diffusion.generate_handwriting_diffusion_raw import (
    generate_handwriting,
)
from docdjinn.generation.models import PipelineParameters
from docdjinn.generation.models._log import DocLogKey, SynDocumentLog
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar


def pipeline_create_handwriting_images(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()
    has_handwriting = list(
        [f for f in dsfiles.handwritten_bboxes_directory.iterdir() if f.is_file()]
    )

    if params.generate_handwriting and has_handwriting:
        # sentences_path = dsfiles.handwritten_text_images_directory / "sentences"
        # handwriting_exists = sentences_path.exists()
        # if handwriting_exists:
        #     print(f"Existing handwriting found at {sentences_path} - skipping step.")
        #     return

        with get_progress_bar() as progress:
            generate_handwriting(
                input_dir=dsfiles.handwritten_bboxes_directory,
                output_dir=dsfiles.handwritten_text_images_directory,
                run_dir=GENERATION.HANDWRITING_MODEL_CHECKPOINT.parent,
                checkpoint=GENERATION.HANDWRITING_MODEL_CHECKPOINT.name,
                progress=progress,
                word_gap=40,
                segment_gap=0,
                allowed_writers=[str(s) for s in WRITER_STYLES],
                baseline_percentile=50,
                batch_size=params.handwriting_batch_size,
            )

            print(f"{params.blur_handwriting_images=}")
            if params.blur_handwriting_images:
                blur_handwriting(
                    input_root=dsfiles.handwritten_text_images_directory / "sentences",
                    in_place=True,
                    suffix="",
                )

            # Log selected writer styles
            log_path = dsfiles.handwritten_text_images_directory / "raw_token_map.json"
            genlogs = json.loads(log_path.read_text(encoding="utf-8"))
            for k, v in genlogs["file_author_styles"].items():
                doc_id = k.replace(".json", "")
                dsdef.write_to_document_log(
                    document_id=doc_id,
                    vals={DocLogKey.handwriting_generation_authorid_to_writerstyle: v},
                )

    else:
        print("No handwriting bboxes found - skipping step.")
