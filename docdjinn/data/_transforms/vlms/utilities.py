

import json
from docdjinn.data._core._utilities import TaskType


def _prepare_system_messages(task_type: TaskType, labels: list[str]) -> str:
    if task_type == TaskType.sequence_classification:
        return f"You are a document classification model. Classify the document into one of the given categories: {json.dumps(labels)}."

    elif task_type == TaskType.token_classification:
        return f"You are an information extraction model. Extract all the entities present in this document. Choose from the given entity categories: {json.dumps(labels)}."

    elif task_type == TaskType.extractive_qa:
        return "You are a question answering model. Answer the question based on the content of the document."

    elif task_type == TaskType.layout_analysis:
        return f"""
        You are a layout analysis model. Extract the layout entities present in the document. 
        Choose from the given layout categories: {json.dumps(labels)}. 
        Provide the output in the format <x0><y0><x1><y1><label>, 
        where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the bounding box respectively.

        The coordinates should be normalized relative to the full image such that:
        - Top-left (0,0)
        - Bottom-right (1000,1000)
        - All bounding boxes must be integers in the range [0, 1000]

        Example output:
        100,150,400,300,{labels[0]}
        50,350,950,450,{labels[1]}
        """
    else:
        raise NotImplementedError(f"Task type {task_type} not supported.")


def _format_sample_into_chat_template(system_message, image, text_input, target_text):
    return {
        "images": [image],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": text_input,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            },
        ],
    }
