from bs4 import Tag
from docdjinn.generation.constants import HANDWRITING_CLASS_NAME


def get_author_id_from_field(field: Tag) -> str | None:
    all_classes = field.get("class", [])  # type: ignore
    return get_author_id(all_classes)


def get_author_id(all_classes: list[str]) -> str | None:
    other_classes = [c for c in all_classes if c != HANDWRITING_CLASS_NAME]  # type: ignore
    valid_author_ids = [c for c in other_classes if c.startswith("author")]
    author_id = valid_author_ids[0] if valid_author_ids else None
    return author_id


def get_all_author_ids(soup) -> set[str]:
    fields = soup.find_all(class_=HANDWRITING_CLASS_NAME)

    # Extract text content
    result = set()
    for i, field in enumerate(fields):
        author_id = get_author_id_from_field(field)
        result.add(author_id)

    return result
