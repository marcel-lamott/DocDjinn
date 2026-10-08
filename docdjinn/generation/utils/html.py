from bs4 import Tag


def get_field_text(field: Tag) -> str:
    """
    Extract text from a BeautifulSoup Tag.

    Works for:
    - <input> elements (uses 'value' attribute)
    - Other tags (uses inner text)
    """
    if field.name == "input":
        return field.get("value", "").strip()  # type: ignore
    else:
        return field.text.strip()
