from dataclasses import is_dataclass
import pathlib
import base64
from typing import get_type_hints


def from_dict(cls, data: dict):
    """
    Recursively parse a dictionary into a dataclass instance.
    Handles nested dataclasses and special types like pathlib.Path.
    """
    type_hints = get_type_hints(cls)
    kwargs = {}

    for field_name, field_type in type_hints.items():
        value = data.get(field_name)

        if value is None:
            kwargs[field_name] = None
        elif getattr(field_type, "__origin__", None) is list:
            subtype = field_type.__args__[0]
            if is_dataclass(subtype):
                kwargs[field_name] = [from_dict(subtype, v) for v in value]
            elif subtype == pathlib.Path:
                kwargs[field_name] = [pathlib.Path(v) for v in value]
            else:
                kwargs[field_name] = value
        elif is_dataclass(field_type):
            kwargs[field_name] = from_dict(field_type, value)
        elif field_type == pathlib.Path:
            kwargs[field_name] = pathlib.Path(value)
        else:
            kwargs[field_name] = value

    return cls(**kwargs)


def image_to_base64(imgpath: pathlib.Path) -> str:
    with open(imgpath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")
