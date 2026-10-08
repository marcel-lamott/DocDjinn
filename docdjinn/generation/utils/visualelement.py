__PREFIX = "&?ve"


def get_visual_element_id(i: int) -> str:
    return f"{__PREFIX}{i}"


def is_visual_element_id(s: str) -> bool:
    if s.startswith(__PREFIX):
        s = s.replace(__PREFIX, "")
        return s.isdigit()
    else:
        return False
