import logging

from bs4 import BeautifulSoup
import cssutils
from docdjinn import ENV
from docdjinn.generation.constants import (
    BS_PARSER,
    HANDWRITING_CLASS_NAME,
    HANDWRITING_FONT_SIZE,
)
from docdjinn.generation.utils.handwriting import get_all_author_ids
from docdjinn.generation.utils.visualelement import get_visual_element_id

# Your input HTML (replace with reading from file if needed)
html = """
<html>
<head>
  <style>
    @page {
      size: A4;
      margin: 1in;
    }
    body {
      display: block;
      font-size: 12pt;
    }
  </style>
</head>
<body>
  <p>Hello World</p>
</body>
</html>
"""

# html_path = ENV.DATA_DIR / "html" / "receipt3.html"
# html = html_path.read_text(encoding="utf-8")


# Get cssutils logger
cssutils_logger = logging.getLogger("CSSUTILS")

# Remove all handlers (if any) and prevent propagation
cssutils_logger.handlers.clear()
cssutils_logger.propagate = False

# Add a NullHandler so it discards all logs
cssutils_logger.addHandler(logging.NullHandler())


def unmark_visual_elements_alt(soup: BeautifulSoup):
    fields = soup.find_all(attrs={"data-placeholder": True})

    style_tag = soup.find("style")
    if not style_tag:
        raise ValueError("No <style> tag found in HTML.")
    # Parse the CSS
    css = cssutils.parseString(style_tag.string)  # type: ignore

    for i, div in enumerate(fields):
        div.clear()  # type: ignore
        style = cssutils.parseStyle(div.get("style"))  # type: ignore
        del style["font-size"]
        del style["text-align"]
        del style["display"]
        del style["justify-content"]
        del style["align-items"]
        # Delete stuff that the LLM may have added
        del style["border"]
        del style["background-color"]
        del style["background-image"]

        # Maybe the element has a class that rotates it, this happened for stamps
        # Copy rotate from potential class
        transforms = []
        viselemclasses = div.get("class")
        if viselemclasses:  # type: ignore
            for cls in viselemclasses:  # type: ignore
                for rule in css:
                    if rule.type == rule.STYLE_RULE and rule.selectorText == f".{cls}":
                        if "transform" in rule.style:
                            transforms.append(rule.style["transform"])

        if len(transforms) > 1:
            raise ValueError(
                f"Visual element has multiple transforms, assigned by {len(transforms)} classes"
            )

        if len(transforms) == 1:
            # Copy transform
            style["transform"] = transforms[0]
            # print(f"Copied transform! {transforms[0]}")

        div["style"] = style.cssText  # type: ignore
        div["class"] = []  # type: ignore

    return soup


def unmark_visual_elements(soup: BeautifulSoup):
    # Find all elements that have the data-placeholder attribute
    fields = soup.find_all(attrs={"data-placeholder": True})

    # Get the style tag
    style_tag = soup.find("style")
    if not style_tag:
        raise ValueError("No <style> tag found in HTML.")

    # Parse the CSS
    css = cssutils.parseString(style_tag.string)  # type: ignore

    # Remove any CSS rules targeting data-placeholder elements
    to_remove = []
    for rule in css:
        if rule.type == rule.STYLE_RULE:
            selector = rule.selectorText
            # Match any selector like div[data-placeholder="..."]
            if "[data-placeholder" in selector:
                to_remove.append(rule)

    for rule in to_remove:
        css.deleteRule(rule)

    # Apply inline style cleanup for each placeholder field
    for div in fields:
        div.clear()  # type: ignore
        style = cssutils.parseStyle(div.get("style", ""))  # type: ignore

        for prop in [
            "font-size",
            "text-align",
            "display",
            "justify-content",
            "align-items",
            "border",
            "background-color",
            "background-image",
            "background",
            "color",
        ]:
            if prop in style:
                del style[prop]

        # Copy transform from any class rules, if needed
        transforms = []
        viselemclasses = div.get("class")  # type: ignore
        if viselemclasses:
            for cls in viselemclasses:
                for rule in css:
                    if rule.type == rule.STYLE_RULE and rule.selectorText == f".{cls}":
                        if "transform" in rule.style:
                            transforms.append(rule.style["transform"])

        if len(transforms) > 1:
            raise ValueError(
                f"Visual element has multiple transforms, assigned by {len(transforms)} classes"
            )

        if len(transforms) == 1:
            style["transform"] = transforms[0]

        div["style"] = style.cssText  # type: ignore
        div["class"] = []  # type: ignore

    # Write the cleaned CSS back into the style tag
    style_tag.string = css.cssText.decode("utf-8")  # type: ignore

    return soup


def increase_handwriting_font_size(soup: BeautifulSoup, dbg=False) -> BeautifulSoup:
    # Find <style> tag
    style_tag = soup.find("style")
    if not style_tag:
        raise ValueError("No <style> tag found in HTML.")

    # Parse the CSS
    css = cssutils.parseString(style_tag.string)

    # Increase handwriting font size
    hwselector = f".{HANDWRITING_CLASS_NAME}"
    hw_rule = None
    for rule in css:
        if rule.type == rule.STYLE_RULE and hwselector == rule.selectorText:
            hw_rule = rule
            break

    found = hw_rule is not None
    if dbg:
        ...
        # print(hw_rule)
        # input()
    if not found:
        hw_rule = cssutils.css.CSSStyleRule(selectorText=hwselector)
    if hw_rule:
        # hw_rule.style["font-size"] = f"{HANDWRITING_FONT_SIZE}px"
        hw_rule.style.setProperty(
            "font-size", f"{HANDWRITING_FONT_SIZE}px", priority="important"
        )  # = f"{HANDWRITING_FONT_SIZE}px"
    if not found:
        css.add(hw_rule)

    # Replace style content
    style_tag.string = css.cssText.decode("utf-8")

    # Output modified HTML
    return soup


def postprocess_handwriting(soup: BeautifulSoup) -> BeautifulSoup:
    # Find <style> tag
    style_tag = soup.find("style")
    if not style_tag:
        raise ValueError("No <style> tag found in HTML.")

    # Parse the CSS
    css = cssutils.parseString(style_tag.string)

    hwselector = f".{HANDWRITING_CLASS_NAME}"
    hw_rule = None
    for rule in css:
        if rule.type == rule.STYLE_RULE and hwselector in rule.selectorText:
            hw_rule = rule
            break

    found = hw_rule is not None
    if not found:
        hw_rule = cssutils.css.CSSStyleRule(selectorText=hwselector)
    if hw_rule:
        hw_rule.style["color"] = "transparent"  # This removes the text from PDF
        # hw_rule.style["-webkit-print-color-adjust"] = "exact"
        # hw_rule.style["opacity"] = 0 # doesnt work
        # hw_rule.style["color"] = "rgba(0, 0, 0, 0.01)" # wors but looks bad
        hw_rule.style["text-shadow"] = "none"
        hw_rule.style["text-decoration"] = "none"
    if not found:
        css.add(hw_rule)

    all_author_ids = get_all_author_ids(soup)
    for authorid in all_author_ids:
        authorselector = f".{authorid}"
        author_rule = None
        for rule in css:
            if rule.type == rule.STYLE_RULE and authorselector in rule.selectorText:
                author_rule = rule
                break

        found = author_rule is not None
        if not found:
            author_rule = cssutils.css.CSSStyleRule(selectorText=authorselector)
        if author_rule:
            author_rule.style["color"] = "transparent"  # This removes the text from PDF
            author_rule.style["text-shadow"] = "none"
            author_rule.style["text-decoration"] = "none"
        if not found:
            css.add(author_rule)

    # Replace style content
    style_tag.string = css.cssText.decode("utf-8")

    # Output modified HTML
    return soup
