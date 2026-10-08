from collections import defaultdict
import subprocess
import json

from docdjinn.generation.models import OCRBox


def convert_to_pdf_coordinates(positions: list[dict]) -> list[dict]:
    """
    Convert pixel coordinates to PDF points.
    """

    def pixels_to_points(pixels, dpi=96):
        return (pixels / dpi) * 72

    pdf_positions = []
    for pos in positions:
        pdf_pos = {
            "index": pos["index"],
            "text": pos["text"],
            "x_points": pixels_to_points(pos["x"]),
            "y_points": pixels_to_points(pos["y"]),
            "width_points": pixels_to_points(pos["width"]),
            "height_points": pixels_to_points(pos["height"]),
            "x_pixels": pos["x"],
            "y_pixels": pos["y"],
            "width_pixels": pos["width"],
            "height_pixels": pos["height"],
        }
        pdf_positions.append(pdf_pos)

    return pdf_positions


def prepare_html_for_position_extraction(
    html: str, classes: list[str] | None = None
) -> str:
    if classes is None:
        classes = ["handwritten"]

    # Generate JavaScript selectors for all classes
    classes_json = json.dumps(classes)

    # First, inject JavaScript to capture positions
    position_capture_script = f"""
    <script>
    window.addEventListener('load', function() {{
        const classNames = {classes_json};
        const positionsByClass = {{}};

        classNames.forEach(className => {{
            const elements = document.querySelectorAll('.' + className);
            const positions = [];

            elements.forEach((element, index) => {{
                const rect = element.getBoundingClientRect();
                positions.push({{
                    index: index,
                    className: className,
                    text: element.textContent.trim(),
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height
                }});
            }});

            positionsByClass[className] = positions;
        }});

        // Write to a hidden element that we can extract
        const dataDiv = document.createElement('div');
        dataDiv.id = '__positions_data__';
        dataDiv.style.display = 'none';
        dataDiv.textContent = JSON.stringify(positionsByClass);
        document.body.appendChild(dataDiv);
    }});
    </script>
    """

    # Inject the script before </body> or at the end
    if "</body>" in html:
        html_with_script = html.replace("</body>", f"{position_capture_script}</body>")
    else:
        html_with_script = html + position_capture_script

    return html_with_script


def extract_positions_from_subprocess_result(
    result: subprocess.CompletedProcess[str],
) -> dict[str, list[dict]]:
    # Extract positions from DOM dump
    positions_by_class = {}
    if result.returncode == 0:
        # Look for the hidden div with positions data
        import re

        match = re.search(
            r'<div id="__positions_data__"[^>]*>([^<]+)</div>', result.stdout
        )
        if match:
            try:
                positions_by_class = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Convert all positions to PDF coordinates
    result_dict = {}
    for class_name, positions in positions_by_class.items():
        result_dict[class_name] = convert_to_pdf_coordinates(positions)

    return result_dict


def convert_to_bboxes(
    positions_by_class: dict[str, list[dict]],
) -> dict[str, list[OCRBox]]:
    result = defaultdict(list)
    for classname, entries in positions_by_class.items():
        for e in entries:
            x0 = e["x_points"]
            y0 = e["y_points"]
            width = e["width_points"]
            height = e["height_points"]
            txt = e["text"]
            bbox = OCRBox(
                x0=x0,
                y0=y0,
                x2=x0 + width,
                y2=y0 + height,
                text=txt,
                block_no=-1,
                line_no=-1,
                word_no=-1,
            )
            result[classname].append(bbox)

    return result
