import io
from flask import Response
from PIL import Image, ImageDraw, ImageFont

from .data_manager import data_manager
from .config import settings
from docdjinn.logging import get_logger

logger = get_logger(__name__)

def setup_server_routes(server):
    """Setup Flask server routes."""

    @server.route("/image/<string:sample_id>")
    def serve_image(sample_id: str):
        """Serve individual document image."""
        try:
            sample = data_manager.dataset.train.get_by_id(sample_id)
            assert sample.sample_id == sample_id, (
                f"Mismatched sample ID, found {sample.sample_id}, expected {sample_id}"
            )
            image = sample.image.content
            img_io = io.BytesIO()
            image.save(img_io, "PNG")
            img_io.seek(0)
            return Response(img_io.getvalue(), mimetype="image/png")
        except (ValueError, IndexError) as e:
            return f"Invalid sample ID: {str(e)}", 404
        except Exception as e:
            return f"Error serving image: {str(e)}", 500

    @server.route("/cluster_grid/<string:sample_ids>")
    def serve_cluster_grid(sample_ids):
        """Create and serve a grid image from multiple document images."""
        try:
            sample_ids = _parse_ids(sample_ids)
            grid_img = _create_grid_image(sample_ids)
            return _image_response(grid_img)
        except Exception as e:
            return f"Error creating grid: {str(e)}", 500


def _parse_ids(sample_ids):
    """Parse and validate sample_ids list."""
    return sample_ids.split(",")[: settings.max_images]


def _create_grid_image(sample_ids):
    """Create a grid image from document sample_ids."""
    cols = min(settings.max_cols, len(sample_ids))
    rows = (len(sample_ids) + cols - 1) // cols

    grid_width = cols * settings.thumb_width + (cols - 1) * settings.spacing
    grid_height = rows * settings.thumb_height + (rows - 1) * settings.spacing

    grid_img = Image.new("RGB", (grid_width, grid_height), "white")

    for i, sample_id in enumerate(sample_ids):
        _add_thumbnail_to_grid(grid_img, sample_id, i, cols)

    return grid_img


def _add_thumbnail_to_grid(grid_img, sample_id, position, cols):
    """Add a single thumbnail to the grid."""
    row = position // cols
    col = position % cols
    x = col * (settings.thumb_width + settings.spacing)
    y = row * (settings.thumb_height + settings.spacing)

    try:
        sample = data_manager.dataset.train.get_by_id(sample_id)
        assert sample.sample_id == sample_id, (
            f"Mismatched sample ID, found {sample.sample_id}, expected {sample_id}"
        )
        image = sample.image.content
        image.thumbnail(
            (settings.thumb_width, settings.thumb_height - 30), Image.Resampling.LANCZOS
        )
        grid_img.paste(image, (x, y))
        _add_label(grid_img, sample_id, x, y + image.height + 5)
    except Exception as e:
        logger.exception(f"Error loading image for sample ID {sample_id}")
        _draw_error_placeholder(grid_img, sample_id, x, y)


def _add_label(grid_img, text, x, y):
    """Add text label to the grid."""
    draw = ImageDraw.Draw(grid_img)
    font = _get_font()
    draw.text((x, y), text, fill="black", font=font)


def _get_font():
    """Get font for text rendering."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        return ImageFont.load_default()


def _draw_error_placeholder(grid_img, index, x, y):
    """Draw error placeholder when image fails to load."""
    draw = ImageDraw.Draw(grid_img)
    draw.rectangle(
        [x, y, x + settings.thumb_width, y + settings.thumb_height - 30],
        outline="gray",
        fill="lightgray",
    )
    draw.text((x + 10, y + 10), f"Error loading\n{index}", fill="black")


def _image_response(image):
    """Convert PIL Image to Flask Response."""
    img_io = io.BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)
    return Response(img_io.getvalue(), mimetype="image/png")
