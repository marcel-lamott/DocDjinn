import json
import pathlib

from rich.console import Console
from rich.table import Table

ANTHROPIC_PRICING = {
    "claude-sonnet-4-20250514": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
}


def calculate_message_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """
    Calculate the cost of a single message based on token usage.

    Args:
        model: The model name (e.g., "claude-sonnet-4-5-20250929")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cache_creation_input_tokens: Number of tokens used for cache creation
        cache_read_input_tokens: Number of tokens read from cache

    Returns:
        Cost in USD
    """
    if model not in ANTHROPIC_PRICING:
        print(f"Warning: Unknown model '{model}'. Using Claude Sonnet 4.5 pricing.")
        model = "claude-sonnet-4-5-20250929"

    pricing = ANTHROPIC_PRICING[model]

    regular_input_tokens = (
        input_tokens - cache_creation_input_tokens - cache_read_input_tokens
    )

    cost_usd = (
        (regular_input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
        + (cache_creation_input_tokens / 1_000_000) * pricing["cache_write"]
        + (cache_read_input_tokens / 1_000_000) * pricing["cache_read"]
    )

    return cost_usd


def get_total_cost(batch_data_directory: pathlib.Path) -> dict:
    """
    Calculate the total cost across all batches in a directory.

    Args:
        batch_data_directory: Directory containing batch metadata files

    Returns:
        Dictionary with aggregated cost information
    """
    total_cost_summary = {
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_creation_tokens": 0,
        "total_cache_read_tokens": 0,
        "num_batches": 0,
        "num_messages": 0,
    }

    for batch_file in batch_data_directory.iterdir():
        if batch_file.is_file() and batch_file.suffix == ".json":
            batch_metadata = json.loads(batch_file.read_text())

            if batch_metadata.get("processing_status") == "ended":
                cost_tracking = batch_metadata.get("cost_tracking", {})

                total_cost_summary["total_cost_usd"] += cost_tracking.get(
                    "total_cost_usd", 0.0
                )
                total_cost_summary["total_input_tokens"] += cost_tracking.get(
                    "total_input_tokens", 0
                )
                total_cost_summary["total_output_tokens"] += cost_tracking.get(
                    "total_output_tokens", 0
                )
                total_cost_summary["total_cache_creation_tokens"] += cost_tracking.get(
                    "total_cache_creation_tokens", 0
                )
                total_cost_summary["total_cache_read_tokens"] += cost_tracking.get(
                    "total_cache_read_tokens", 0
                )
                total_cost_summary["num_batches"] += 1
                total_cost_summary["num_messages"] += len(
                    batch_metadata.get("message_ids", [])
                )

    return total_cost_summary


def print_cost_report(
    batch_data_directory: pathlib.Path, dataset_log_path: pathlib.Path | None = None
):
    """
    Print a formatted cost report using Rich tables.

    Args:
        batch_data_directory: Directory containing batch metadata files
        dataset_log_path: Optional path to dataset log for per-document cost calculation
    """
    single_page_pdfs_count = -1
    if dataset_log_path and dataset_log_path.exists():
        dataset_log = json.loads(dataset_log_path.read_text(encoding="utf-8"))
        single_page_pdfs_count = dataset_log.get("valid_samples", {}).get("total", 0)

    total_cost_summary = get_total_cost(batch_data_directory)

    console = Console()

    table = Table(
        title="Batch Cost Report", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan", width=35)
    table.add_column("Value", justify="right", style="white", width=20)

    table.add_row("Number of batches", str(total_cost_summary["num_batches"]))
    table.add_row("Number of messages", str(total_cost_summary["num_messages"]))
    table.add_row("Number of PDFs", str(single_page_pdfs_count))

    table.add_section()
    table.add_row("Total input tokens", f"{total_cost_summary['total_input_tokens']:,}")
    table.add_row(
        "Total output tokens", f"{total_cost_summary['total_output_tokens']:,}"
    )
    table.add_row(
        "Total cache creation tokens",
        f"{total_cost_summary['total_cache_creation_tokens']:,}",
    )
    table.add_row(
        "Total cache read tokens", f"{total_cost_summary['total_cache_read_tokens']:,}"
    )

    table.add_section()
    total_cost_usd = total_cost_summary["total_cost_usd"] / 2.0
    table.add_row(
        "[bold green]TOTAL COST \n(including 50% batch discount)[/bold green]",
        f"[bold green]${total_cost_usd:.2f} USD[/bold green]",
    )

    if single_page_pdfs_count > 0:
        avg_cost_per_document = total_cost_usd / single_page_pdfs_count
        table.add_row(
            "[bold yellow]Average cost per document[/bold yellow]",
            f"[bold yellow]${avg_cost_per_document:.2f} USD[/bold yellow]",
        )
        table.add_row(
            "  Documents counted", f"{single_page_pdfs_count} single-page PDFs"
        )

    console.print()
    console.print(table)
    console.print()

    return total_cost_summary
