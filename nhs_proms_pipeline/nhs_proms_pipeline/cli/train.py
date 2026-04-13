"""Training CLI entry-point.

Usage::

    # Train for Knee (default)
    nhs-train

    # Train for Hip
    nhs-train --procedure HIP

    # Skip already-done steps
    nhs-train --skip-collection --skip-preprocessing

    # List enabled models
    nhs-train --list-models
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from nhs_proms_pipeline.config import PipelineSettings, ProcedureType
from nhs_proms_pipeline.modelling.registry import available_models
from nhs_proms_pipeline.pipeline import TrainingPipeline
from nhs_proms_pipeline.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="nhs-train",
    help="Train the NHS PROMs health-gain prediction pipeline.",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    procedure: ProcedureType = typer.Option(
        ProcedureType.KNEE,
        "--procedure",
        "-p",
        help="Surgical procedure type (KNEE or HIP).",
        case_sensitive=False,
    ),
    skip_collection: bool = typer.Option(
        False,
        "--skip-collection",
        help="Skip Step 1 — data collection (assumes 1.1-Reduced.parquet exists).",
    ),
    skip_preprocessing: bool = typer.Option(
        False,
        "--skip-preprocessing",
        help="Skip Step 2.0 — data pre-processing (assumes 2.0-preprocessing.parquet exists).",
    ),
    skip_preparation: bool = typer.Option(
        False,
        "--skip-preparation",
        help="Skip Step 2.1 — data preparation (assumes 2.1-train/test.parquet exist).",
    ),
    list_models: bool = typer.Option(
        False,
        "--list-models",
        help="Print available model names and exit.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    """Run the full NHS PROMs training pipeline."""
    configure_logging(log_level)
    logger = get_logger(__name__)

    if list_models:
        console.print("\n[bold]Available models:[/bold]")
        for name in available_models():
            console.print(f"  • {name}")
        raise typer.Exit()

    try:
        settings = PipelineSettings(procedure_type=procedure)
        console.print(
            f"\n[bold cyan]NHS PROMs Training Pipeline[/bold cyan] — "
            f"Procedure: [bold]{procedure.value}[/bold]\n"
        )

        pipeline = TrainingPipeline(settings)
        best = pipeline.run(
            skip_collection=skip_collection,
            skip_preprocessing=skip_preprocessing,
            skip_preparation=skip_preparation,
        )

        # ── Success summary ─────────────────────────────────────────────────
        table = Table(title="Best Model Summary", show_header=True)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Procedure", procedure.value)
        table.add_row("Dataset", best.dataset_label)
        table.add_row("Model", best.model_name)
        table.add_row("Test RMSE", f"{best.test_rmse:.4f}")
        table.add_row("Test R²", f"{best.test_r2:.4f}")
        table.add_row("MCID F₂", f"{best.mcid_f2:.4f}" if best.mcid_f2 else "N/A")
        table.add_row("Pipeline saved to", str(best.pipeline_path))
        console.print(table)

    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Unexpected error during training.")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
