"""Prediction CLI entry-point.

A doctor can supply patient data either interactively or via a JSON file.

Usage::

    # Interactive prompt
    nhs-predict

    # JSON file input
    nhs-predict --input patient.json

    # Specify procedure and output JSON result
    nhs-predict --input patient.json --procedure KNEE --output result.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nhs_proms_pipeline.config import PipelineSettings, ProcedureType
from nhs_proms_pipeline.pipeline import InferencePipeline
from nhs_proms_pipeline.schemas.patient import ComorbidityProfile, PatientRecord, PreOpEQ5D
from nhs_proms_pipeline.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="nhs-predict",
    help="Predict post-operative health gain for a patient.",
    add_completion=False,
)
console = Console()


@app.command()
def predict(
    procedure: ProcedureType = typer.Option(
        ProcedureType.KNEE,
        "--procedure",
        "-p",
        help="Surgical procedure type (KNEE or HIP).",
        case_sensitive=False,
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to a JSON file containing the patient record.",
        exists=False,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write the prediction result as JSON.",
    ),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        help="Logging verbosity.",
    ),
    show_sample: bool = typer.Option(
        False,
        "--show-sample",
        help="Print a sample input JSON and exit.",
    ),
) -> None:
    """Generate a health-gain prediction for a single patient."""
    configure_logging(log_level)
    logger = get_logger(__name__)

    if show_sample:
        _print_sample_json(procedure)
        raise typer.Exit()

    try:
        if input_file is not None:
            patient = _load_patient_from_file(input_file)
        else:
            patient = _prompt_patient(procedure)

        settings = PipelineSettings(procedure_type=procedure)
        pipeline = InferencePipeline(settings)

        console.print("\n[bold cyan]Running prediction…[/bold cyan]")
        result = pipeline.predict(patient)

        # ── Display result ─────────────────────────────────────────────────
        benefit_str = "[bold green]YES — Expected to benefit[/bold green]" if result.predicted_benefit \
            else "[bold red]NO — Not expected to benefit[/bold red]"

        table = Table(title="Prediction Result", show_header=False)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Procedure", procedure.value)
        table.add_row("Predicted Health Gain", f"{result.predicted_health_gain:+.1f} pts")
        table.add_row("Expected Benefit", benefit_str)
        table.add_row("MCID Threshold", f"{result.mcid_threshold:.0f} pts")
        table.add_row("Model", result.model_name)
        console.print(table)

        console.print(
            Panel(
                result.confidence_note,
                title="Clinical Note",
                border_style="yellow",
            )
        )

        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(result.model_dump_json(indent=2))
            console.print(f"\n[green]Result saved to {output_file}[/green]")

    except FileNotFoundError as exc:
        console.print(
            f"[bold red]Error:[/bold red] {exc}\n"
            "Have you run [bold]nhs-train[/bold] first?"
        )
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[bold red]Validation error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction error.")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


# ── Helper functions ──────────────────────────────────────────────────────────


def _load_patient_from_file(path: Path) -> PatientRecord:
    """Parse and validate a patient JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Validated :class:`PatientRecord`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the JSON is invalid or fails Pydantic validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    raw = json.loads(path.read_text())
    return PatientRecord.model_validate(raw)


def _prompt_patient(procedure: ProcedureType) -> PatientRecord:
    """Interactively prompt the user for patient data.

    Returns:
        Validated :class:`PatientRecord`.
    """
    console.print(
        f"\n[bold]Enter patient data for [cyan]{procedure.value}[/cyan] replacement:[/bold]"
    )
    console.print("(Press Ctrl+C at any time to cancel)\n")

    age_band = typer.prompt(
        "Age Band",
        default="65 to 69",
        prompt_suffix="\n  Options: Under 45 | 45 to 49 | 50 to 54 | 55 to 59 | "
                      "60 to 64 | 65 to 69 | 70 to 74 | 75 to 79 | 80 to 84 | "
                      "85 to 89 | 90 and over\n  > ",
    )
    gender = typer.prompt("Gender (1=Male, 2=Female)", default=2, type=int)
    symptom_period = typer.prompt(
        "Symptom period (1=<1yr, 2=1-5yrs, 3=6-10yrs, 4=>10yrs)", default=2, type=int
    )

    console.print("\n[bold]Oxford Knee/Hip Score — 12 pre-op dimension questions (1=best, 5=worst):[/bold]")
    dims = [typer.prompt(f"  Q{i}", default=3, type=int) for i in range(1, 13)]

    console.print("\n[bold]EQ-5D pre-op dimensions (1=No problems, 2=Moderate, 3=Extreme):[/bold]")
    mobility = typer.prompt("  Mobility", default=2, type=int)
    self_care = typer.prompt("  Self-care", default=1, type=int)
    activity = typer.prompt("  Usual activities", default=2, type=int)
    discomfort = typer.prompt("  Pain/discomfort", default=3, type=int)
    anxiety = typer.prompt("  Anxiety/depression", default=1, type=int)

    return PatientRecord(
        age_band=age_band,
        gender=float(gender),
        pre_op_q_1=dims[0], pre_op_q_2=dims[1], pre_op_q_3=dims[2],
        pre_op_q_4=dims[3], pre_op_q_5=dims[4], pre_op_q_6=dims[5],
        pre_op_q_7=dims[6], pre_op_q_8=dims[7], pre_op_q_9=dims[8],
        pre_op_q_10=dims[9], pre_op_q_11=dims[10], pre_op_q_12=dims[11],
        eq5d=PreOpEQ5D(
            mobility=mobility,
            self_care=self_care,
            activity=activity,
            discomfort=discomfort,
            anxiety=anxiety,
        ),
        symptom_period=symptom_period,
        comorbidities=ComorbidityProfile(),
    )


def _print_sample_json(procedure: ProcedureType) -> None:
    """Print a sample patient JSON to stdout."""
    sample = PatientRecord(
        age_band="65 to 69",
        gender=2.0,
        pre_op_q_1=3, pre_op_q_2=3, pre_op_q_3=3, pre_op_q_4=3,
        pre_op_q_5=3, pre_op_q_6=3, pre_op_q_7=3, pre_op_q_8=3,
        pre_op_q_9=3, pre_op_q_10=3, pre_op_q_11=3, pre_op_q_12=3,
        eq5d=PreOpEQ5D(mobility=2, self_care=1, activity=2, discomfort=3, anxiety=1),
        symptom_period=2,
        comorbidities=ComorbidityProfile(diabetes=1, high_bp=1),
    )
    console.print(f"\n[bold]Sample patient JSON ({procedure.value}):[/bold]\n")
    console.print(sample.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
