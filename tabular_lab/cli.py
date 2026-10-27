from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import load_task

app = typer.Typer(no_args_is_help=True, help="Reproducible, budget-aware tabular classification experiments.")
auth = typer.Typer(no_args_is_help=True, help="Manage local DeepSeek credentials without making API calls.")
app.add_typer(auth, name="auth")


@auth.command("login")
def auth_login():
    """Save a key in Windows Credential Manager for future processes."""
    from .credentials import CredentialError, WindowsCredentialStore
    try:
        store = WindowsCredentialStore()
        key = typer.prompt("DeepSeek API key (hidden input)", hide_input=True)
        store.write(key)
        typer.echo("Saved in Windows Credential Manager for this user. No API request was made.")
    except CredentialError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@auth.command("status")
def auth_status():
    """Check local key availability; never print the key or call the API."""
    from .credentials import CredentialError, get_api_key
    try:
        if not get_api_key():
            typer.echo("No DeepSeek key configured.")
            raise typer.Exit(1)
        typer.echo("DeepSeek key is configured locally (value hidden; not API-verified).")
    except CredentialError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@auth.command("logout")
def auth_logout():
    """Remove the saved Windows credential; leave environment overrides unchanged."""
    import os

    from .credentials import CredentialError, delete_api_key
    try:
        removed = delete_api_key()
        typer.echo("Saved credential removed." if removed else "No saved credential to remove.")
        if os.environ.get("DEEPSEEK_API_KEY", "").strip():
            typer.echo("DEEPSEEK_API_KEY is still set in this process and remains an active override.")
    except CredentialError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def run(task: Annotated[Path, typer.Option(exists=True)],
        strategy: Annotated[str | None, typer.Option()] = None,
        provider: Annotated[str | None, typer.Option()] = None,
        device: Annotated[str | None, typer.Option()] = None):
    """Run a complete autonomous experiment and export its best model."""
    from .runner import run_task
    try:
        from .config import TaskConfig
        config = load_task(task)
        values = config.model_dump()
        if strategy:
            values["strategy"] = strategy
        if provider:
            values["llm"]["provider"] = provider
        if device:
            values["device"] = device
        config = TaskConfig.model_validate(values)
        result = run_task(config)
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "completed":
            raise typer.Exit(2)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def predict(model: Annotated[Path, typer.Option(exists=True)],
            input: Annotated[Path, typer.Option(exists=True)],
            output: Annotated[Path, typer.Option()]):
    """Predict with an exported model; no LLM or API key is needed."""
    from .runner import predict as predict_rows
    try:
        rows = predict_rows(model.resolve(), input, output)
        typer.echo(f"Wrote {rows} predictions to {output}")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def prepare(dataset: Annotated[str, typer.Option()] = "demo",
            output: Annotated[Path, typer.Option()] = Path("data")):
    """Prepare demo, breast-cancer, wine, wine-quality, adult, or all datasets."""
    from .datasets import prepare as prepare_data
    names = ["demo", "breast-cancer", "wine", "wine-quality", "adult"] if dataset == "all" else [dataset]
    for name in names:
        typer.echo(json.dumps(prepare_data(name, output), ensure_ascii=False, indent=2))


@app.command()
def benchmark(suite: Annotated[Path, typer.Option(exists=True)]):
    """Compare fixed baselines, random search and agent with matching limits."""
    from .benchmark import run_benchmark
    typer.echo(str(run_benchmark(suite)))


@app.command()
def publish_example(run: Annotated[Path, typer.Option(exists=True)],
                    output: Annotated[Path, typer.Option()]):
    """Curate a local run into a small, explicitly publishable example."""
    from .benchmark import curate_run
    curate_run(run, output)
    typer.echo(str(output))


@app.command()
def docker_check(image: Annotated[str, typer.Option()] = "tabular-lab-worker:0.1.0",
                 device: Annotated[str, typer.Option()] = "cpu"):
    """Validate V2 prerequisites without disabling V1."""
    from .sandbox import preflight
    try:
        typer.echo(json.dumps(preflight(image, device), indent=2))
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
