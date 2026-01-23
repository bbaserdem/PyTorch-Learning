"""CLI interface for pytorch-learning."""

import typer
from rich.console import Console

app = typer.Typer(
    name="pytorch-learn",
    help="PyTorch learning CLI",
    add_completion=False,
)
console = Console()


@app.command()
def hello(name: str = typer.Argument("World", help="Name to greet")) -> None:
    """Say hello - a simple test command."""
    console.print(f"[bold green]Hello, {name}![/bold green]")
    console.print("Welcome to PyTorch learning!")


@app.command()
def version() -> None:
    """Show the version."""
    from pytorch_learning import __version__

    console.print(f"pytorch-learning version: [bold]{__version__}[/bold]")


@app.command()
def info() -> None:
    """Show information about the environment."""
    console.print("[bold]PyTorch Learning Environment[/bold]\n")

    try:
        import torch

        console.print(f"PyTorch version: {torch.__version__}")
        console.print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        console.print(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        console.print("[red]PyTorch not installed[/red]")

    try:
        import torchvision

        console.print(f"TorchVision version: {torchvision.__version__}")
    except ImportError:
        console.print("[red]TorchVision not installed[/red]")


if __name__ == "__main__":
    app()
