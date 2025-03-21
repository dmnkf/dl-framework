import subprocess
from pathlib import Path
import pytest
import os


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip in CI environment as it requires data not accessible in CI",
)
def test_quarto_render(tmp_path: Path):
    """Test that Quarto can render a simple document."""
    # Create a test Quarto document
    # Try to render the document
    result = subprocess.run(
        [
            "quarto",
            "render",
            str(Path(__file__).parent.parent / "notebooks" / "quarto_book"),
            "--to",
            "html",
        ],
        capture_output=True,
        text=True,
    )

    # Check if the command was successful
    assert result.returncode == 0, f"Quarto render failed with error: {result.stderr}"
