"""Run Bree's unmodified generator in its own directory, regardless of caller CWD."""
from pathlib import Path
import subprocess
import sys


def main() -> None:
    directory = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(directory / "generate_data.py")],
                   cwd=directory, check=True)


if __name__ == "__main__":
    main()
