"""ЛР3: трек B, синтез правил распределения задач, вариант 20."""

from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))
from dispatch_experiments import cli

if __name__ == "__main__":
    cli(3, DIRECTORY)
