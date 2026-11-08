"""Entry point for IDEs (VS Code, PyCharm, Spyder, IDLE): open this file and press Run.

With no arguments it trains a fast Decision Tree, tests it and classifies a few records. See `idsbench/runner.py`
for the other commands (train, predict).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # works without `pip install -e .`

from idsbench.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
