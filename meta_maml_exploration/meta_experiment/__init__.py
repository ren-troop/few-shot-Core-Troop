"""MAML exploration experiments on Omniglot."""

# The shared ``src.common`` helpers live at the repository root while the
# training entry points run from this subdirectory.
from pathlib import Path
import sys

_repository_root = str(Path(__file__).resolve().parents[2])
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)

__all__ = ["data", "model", "learner", "utils"]

