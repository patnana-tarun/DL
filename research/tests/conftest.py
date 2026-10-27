"""Make research/ importable (importing it runs research/paths.py, which chdirs to the project root)."""
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
