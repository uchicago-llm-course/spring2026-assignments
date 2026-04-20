"""Put the HW3 root on sys.path so tests can `import src.*`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
