"""
conftest.py

Pytest automatically loads this file before running any tests. Its only
job here is to make sure Python can find and import our project's modules
(like producer/generate_transactions.py) regardless of which folder
pytest is run from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))