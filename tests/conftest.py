"""Test configuration: ensure skill/memabra/ is importable as 'memabra'."""

import os
import sys

# Add skill/ directory to path so 'import memabra' resolves to skill/memabra/
skill_dir = os.path.join(os.path.dirname(__file__), '..', 'skill')
sys.path.insert(0, os.path.abspath(skill_dir))
