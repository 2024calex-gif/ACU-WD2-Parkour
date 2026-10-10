"""Make the pipeline folder importable (mab, skel, retarget, dumps, wd2, clipnames, config) from the tests."""
import os
import sys

PIPELINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)
