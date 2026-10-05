"""Location of the hand-written subproblem solvers (``src-ini/customized-subprob``).

Layout::

    customized-subprob/
      s2forward/    fixed-fleet Stage-2 assignment (fleet layout + exact subset DP)
      s2backward/   Stage-2 Lagrangian oracle (fleet pieces, piece table) + bpc/
      s3backward/   Stage-3 ESPPRC / PCTSP C++ (espprc_cpp)

The directory name contains a hyphen, so the Python packages ``s2forward`` and
``s2backward`` are made importable by putting the root on ``sys.path``.
Importing this module does that once; every consumer imports it before
``import s2forward`` / ``import s2backward``.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "customized-subprob"
)
S2_FORWARD_DIR = os.path.join(ROOT, "s2forward")
S2_BACKWARD_DIR = os.path.join(ROOT, "s2backward")
S2_BPC_DIR = os.path.join(S2_BACKWARD_DIR, "bpc")
S3_BACKWARD_DIR = os.path.join(ROOT, "s3backward")


def ensure_import_path() -> str:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return ROOT


ensure_import_path()
