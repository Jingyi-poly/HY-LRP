"""Run only the independent EF, using the familiar comparison parameter block.

The original ``python src-ini/run_ef_only.py`` entry is retained. Optional
arguments are shared with ``tools/run_ef_only.py``; no SDDP solver is run.
"""
from tools.run_ef_only import main


if __name__ == '__main__':
    raise SystemExit(main())
