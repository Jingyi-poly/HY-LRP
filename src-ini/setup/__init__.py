"""Shared entry-point setup helpers.

Keep this module import-light: runtime defaults must be applied before NumPy,
Gurobi, models, or solver modules are imported.
"""
