"""Exploratory data analysis for the PE dataset profiles.

This package only *reads* derived artifacts and writes an EDA run directory. It never
touches the read-only release, never modifies a manifest, and never produces anything that
training consumes -- so an EDA run can be repeated at any time without invalidating a
model run.

Layout deviates from the repository's usual source/ + tools/ split on purpose: the whole
analysis stage is self-contained here so it can be read, copied or dropped as one unit.
"""
