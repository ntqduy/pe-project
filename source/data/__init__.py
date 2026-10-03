"""Everything about the cohort, in one package.

``profiles/``   which patients a run uses: the three active dataset profiles (YAML, not code)
``build/``      stage 0, offline: raw INSPECT -> eligible cohort -> manifests, EHR, sPESI, caches
runtime         ``paths`` (project roots), ``dataset`` (PyTorch Dataset), ``manifests`` (read and
                audit manifests), ``experiment_splits`` (training fractions),
                ``preflight`` (input checks before a run)
"""
