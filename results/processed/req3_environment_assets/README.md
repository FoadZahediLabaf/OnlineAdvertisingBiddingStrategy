# Requirement 3 saved Gaussian-Constellation assets

These files save the final environment assets used for the Requirement 3
diagnostic experiments.

Each setting folder contains:

- `config.json`: all seeds, sKL range, change frequency, anchor, and schedule.
- `constellation.npz`: Gaussian mode means (`mus`) and covariances (`sigmas`).
- `windows.csv`: fixed regime windows and active mode per window.
- `streams.csv`: repeated generated competing-bid time series.
- `streams.npz`: compact NumPy copy of all repeated streams.

The first constellation star is fixed to `mu = [0.5] * 5` and
`Sigma = 0.01 I`, so all modes keep low variance while respecting the target
pairwise sKL range.
