# Experiment configurations

Use TOML (Python 3.11 `tomllib`) for versioned experiment settings. `baseline.toml`
is a planning template, not a runnable environment or PPO configuration yet.
The next milestone should add validated environment and learner parameters here.
Record the resolved config, seed, Git commit, dependency snapshot and selected
hardware with each run. Keep machine paths and credentials out of shared configs.
