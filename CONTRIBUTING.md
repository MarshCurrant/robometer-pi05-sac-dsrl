# Contributing

Changes to the reference recipe require a regression test and must not overwrite
`configs/reproduction/sf73jk43.yaml`. Add ablations or task extensions as new YAML files.
Run `python -m pytest`, `python -m py_compile scripts/*.py`, and `bash -n scripts/*.sh` before
opening a pull request. Never commit model weights, simulator assets, videos, W&B runs, or
machine-local environment files.
