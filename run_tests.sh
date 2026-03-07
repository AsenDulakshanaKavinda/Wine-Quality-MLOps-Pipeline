#!/bin/bash

# activate virtual environment (optional if already active)
source .venv/bin/activate

# set python path to project root
export PYTHONPATH=.

# run pytest
pytest -v