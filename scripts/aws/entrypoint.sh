#!/bin/bash

source /app/.rye/env
source /app/.rye/global/.venv/bin/activate

python sm_entrypoint.py "$@"
