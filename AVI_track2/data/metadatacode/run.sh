#!/usr/bin/env bash
set -euo pipefail

python ./data/metadatacode/csvcode.py \
  --input ./data/test_data.csv \
  --output ./data/metadatacode/test_data_meta.csv
