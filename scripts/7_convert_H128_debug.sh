#!/usr/bin/env bash
source env.sh

$DFL_PYTHON "$DFL_SRC/main.py" convert \
    --input-dir "$DFL_WORKSPACE/data_dst" \
    --output-dir "$DFL_WORKSPACE/data_dst/merged" \
    --aligned-dir "$DFL_WORKSPACE/data_dst/aligned" \
    --model-dir "$DFL_WORKSPACE/model" \
    --model H128 \
    --debug
    
