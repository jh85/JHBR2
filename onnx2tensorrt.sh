#!/bin/bash

ONNX_FILE="$1"
if [ -z "$ONNX_FILE" ]; then
    echo "Usage: $0 model.onnx"
    exit 1
fi
TRT_FILE="${ONNX_FILE%.onnx}.engine"
trtexec --onnx="$ONNX_FILE" --saveEngine="$TRT_FILE" --fp16
