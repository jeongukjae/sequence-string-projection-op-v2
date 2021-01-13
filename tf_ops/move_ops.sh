#!/bin/bash

RUNFILES_DIR=$(pwd)
cp -f "${RUNFILES_DIR}/tf_ops/libsequence_string_projection_op_v2_py_gen_op.so" \
  "${BUILD_WORKSPACE_DIRECTORY}/tf_ops"
cp -f "${RUNFILES_DIR}/tf_ops/sequence_string_projection_op_v2.py" \
  "${BUILD_WORKSPACE_DIRECTORY}/tf_ops"
