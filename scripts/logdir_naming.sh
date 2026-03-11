#!/bin/bash

logdir_run_tag() {
  local version timestamp
  version=${RUN_VERSION:-v1}
  timestamp=${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}
  echo "${RUN_TAG:-${version}_${timestamp}}"
}

default_versioned_logdir() {
  local root_dir prefix
  root_dir=$1
  prefix=$2
  echo "${root_dir}/logdir/${prefix}_$(logdir_run_tag)"
}
