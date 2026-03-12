#!/bin/bash

git_run_metadata_line() {
  local repo_root=$1
  local head branch dirty origin_head

  if ! head=$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null); then
    echo "git_version=unknown"
    return 0
  fi

  branch=$(git -C "$repo_root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
  if [ -n "$(git -C "$repo_root" status --short 2>/dev/null)" ]; then
    dirty=dirty
  else
    dirty=clean
  fi
  origin_head=$(git -C "$repo_root" rev-parse --short origin/main 2>/dev/null || echo "unknown")

  echo "git_branch=${branch} git_head=${head} git_worktree=${dirty} git_origin_main=${origin_head}"
}

print_git_run_metadata() {
  local repo_root=$1
  local label=${2:-run}
  echo "[$(date '+%F %T')] ${label}: $(git_run_metadata_line "$repo_root")"
}
