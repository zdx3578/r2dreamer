#!/bin/bash

git_require_synced_repo() {
  local repo_root=$1
  local allow_dirty=${ALLOW_DIRTY_RUN:-0}
  local head origin_head

  if ! head=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null); then
    echo "git sync guard: unable to resolve HEAD for $repo_root" >&2
    return 1
  fi
  if ! origin_head=$(git -C "$repo_root" rev-parse origin/main 2>/dev/null); then
    echo "git sync guard: unable to resolve origin/main for $repo_root" >&2
    return 1
  fi

  if [ "$head" != "$origin_head" ]; then
    echo "git sync guard: HEAD != origin/main" >&2
    echo "  repo: $repo_root" >&2
    echo "  head: $head" >&2
    echo "  origin/main: $origin_head" >&2
    echo "Run 'git pull --ff-only' before launching experiments." >&2
    return 1
  fi

  if [ "$allow_dirty" != "1" ] && [ -n "$(git -C "$repo_root" status --short 2>/dev/null)" ]; then
    echo "git sync guard: worktree is dirty for $repo_root" >&2
    echo "Commit/stash changes before launching experiments, or set ALLOW_DIRTY_RUN=1 to override." >&2
    return 1
  fi
}
