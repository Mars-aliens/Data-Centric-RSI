#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET="$ROOT/baselines/synthesis"

fetch_repo() {
  local name="$1"
  local url="$2"
  local revision="$3"
  shift 3
  local path="$TARGET/$name"

  if [[ -e "$path" ]]; then
    if [[ ! -d "$path/.git" ]]; then
      printf 'Refusing to overwrite non-git path: %s\n' "$path" >&2
      return 1
    fi
    local current
    current="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$current" != "$revision" ]]; then
      printf 'Revision mismatch in %s: expected %s, found %s\n' \
        "$path" "$revision" "${current:-no checked out commit}" >&2
      return 1
    fi
    printf '%s already at %s\n' "$name" "$revision"
    return 0
  fi

  git clone --filter=blob:none --no-checkout "$url" "$path"
  git -C "$path" sparse-checkout init --no-cone
  git -C "$path" sparse-checkout set --no-cone -- "$@"
  git -C "$path" checkout "$revision"

  local current
  current="$(git -C "$path" rev-parse HEAD)"
  [[ "$current" == "$revision" ]]
}

mkdir -p "$TARGET"

fetch_repo \
  self-instruct-upstream \
  https://github.com/yizhongw/self-instruct.git \
  0b26ccaa415992100fa32df62d41b994cf928e23 \
  /README.md /LICENSE /requirements.txt /scripts/ /self_instruct/

fetch_repo \
  wizardlm-upstream \
  https://github.com/nlpxucan/WizardLM.git \
  cd4b4cc022b6426aee62d51f7ae99a4a6ec9aaef \
  /README.md /Evol_Instruct/ /WizardLM/CODE_LICENSE

printf 'Pinned synthesis baseline sources are ready under %s\n' "$TARGET"
