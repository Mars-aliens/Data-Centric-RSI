#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_CONTEXT="$PROJECT_ROOT/docker/candidate"

: "${PYTORCH_BASE_IMAGE:?Set PYTORCH_BASE_IMAGE to a PyTorch CUDA runtime image pinned with @sha256:<digest>}"
if [[ ! "$PYTORCH_BASE_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  printf 'PYTORCH_BASE_IMAGE must end with @sha256:<64 lowercase hex characters>\n' >&2
  exit 2
fi

IMAGE_TAG="${T1_CANDIDATE_IMAGE_TAG:-mle-lego-t1-candidate:local}"
docker build --pull \
  --build-arg "PYTORCH_BASE_IMAGE=$PYTORCH_BASE_IMAGE" \
  --file "$BUILD_CONTEXT/Dockerfile" \
  --tag "$IMAGE_TAG" \
  "$BUILD_CONTEXT"

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG")"
if [[ ! "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'Docker returned an unexpected immutable image ID: %s\n' "$IMAGE_ID" >&2
  exit 1
fi
printf 'image_tag=%s\nimage_id=%s\n' "$IMAGE_TAG" "$IMAGE_ID"
