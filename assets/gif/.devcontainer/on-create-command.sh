#!/usr/bin/env bash
set -e

# Install the pre-commit checks
pre-commit install --config .pre-commit-config.yaml --hook-type pre-commit --hook-type pre-push