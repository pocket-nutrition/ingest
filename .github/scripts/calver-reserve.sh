#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
#
# Reserves the next CalVer tag (vYYYY.M.N) on the commit being built and prints the version
# (without the leading "v") on stdout. N restarts at 1 every month.
#
# The tag is created through the API rather than `git push`, because the API refuses to create a ref
# that already exists. That makes the reservation atomic: when two runs race, one wins and the other
# recomputes and tries the next number, so no two commits ever share a version.
#
# Needs: gh (preinstalled on GitHub runners), GH_TOKEN with contents:write, GITHUB_REPOSITORY, GITHUB_SHA.
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"

prefix="$(date -u +%Y.%-m)"

for attempt in 1 2 3 4 5; do
  # Highest N already used this month. The trailing dot in the prefix stops "2026.1." from matching
  # "2026.11.". Non-numeric suffixes (e.g. a manual "v2026.11.1-rc1") are ignored.
  last="$(
    gh api --paginate "repos/${GITHUB_REPOSITORY}/git/matching-refs/tags/v${prefix}." --jq '.[].ref' \
      | sed -E "s|^refs/tags/v[0-9]+\.[0-9]+\.||" \
      | grep -E '^[0-9]+$' \
      | sort -n \
      | tail -n 1 || true
  )"
  version="${prefix}.$(( ${last:-0} + 1 ))"

  if gh api "repos/${GITHUB_REPOSITORY}/git/refs" \
       -f "ref=refs/tags/v${version}" -f "sha=${GITHUB_SHA}" --silent 2>/dev/null; then
    echo "${version}"
    exit 0
  fi

  echo "tag v${version} was taken by a concurrent run (attempt ${attempt}/5), retrying" >&2
  sleep $(( attempt * 2 ))
done

echo "could not reserve a CalVer tag after 5 attempts" >&2
exit 1
