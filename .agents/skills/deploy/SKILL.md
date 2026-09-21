---
name: deploy
description: "Full release workflow for Comtegra-built applications (llmproxy, microservices). Use when the user asks to deploy, release, or ship a repo/app, cut a release, or push a build/tag. Drives the whole process: repo state report, changelog since last tag, test suite (incl. env-gated e2e), staging job if available, otherwise semver tag + push for the prod image build."
---

# Deploy (Comtegra release workflow)

Run this end-to-end when asked to deploy/release an app in a given repo.
Work phase by phase; after each phase give a short progress report, and end
with a full summary of everything done. Ask the user only where input is
genuinely needed (e2e credentials, version bump confirmation).

Guardrails — hard stops:

* **Never deploy with a dirty tree or failing tests.** Commit (with the
  user's OK) or ask before proceeding.
* **Never commit secrets.** E2E credentials arrive via environment
  variables at runtime only (see Phase 3).
* **Never force-push tags or reuse an existing tag.** If a tag was pushed
  by mistake, tell the user; do not rewrite it.
* **Never invent a staging pipeline.** If detection is ambiguous, say so
  and fall back to asking.

## Phase 0 — Preflight: repo state

In the target repo:

```bash
git status && git log --oneline -3
git pull --ff-only            # never merge; abort and report on divergence
```

* Dirty tree → report it, offer to commit; don't stash silently.
* Diverged from origin → stop and report.
* If the repo has no venv but tests need one → report; don't install
  packages without asking.

## Phase 1 — Report: what is being released

```bash
git describe --tags --abbrev=0   # last tag
git log --oneline <last-tag>..HEAD
git diff --stat <last-tag>..HEAD
```

Produce a short human changelog: features / fixes / new endpoints / config
options / breaking changes. This drives the version bump in Phase 5:

| Changes since last tag | Bump |
|---|---|
| Breaking (API, config, billing semantics) | major |
| New feature / endpoint / metric | minor |
| Fixes / tests / docs only | patch |
| Nothing new (last tag == HEAD) | nothing to release — stop and say so |

## Phase 2 — Test suite

Detect the runner for THIS repo and prefer its native one:

* pytest project (pytest config or `tests/` with pytest style) →
  `.venv/bin/python -m pytest tests/ -q`
* plain unittest (CI runs `python -m unittest`) →
  `.venv/bin/python -m unittest`

Note: llmproxy's GitLab CI test stage runs `python3 -m unittest`, so all
tests must stay unittest-discoverable (pytest-only plugins in *assertions*
are fine as long as `python -m unittest` still collects and passes them;
verify both locally before releasing).

Rules:

* Failing tests → stop, fix or ask. Never release red.
* Report counts (passed/skipped) in the summary.
* Skipped tests matter: investigate why they skipped before Phase 3.

## Phase 3 — E2E tests (env-gated suites)

Find env-gated e2e suites that were skipped in Phase 2:

```bash
grep -rn "skipIf\|skipUnless" tests/ | grep -i "environ"
grep -rn "os.environ.get" tests/
```

For each gated suite, report: which ENVs it needs and what they point at.
llmproxy example: `tests/test_e2e_marker.py` needs:

* `MARKER_URL` — production marker microservice base URL
* `MARKER_TOKEN` — the microservice APP_TOKEN
* `MARKER_DEVICE` — billing device label (default `cpu`)

Then **ask the user** whether to run e2e and to provide the values. Run:

```bash
MARKER_URL=... MARKER_TOKEN=... .venv/bin/python -m pytest tests/test_e2e_marker.py -v
```

* Values are passed inline for the pytest process only — never written to
  any file, never echoed into committed content.
* If the user declines, proceed with unit tests only and note in the final
  report that e2e was not run.
* E2E hits production services — it is always opt-in, never automatic.

## Phase 4 — Staging vs prod path

Detect a staging pipeline for this repo:

```bash
cat .gitlab-ci.yml 2>/dev/null        # stages/jobs named staging/dev/preview?
ls .github/workflows 2>/dev/null
ls scripts/ | grep -i "deploy\|stage"
grep -rn "staging" k8s/ compose.yaml scripts/ 2>/dev/null
```

* **Staging job found** (CI stage/job targeting a staging environment, a
  repo deploy script, manifests pointing at a staging namespace): trigger
  it the way the repo defines (`glab`, its own script, or a manual
  command), watch it finish, report the staging URL / rollout status.
  If triggering requires credentials or an unknown mechanism → ask the
  user; do not improvise.
* **No staging pipeline** (current state of llmproxy: GitLab CI has only
  `test` on push and `image` on tags): skip staging, proceed to Phase 5 —
  tag and push; the tag itself triggers the prod image build.

## Phase 5 — Tag and push (prod build)

Conventions (llmproxy/Comtegra, verify per repo before first use):

* Plain semver, **no `v` prefix**: `1.10.0`, never `v1.10.0`.
* **Lightweight tags** (`git tag X`), not annotated.
* Tag the release commit on the default branch, push branch + tag:

```bash
git tag <new-version>
git push origin <default-branch>
git push origin <new-version>
```

Propose the bump from the Phase 1 table and confirm with the user if the
call is close (e.g. feature-vs-fix); routine bumps can proceed but must be
stated in the report. Never retag an existing version.

What the tag triggers (llmproxy, from `.gitlab-ci.yml`): kaniko builds the
Docker image `${CI_REGISTRY_IMAGE}:${CI_COMMIT_TAG}`. The running
deployment (k8s manifests live in the ops repo, not here) is updated
separately by whoever owns that — mention it in the report so the user
knows the release isn't live until rollout.

## Phase 6 — Final report

Summarize:

1. Repo state at start (branch, cleanliness, sync)
2. Changelog since previous tag
3. Test results (unit counts; e2e run-or-not, with which envs)
4. Staging: ran / not present (with evidence)
5. Version decision rationale, tag pushed, what pipeline it triggers
6. Follow-ups: rollout step, CI status link if known, any caveats
   (e.g. skipped tests, known gaps)

If `glab` or the GitLab API is available, offer to watch the pipeline for
the new tag. For post-deploy verification of cluster workloads, the
`k8s-readonly` skill can inspect the CGC cluster (read-only; requires VPN).
