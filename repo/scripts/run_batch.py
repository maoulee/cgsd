#!/usr/bin/env python3
"""YAML batch launcher (user request 2026-10-10): the env-prefix recipe is
error-prone — a dead flag (SEQ_GHOST_EDGES) rode along for weeks because
every batch copy-pasted the same prefix. The YAML profile is now the single
source of truth; this runner resolves it and spawns the module as a
subprocess (so the child's own env-echo log header reflects the profile).

Profile schema (config/batches/*.yaml):

    name: r9a_hard_36x3          # {name} templates into out/log
    desc: ...                    # free text
    module: kgqa.rl.seq_rollout  # default
    python: /opt/conda/envs/rl/bin/python   # default: this interp
    args: []                     # extra CLI args to the module
    env:                         # exported verbatim (strings)
      SPLIT: test_v4
      ...
    out: reports/{name}.json     # optional; refusal guard unless allow_resume
    log: logs/{name}.log         # default logs/{name}.log
    allow_resume: false          # OUT must not pre-exist (RESUME-append trap)

Usage:
    python scripts/run_batch.py config/batches/hp_36x3.yaml [--name r9b_hard_36x3]
           [--env KEY=VAL ...] [--dry-run]

`--dry-run` prints the resolved env + command without executing. `--name`
overrides the profile name (out/log re-template). `--env` overrides single
vars (last wins) — replay recipes use this for RESULTS/REPLAY_ONLY.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

_REQUIRED = ("name",)
_DEFAULT_PYTHON = sys.executable


def load_profile(path: Path, name_override=None, env_overrides=None):
    """Resolve a profile into (name, env dict, [argv]) — pure, testable."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"profile {path} is not a mapping")
    for key in _REQUIRED:
        if not raw.get(key):
            raise SystemExit(f"profile {path} misses required key {key!r}")
    name = name_override or raw["name"]
    env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()}
    if env_overrides:
        env.update(env_overrides)
    out = raw.get("out")
    if out:
        out = out.format(name=name)
        env["OUT"] = out
    else:
        out = None            # empty/absent out: no OUT env, no guard
    log = (raw.get("log") or "logs/{name}.log").format(name=name)
    argv = [raw.get("python") or _DEFAULT_PYTHON, "-m",
            raw.get("module") or "kgqa.rl.seq_rollout"]
    argv += [str(a) for a in (raw.get("args") or [])]
    return {"name": name, "env": env, "argv": argv,
            "out": out, "log": log,
            "allow_resume": bool(raw.get("allow_resume", False))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile", type=Path)
    ap.add_argument("--name", help="override profile name (re-templates out/log)")
    ap.add_argument("--env", action="append", default=[],
                    metavar="KEY=VAL", help="override one env var (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    overrides = {}
    for kv in args.env:
        k, _, v = kv.partition("=")
        if not k:
            raise SystemExit(f"bad --env {kv!r}")
        overrides[k] = v
    prof = load_profile(args.profile, name_override=args.name,
                        env_overrides=overrides)

    # RESUME-append trap (SESSION_MEMORY): a pre-existing OUT silently
    # appends when CASE_BATCH > case count — refuse unless allow_resume.
    if prof["out"]:
        outp = ROOT / prof["out"]
        if outp.exists() and not prof["allow_resume"]:
            raise SystemExit(
                f"OUT {outp} already exists (RESUME would APPEND). "
                f"Delete it or set allow_resume: true.")
        outp.parent.mkdir(parents=True, exist_ok=True)
    logp = ROOT / prof["log"]
    logp.parent.mkdir(parents=True, exist_ok=True)

    print(f"═══ batch {prof['name']} (profile {args.profile}) ═══", flush=True)
    print("  env: " + " ".join(f"{k}={prof['env'][k]}"
                               for k in sorted(prof['env'])), flush=True)
    print(f"  cmd: {' '.join(prof['argv'])}", flush=True)
    print(f"  out: {prof['out']}  log: {prof['log']}", flush=True)
    if args.dry_run:
        print("  (dry-run, not executing)", flush=True)
        return

    child_env = dict(os.environ)
    child_env.update(prof["env"])
    with open(logp, "w") as lf:
        rc = subprocess.call(prof["argv"], cwd=str(ROOT), env=child_env,
                             stdout=lf, stderr=subprocess.STDOUT)
    print(f"  exit={rc} → {prof['log']}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
