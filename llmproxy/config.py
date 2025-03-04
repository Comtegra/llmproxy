import os
import sys
import tomllib


def load(path=None):
    choices = [path, os.environ.get("LLMPROXY_CONFIG"), "config.toml"]

    for p in choices:
        if not p:
            continue
        with open(p, "rb") as f:
            cfg = tomllib.load(f)
            print("Loaded config from \"%s\"" % p, file=sys.stderr)
            cfg["_path"] = p
            return cfg
