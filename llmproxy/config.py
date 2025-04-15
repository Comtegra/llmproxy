import importlib.resources
import os
import sys
import tomllib


def load(path=None, create=False):
    choices = [path, os.environ.get("LLMPROXY_CONFIG"), "config.toml"]

    for p in choices:
        if not p:
            continue

        if create:
            if not os.path.lexists(p):
                config = importlib.resources.files("llmproxy") \
                    .joinpath("config.toml").read_bytes()
                try:
                    with open(p, "wb") as f:
                        f.write(config)
                    print("Created default config file", file=sys.stderr)
                except OSError as e:
                    print("Failed creating default config file:", e,
                        file=sys.stderr)
            else:
                print("Skipped creating default config because it already exists",
                    file=sys.stderr)

        with open(p, "rb") as f:
            cfg = tomllib.load(f)
            print("Loaded config from \"%s\"" % p, file=sys.stderr)
            cfg["_path"] = p
            return cfg
