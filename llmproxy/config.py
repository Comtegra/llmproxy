import importlib.resources
import os
import sys
import tomllib


class ConfigError(Exception):
    pass


def validate(cfg):
    for name, meta in cfg.get("backends", {}).items():
        if "max_model_len" not in meta:
            continue

        value = meta["max_model_len"]
        if type(value) is not int or value <= 0:
            raise ConfigError(
                'Backend "%s" max_model_len must be a positive integer' %
                name)


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
            try:
                cfg = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise ConfigError("Failed parsing config: %s" % e) from e
            print("Loaded config from \"%s\"" % p, file=sys.stderr)
            cfg["_path"] = p

        if db_uri := os.environ.get("LLMPROXY_DB_URI"):
            if "db" not in cfg:
                cfg["db"] = {}
            cfg["db"]["uri"] = db_uri
            print("Loaded database URI from the LLMPROXY_DB_URI env var",
                file=sys.stderr)

        validate(cfg)

        return cfg
