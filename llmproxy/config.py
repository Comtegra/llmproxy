import importlib.resources
import os
import sys
import tomllib


class ConfigError(Exception):
    pass


def _validate_rate_limit(rl, prefix=""):
    """Validate a ``[rate_limit]`` / ``[backends.X.rate_limit]`` block.

    Phase 1 checks ``rpm`` and ``concurrency`` only; phase 2 appends the two
    ``*_tpd`` keys to the same loop. Each, if present, must be a non-negative
    int (``type(v) is not int`` excludes ``bool``). 0 = unlimited, NULL/absent
    = fall through. ``prefix`` labels backend errors.
    """
    for key in ("rpm", "concurrency"):
        if key in rl:
            value = rl[key]
            if type(value) is not int or value < 0:
                raise ConfigError(
                    "%srate_limit.%s must be a non-negative integer" %
                    (prefix, key))


def validate(cfg):
    for key in ("client_max_size", "max_json_body"):
        if key in cfg:
            value = cfg[key]
            if type(value) is not int or value <= 0:
                raise ConfigError("%s must be a positive integer" % key)

    if "auth_cache_ttl" in cfg:
        value = cfg["auth_cache_ttl"]
        if type(value) is not int or value < 0:
            raise ConfigError(
                "auth_cache_ttl must be a non-negative integer")

    if "provenance" in cfg:
        # Must be a table: `provenance = true` would otherwise only blow up at
        # request time inside cfg.get("provenance", {}).get(...).
        p = cfg["provenance"]
        if type(p) is not dict:
            raise ConfigError("provenance must be a table")
        if "enabled" in p and type(p["enabled"]) is not bool:
            raise ConfigError("provenance.enabled must be a boolean")
        if "generator" in p:
            value = p["generator"]
            if type(value) is not str or not value.strip():
                raise ConfigError(
                    "provenance.generator must be a non-empty string")

    _validate_rate_limit(cfg.get("rate_limit", {}), "")

    for name, meta in cfg.get("backends", {}).items():
        if "max_model_len" in meta:
            value = meta["max_model_len"]
            if type(value) is not int or value <= 0:
                raise ConfigError(
                    'Backend "%s" max_model_len must be a positive integer' %
                    name)

        if "timeout" in meta:
            value = meta["timeout"]
            if type(value) not in (int, float) or value <= 0:
                raise ConfigError(
                    'Backend "%s" timeout must be a positive number' % name)

        _validate_rate_limit(meta.get("rate_limit", {}),
            'Backend "%s" ' % name)


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
