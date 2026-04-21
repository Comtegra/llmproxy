import importlib.resources
import os
import sys
import tomllib


_VALID_TYPES = {"chat", "embedding", "audio"}
_REQUIRED_STR_FIELDS = ("url", "token", "device", "model")


def validate_backends(backends):
    errors = []
    for name, cfg in backends.items():
        for field in _REQUIRED_STR_FIELDS:
            if field not in cfg:
                errors.append("backend %r: missing required field %r"
                    % (name, field))
                continue
            v = cfg[field]
            if not isinstance(v, str):
                errors.append("backend %r: %s must be str, got %s"
                    % (name, field, type(v).__name__))
            elif not v or v != v.strip():
                errors.append(
                    "backend %r: %s must be non-empty and trimmed, got %r"
                    % (name, field, v))

        if "type" not in cfg:
            errors.append("backend %r: missing required field 'type'" % name)

        t = cfg.get("type")
        if t is not None and t not in _VALID_TYPES:
            errors.append("backend %r: type %r not in %s"
                % (name, t, sorted(_VALID_TYPES)))

        if (cl := cfg.get("context_length")) is not None:
            if not isinstance(cl, int):
                errors.append("backend %r: context_length must be int, got %s"
                    % (name, type(cl).__name__))
            elif t == "audio":
                errors.append(
                    "backend %r: context_length not applicable for type=audio"
                    % name)

        if (q := cfg.get("quantization")) is not None:
            if not isinstance(q, str):
                errors.append("backend %r: quantization must be str, got %s"
                    % (name, type(q).__name__))
            elif not q or q != q.strip():
                errors.append(
                    "backend %r: quantization must be non-empty and trimmed,"
                    " got %r" % (name, q))

    if errors:
        raise ValueError("Invalid backend config:\n  " + "\n  ".join(errors))


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

        validate_backends(cfg.get("backends", {}))

        if db_uri := os.environ.get("LLMPROXY_DB_URI"):
            if "db" not in cfg:
                cfg["db"] = {}
            cfg["db"]["uri"] = db_uri
            print("Loaded database URI from the LLMPROXY_DB_URI env var",
                file=sys.stderr)

        return cfg
