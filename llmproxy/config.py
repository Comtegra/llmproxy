import importlib.resources
import math
import os
import sys
import tomllib


class ConfigError(Exception):
    pass


# Every backend must declare one of these as its ``type``. Each endpoint
# handler passes the type it serves to proxy.request(), which rejects a model
# of any other type before the request reaches the backend.
BACKEND_TYPES = ("chat", "embedding", "transcription", "conversion", "decision")

DECISION_LIMITS = {"max_questions": 16, "max_branches": 64,
    "max_response_bytes": 4 * 1024 * 1024}


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
    _validate_budget_options(cfg.get("budget", {}), "budget", global_options=True)
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
        # Required: without it the proxy cannot tell which endpoints may reach
        # this backend (e.g. chat completions sent to an embedding model).
        if meta.get("type") not in BACKEND_TYPES:
            raise ConfigError('Backend "%s" type must be one of: %s' %
                (name, ", ".join(BACKEND_TYPES)))

        if meta["type"] == "decision":
            for key in ("url", "model", "device"):
                if type(meta.get(key)) is not str or not meta[key].strip():
                    raise ConfigError('Backend "%s" %s must be a non-empty string' % (name, key))
            if "token" in meta and type(meta["token"]) is not str:
                raise ConfigError('Backend "%s" token must be a string' % name)
            if "timeout" in meta and (type(meta["timeout"]) not in (int, float)
                    or not math.isfinite(meta["timeout"]) or meta["timeout"] <= 0):
                raise ConfigError('Backend "%s" timeout must be finite and positive' % name)
        if meta["type"] == "conversion":
            # Besides authenticating the proxy to the backend, the token keys
            # the signature of background conversion ids (conversion_id.py):
            # with an empty one anybody could mint an id for any job.
            if type(meta.get("token")) is not str or not meta["token"].strip():
                raise ConfigError('Backend "%s" token must be a non-empty string' % name)
        if "decision_limits" in meta:
            limits = meta["decision_limits"]
            if meta["type"] != "decision" or type(limits) is not dict:
                raise ConfigError('Backend "%s" decision_limits requires a decision table' % name)
            if set(limits) - set(DECISION_LIMITS):
                raise ConfigError('Backend "%s" unknown decision_limits option' % name)
            for key, value in limits.items():
                if type(value) is not int or value <= 0:
                    raise ConfigError('Backend "%s" decision_limits.%s must be a positive integer' % (name, key))

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
        if "output_budget" in meta or "reasoning" in meta:
            if meta["type"] != "chat":
                raise ConfigError('Backend "%s" budgets require type chat' % name)
            _validate_budget_options(meta.get("output_budget", {}),
                'Backend "%s" output_budget' % name)
            _validate_reasoning(meta.get("reasoning", {}), name)
            policy = meta.get("output_budget", {})
            if meta.get("reasoning", {}).get("native_budget") and "output_budget" not in meta:
                raise ConfigError('Backend "%s" native reasoning budgets require output_budget' % name)
            default, ceiling = policy.get("default_max_tokens"), policy.get("ceiling")
            if default is not None and ceiling is not None and default > ceiling:
                raise ConfigError('Backend "%s" default budget exceeds ceiling' % name)
            if ceiling is not None and ceiling > meta.get("max_model_len", ceiling):
                raise ConfigError('Backend "%s" ceiling exceeds max_model_len' % name)
            reserve = meta.get("reasoning", {}).get("reserve_content_tokens")
            if reserve is not None and default is not None and reserve >= default:
                raise ConfigError('Backend "%s" reserve must be below default budget' % name)


def _validate_budget_options(options, prefix, global_options=False):
    if type(options) is not dict:
        raise ConfigError("%s must be a table" % prefix)
    flags = ("enabled", "echo_headers", "echo_body") if global_options else ("apply_default", "clamp")
    numbers = () if global_options else ("default_max_tokens", "ceiling", "backend_default_max_tokens")
    allowed = set(flags + numbers + (() if global_options else ("mode",)))
    if set(options) - allowed:
        raise ConfigError("%s contains unknown options: %s" %
            (prefix, ", ".join(sorted(set(options) - allowed))))
    for key in flags:
        if key in options and type(options[key]) is not bool:
            raise ConfigError("%s.%s must be a boolean" % (prefix, key))
    for key in numbers:
        if key in options and (type(options[key]) is not int or options[key] <= 0):
            raise ConfigError("%s.%s must be a positive integer" % (prefix, key))
    if not global_options:
        if options.get("mode", "observe") not in ("observe", "enforce"):
            raise ConfigError("%s.mode must be observe or enforce" % prefix)
        for flag, required in (("apply_default", "default_max_tokens"), ("clamp", "ceiling")):
            if options.get(flag) and required not in options:
                raise ConfigError("%s.%s requires %s" % (prefix, flag, required))


def _validate_reasoning(options, name):
    prefix = 'Backend "%s" reasoning' % name
    if type(options) is not dict:
        raise ConfigError("%s must be a table" % prefix)
    allowed = {"native_budget", "endpoints", "reserve_content_tokens", "effort_budgets", "default_effort"}
    if set(options) - allowed:
        raise ConfigError("%s contains unknown options" % prefix)
    if "native_budget" in options and type(options["native_budget"]) is not bool:
        raise ConfigError("%s.native_budget must be a boolean" % prefix)
    endpoints = options.get("endpoints", ["chat"])
    if (type(endpoints) is not list or not endpoints
            or endpoints != ["chat"]):
        raise ConfigError("%s.endpoints must be [chat]; native reasoning budgets "
            "are not supported on Responses/Messages" % prefix)
    reserve = options.get("reserve_content_tokens")
    if reserve is not None and (type(reserve) is not int or reserve <= 0):
        raise ConfigError("%s.reserve_content_tokens must be a positive integer" % prefix)
    efforts = options.get("effort_budgets", {})
    if type(efforts) is not dict:
        raise ConfigError("%s.effort_budgets must be a table" % prefix)
    for effort, value in efforts.items():
        if (effort not in ("minimal", "low", "medium", "high", "xhigh")
                or type(value) is not int or value <= 0):
            raise ConfigError("%s.effort_budgets requires known efforts and positive integers" % prefix)
    if "default_effort" in options and (type(options["default_effort"]) is not str
            or options["default_effort"] not in efforts):
        raise ConfigError("%s.default_effort requires a matching effort_budgets entry" % prefix)


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
