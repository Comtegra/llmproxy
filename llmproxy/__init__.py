import importlib.metadata

try:
    __version__ = importlib.metadata.version("llmproxy")
except importlib.metadata.PackageNotFoundError:  # bare source checkout
    __version__ = "0.0.0-dev"
