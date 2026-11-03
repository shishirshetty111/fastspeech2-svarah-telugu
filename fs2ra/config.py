"""Load project/<Lang>/config.yaml and split it into the three upstream dicts."""
import copy
import os

import yaml

from fs2ra.hf import REPO_ROOT
from fs2ra.sources import DATA_ROOT

OUTPUT_ROOT = os.environ.get("FS2_OUTPUT_ROOT", os.path.join(REPO_ROOT, "output"))


def _resolve(obj):
    if isinstance(obj, dict):
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("$"):
        for key, root in (("$DATA", DATA_ROOT), ("$REPO", REPO_ROOT), ("$OUT", OUTPUT_ROOT)):
            if obj == key or obj.startswith(key + "/"):
                return os.path.normpath(root + obj[len(key):])
    return obj


def load_config(path):
    """Returns the full resolved config dict."""
    with open(path, encoding="utf-8") as f:
        return _resolve(yaml.safe_load(f))


def upstream_configs(cfg):
    """(preprocess_config, model_config, train_config) in upstream format."""
    return copy.deepcopy(cfg["preprocess"]), copy.deepcopy(cfg["model"]), copy.deepcopy(cfg["train"])
