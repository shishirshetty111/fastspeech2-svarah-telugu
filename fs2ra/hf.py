"""Hugging Face access helpers. The token is read from the HF_TOKEN environment
variable or, failing that, from a git-ignored `.env` file; it is never printed."""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_hf_token(env_file=os.path.join(REPO_ROOT, ".env")):
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*HF_TOKEN\s*=\s*['\"]?([^'\"\s]+)", line)
                if m:
                    return m.group(1)
    return None
