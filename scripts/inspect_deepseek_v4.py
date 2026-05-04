"""Inspect DeepSeek-V4 repo structure and config."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download, list_repo_files
import json

def inspect_repo(repo_id):
    print(f"\n{'='*60}")
    print(f"Inspecting: {repo_id}")
    print(f"{'='*60}")
    
    files = list_repo_files(repo_id)
    print(f"\nFiles ({len(files)}):")
    for f in sorted(files):
        if not f.startswith("model-"):  # Skip weight shards
            print(f"  {f}")
    
    weight_files = [f for f in files if f.startswith("model-")]
    print(f"\n  ... plus {len(weight_files)} weight shard files")
    
    # Download and print config
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\nConfig:")
    for k, v in config.items():
        if k not in ("quantization_config", "compress_ratios"):
            print(f"  {k}: {v}")
        elif k == "compress_ratios":
            print(f"  {k}: [{v[0]}, {v[1]}, ... {v[-2]}, {v[-1]}] (len={len(v)})")


if __name__ == "__main__":
    for repo in ["deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash"]:
        inspect_repo(repo)
