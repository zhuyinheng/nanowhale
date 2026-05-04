"""Data preparation utilities for pretraining and SFT."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def preview_fineweb_edu(n=5):
    """Preview FineWeb-Edu samples."""
    print("Previewing FineWeb-Edu...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    for i, sample in enumerate(ds):
        if i >= n:
            break
        text = sample["text"][:200]
        print(f"\n--- Sample {i+1} ---")
        print(f"Score: {sample.get('score', 'N/A')}")
        print(f"Tokens: {sample.get('token_count', 'N/A')}")
        print(f"Text: {text}...")


def preview_smoltalk(n=5):
    """Preview SmolTalk samples."""
    print("\nPreviewing SmolTalk...")
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    for i in range(min(n, len(ds))):
        sample = ds[i]
        msgs = sample["messages"]
        print(f"\n--- Sample {i+1} ({len(msgs)} turns) ---")
        for m in msgs[:3]:
            content = m["content"][:100]
            print(f"  [{m['role']}] {content}...")


def check_tokenizer_stats(tokenizer_path="tokenizer"):
    """Check tokenizer statistics on sample data."""
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"\nTokenizer stats:")
    print(f"  Vocab size: {tok.vocab_size}")
    
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    total_tokens = 0
    total_chars = 0
    for i, sample in enumerate(ds):
        if i >= 100:
            break
        text = sample["text"]
        ids = tok.encode(text)
        total_tokens += len(ids)
        total_chars += len(text)
    
    print(f"  Avg tokens per sample (100 samples): {total_tokens/100:.0f}")
    print(f"  Avg chars per token: {total_chars/total_tokens:.2f}")


if __name__ == "__main__":
    preview_fineweb_edu()
    preview_smoltalk()
    check_tokenizer_stats()
