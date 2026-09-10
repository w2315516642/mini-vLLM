"""Prepare fixed raw-prompt benchmarks without changing mini-vLLM sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


def records(path):
    with Path(path).open(encoding="utf-8") as f:
        if Path(path).suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array")
    return data


def family(identifier):
    # ShareGPT split IDs use a numeric suffix, e.g. original_0, original_9.
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Every conversation must have a nonempty string ID")
    return re.sub(r"_\d+$", "", identifier)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def user_texts(record):
    for turn in record.get("conversations", []):
        if turn.get("role", turn.get("from")) in ("user", "human"):
            text = turn.get("content", turn.get("value"))
            if isinstance(text, str) and text.strip():
                yield text


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(source, excluded, encode, count, length, seed):
    families = {family(row.get("id")) for row in excluded}
    texts = {normalized(t) for row in excluded for t in user_texts(row)}
    prefixes = set()
    for row in excluded:
        for text in user_texts(row):
            ids = encode(text)
            if len(ids) >= length:
                prefixes.add(tuple(ids[:length]))
    candidates = list(source)
    random.Random(seed).shuffle(candidates)
    selected, seen = [], set()
    for row in candidates:
        if family(row.get("id")) in families:
            continue
        turns = row.get("conversations", [])
        if not turns or turns[0].get("from", turns[0].get("role")) not in ("human", "user"):
            continue
        prompt = turns[0].get("value", turns[0].get("content"))
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if any(normalized(t) in texts for t in user_texts(row)):
            continue
        ids = encode(prompt)
        if len(ids) < length:
            continue
        key = tuple(ids[:length])
        if key in prefixes or key in seen:
            continue
        seen.add(key)
        selected.append({"id": row["id"], "prompt": prompt})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} unique eligible prompts; need {count}. No resampling.")
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--exclude", required=True, nargs="+")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.count <= 0 or args.length <= 0:
        p.error("count and length must be positive")
    out = Path(args.output)
    manifest = out.with_suffix(".manifest.json")
    if out.exists() or manifest.exists():
        raise FileExistsError("Use a fresh output path to preserve the fixed benchmark")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    excluded = [row for path in args.exclude for row in records(path)]
    rows = select(records(args.source), excluded,
                  lambda s: tok.encode(s, add_special_tokens=False),
                  args.count, args.length, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"settings": vars(args), "source_sha256": digest(args.source),
              "excluded_sha256": {s: digest(s) for s in args.exclude},
              "output_sha256": digest(out), "unique_token_prefixes": len(rows),
              "ids": [r["id"] for r in rows],
              "limitations": "Raw first-user prompts, no chat template. ID-family and exact normalized text/token-prefix exclusion, not semantic deduplication."}
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} unique prompts to {out}; manifest: {manifest}")


if __name__ == "__main__":
    main()
