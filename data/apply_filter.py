"""
apply_filter.py

Apply filter_accepted.jsonl back to the SFT source jsonls and concatenate.

For each (source_jsonl, filter_accepted) pair, keep only rows whose
(case_id, turn_number) is in that filter's accepted keep_turns; then
concatenate the filtered outputs in the order the pairs were passed.
"""
import argparse
import json
import os


def load_keep(filter_accepted_path):
    """Return {case_id: set of int turn numbers} from a filter_accepted.jsonl file."""
    keep = {}
    with open(filter_accepted_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keep[r["case_id"]] = set(int(t) for t in r.get("keep_turns", []))
    return keep


def filter_source(source_path, keep, out_f):
    """Stream source_path; write lines whose (case_id, turn_number) is in keep.

    Returns (n_in, n_out).
    """
    n_in = n_out = 0
    with open(source_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            r = json.loads(line)
            if r.get("case_id") in keep and r.get("turn_number") in keep[r["case_id"]]:
                out_f.write(line if line.endswith("\n") else line + "\n")
                n_out += 1
    return n_in, n_out


def main():
    p = argparse.ArgumentParser(description="Filter each source jsonl by its own filter_accepted, then concatenate.")
    p.add_argument(
        "--pair", nargs=2, action="append", required=True,
        metavar=("SOURCE_JSONL", "FILTER_ACCEPTED"),
        help="A (source_jsonl, filter_accepted.jsonl) pair. Repeat for each source.",
    )
    p.add_argument("--output", required=True, help="Output filtered+concatenated JSONL.")
    args = p.parse_args()

    print(f"output: {args.output}")
    for src, flt in args.pair:
        print(f"  source={src}  filter={flt}")
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    total_in = total_out = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for src, flt in args.pair:
            keep = load_keep(flt)
            n_in, n_out = filter_source(src, keep, out_f)
            total_in  += n_in
            total_out += n_out
            print(f"  {os.path.basename(src):30s} {n_in:>6d} -> {n_out:>6d}  ({n_out/max(n_in,1):.0%})")

    print(f"\n  total: {total_in} -> {total_out}  ({total_out/max(total_in,1):.0%})")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
