"""Run feature extraction over the LHCO R&D file and save an .npz.

Memory on this box is tight (~7 GB), so workers read their own row range
from the HDF5 rather than receiving big arrays over IPC, and each range is
checkpointed to results/feat_chunks/ so a crashed run can resume.

  python extract.py --max-events 1100000 --chunk-size 5000 --workers 4
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from data.download import DEFAULT_PATH, n_rows, read_block
from data.jet_features import cluster_chunk, _features_for_chunk

CHUNK_DIR = os.path.join(os.path.dirname(__file__), "results", "feat_chunks")


def _process_range(args):
    """Read one [start, stop) row range, extract features, checkpoint, return path."""
    path, start, stop, chunk_dir = args
    out = os.path.join(chunk_dir, f"feat_{start:09d}.npz")
    if os.path.exists(out):
        return out
    pt2d, eta2d, phi2d, labels = read_block(path, start, stop)
    jets, consts, n = cluster_chunk(pt2d, eta2d, phi2d)
    feats = _features_for_chunk(jets, consts, n)
    valid = ~np.isnan(feats).any(axis=1)
    # Write atomically: a worker killed mid-write would otherwise leave a
    # truncated .npz at `out` that the exists() check above treats as done,
    # so it is never re-extracted and the merge fails on every rerun.
    # np.savez appends .npz to any name not already ending in .npz,
    # so the temp file must end in .npz or os.replace won't find it.
    tmp = out[:-4] + ".tmp.npz"
    np.savez(tmp, features=feats[valid], labels=labels[valid])
    os.replace(tmp, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join("results", "features.npz"))
    ap.add_argument("--chunk-dir", default=CHUNK_DIR)
    args = ap.parse_args()

    os.makedirs(args.chunk_dir, exist_ok=True)
    total = n_rows(args.path)
    if args.max_events is not None:
        total = min(total, args.max_events)

    ranges = [(args.path, s, min(s + args.chunk_size, total), args.chunk_dir)
              for s in range(0, total, args.chunk_size)]
    print(f"[extract] {total} events, {len(ranges)} chunks, "
          f"{args.workers} workers")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_range, r): r for r in ranges}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 10 == 0 or done == len(ranges):
                print(f"[extract] {done}/{len(ranges)} chunks done", flush=True)

    # Merge checkpoints in row order.
    feats, labs = [], []
    for _p, s, _stop, _cd in ranges:
        d = np.load(os.path.join(args.chunk_dir, f"feat_{s:09d}.npz"))
        feats.append(d["features"])
        labs.append(d["labels"])
    features = np.concatenate(feats, axis=0)
    labels = np.concatenate(labs, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, features=features, labels=labels)
    print(f"[extract] saved {features.shape} -> {args.out}  "
          f"(signal {int(labels.sum())}, bg {int((labels == 0).sum())})")


if __name__ == "__main__":
    main()
