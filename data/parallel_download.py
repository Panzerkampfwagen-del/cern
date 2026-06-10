"""Resumable parallel range downloader (stdlib only).

Zenodo single-stream throughput is slow and throttled per-connection, so we
fetch disjoint byte ranges concurrently and write them at their offsets with
os.pwrite. Each segment loops until it is fully written (a dropped connection
that returns a short/early EOF is resumed from the last byte, not assumed
complete), and per-thread success is checked after join. Always verify the
result with md5 (see expected checksum on the Zenodo record).

  python data/parallel_download.py --connections 16 [--start-offset N]
"""

import argparse
import os
import threading
import time
import urllib.request

URL = ("https://zenodo.org/api/records/6466204/files/"
       "events_anomalydetection_v2.h5/content")
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "raw",
                            "events_anomalydetection_v2.h5")


def total_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def _fetch_range(url, fd, start, end, progress, lock, results, idx, chunk=1 << 20):
    """Fetch [start, end] fully, resuming from the last written byte on any
    early EOF or error. Records True in results[idx] only if pos passes end."""
    pos = start
    for _attempt in range(1000):
        if pos > end:
            results[idx] = True
            return
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                while pos <= end:
                    buf = r.read(min(chunk, end - pos + 1))
                    if not buf:
                        break  # connection closed; outer loop resumes from pos
                    os.pwrite(fd, buf, pos)
                    pos += len(buf)
                    with lock:
                        progress[0] += len(buf)
        except Exception:
            time.sleep(5)
        if pos <= end:
            time.sleep(2)  # closed/failed early: retry the remainder
    results[idx] = pos > end


def download(path=DEFAULT_PATH, url=URL, connections=16, start_offset=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = total_size(url)
    # The final path is created only by an atomic rename after the size check
    # passes, so its presence at full size is a trustworthy "complete" signal.
    # A half-written download lives at <path>.part: parallel writes leave sparse
    # holes that getsize cannot detect, so its size is never trusted for resume.
    if os.path.exists(path) and os.path.getsize(path) == total:
        print(f"[pdl] already complete: {path}")
        return path

    part = path + ".part"
    # --start-offset lets the caller resume a known-contiguous prefix of .part;
    # without it we re-fetch the whole range (holes are undetectable).
    have = start_offset if start_offset is not None else 0
    print(f"[pdl] total {total/1e9:.3f} GB, fetching from {have/1e9:.3f} GB, "
          f"{connections} connections")

    with open(part, "r+b" if os.path.exists(part) else "w+b") as f:
        f.truncate(total)
    fd = os.open(part, os.O_WRONLY)
    try:
        remaining = total - have
        seg = remaining // connections
        bounds = []
        s = have
        for i in range(connections):
            e = total - 1 if i == connections - 1 else s + seg - 1
            bounds.append((s, e))
            s = e + 1

        progress = [have]
        lock = threading.Lock()
        results = [False] * connections
        threads = [threading.Thread(target=_fetch_range,
                   args=(url, fd, a, b, progress, lock, results, i))
                   for i, (a, b) in enumerate(bounds)]
        for t in threads:
            t.start()
        last, last_t = progress[0], time.time()
        while any(t.is_alive() for t in threads):
            time.sleep(15)
            now = time.time()
            cur = progress[0]
            rate = (cur - last) / max(now - last_t, 1e-6)
            eta = (total - cur) / max(rate, 1) / 60
            print(f"[pdl] {cur/1e9:.3f}/{total/1e9:.3f} GB  "
                  f"{rate/1e6:.2f} MB/s  ETA {eta:.0f} min", flush=True)
            last, last_t = cur, now
        for t in threads:
            t.join()
    finally:
        os.close(fd)

    if not all(results):
        raise RuntimeError(f"segments failed: {results}")
    final = os.path.getsize(part)
    if final != total:
        raise RuntimeError(f"size mismatch: {final} != {total}")
    os.replace(part, path)   # atomic: final path appears only when fully written
    print(f"[pdl] all {connections} segments complete: {final} bytes -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--connections", type=int, default=16)
    ap.add_argument("--start-offset", type=int, default=None)
    args = ap.parse_args()
    download(args.path, connections=args.connections,
             start_offset=args.start_offset)


if __name__ == "__main__":
    main()
