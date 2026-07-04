"""Fetch and inspect the LHC Olympics 2020 R&D dataset.

The R&D file (events_anomalydetection_v2.h5, ~2.9 GB) is an old pandas (0.15.2)
fixed-format frame: 1.1M rows x 2101 columns stored blosc-compressed under
/df/block0_values. Columns 0..2099 are 700 particles x (pT, eta, phi)
interleaved; the final column is the truth label (0 = QCD background,
1 = W'->WZ signal; ~9.09% signal).

We read it directly with PyTables: modern pandas.read_hdf cannot parse this
file's byte-string metadata, and h5py lacks the blosc filter plugin. PyTables
registers blosc itself, so it reads the array node cleanly.
"""

import os
import urllib.request

import numpy as np
import tables

from .parallel_download import total_size

ZENODO_URL = (
    "https://zenodo.org/api/records/6466204/files/"
    "events_anomalydetection_v2.h5/content"
)
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "raw",
                            "events_anomalydetection_v2.h5")
N_PARTICLES = 700


def _expected_size(url):
    """Exact Content-Length from a HEAD request, or None if it cannot be read."""
    try:
        return total_size(url)
    except Exception:
        return None


def download(path=DEFAULT_PATH, url=ZENODO_URL):
    """Download the R&D HDF5 if it is not already present."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Compare against the exact remote size so a truncated file is re-fetched;
    # if HEAD fails, fall back to the coarse >2.9 GB floor.
    if os.path.exists(path):
        expected = _expected_size(url)
        have = os.path.getsize(path)
        complete = have == expected if expected is not None else have > 2_900_000_000
        if complete:
            print(f"[download] already present: {path} ({have / 1e9:.2f} GB)")
            return path
    print(f"[download] fetching {url}\n           -> {path}")
    urllib.request.urlretrieve(url, path)
    print(f"[download] done ({os.path.getsize(path) / 1e9:.2f} GB)")
    return path


def values_node(h5):
    """The block0_values CArray holding the full (rows, 2101) data matrix."""
    for node in h5.walk_nodes("/", "Array"):
        if node.name == "block0_values":
            return node
    raise RuntimeError("no block0_values array found in HDF5")


def n_rows(path=DEFAULT_PATH):
    with tables.open_file(path, "r") as h5:
        return int(values_node(h5).shape[0])


def _split_row_block(arr):
    """(n, 2101) block -> (pt2d, eta2d, phi2d, labels)."""
    labels = arr[:, -1].astype(np.int64)
    feat = arr[:, :-1].reshape(arr.shape[0], N_PARTICLES, 3)
    return (feat[:, :, 0].astype(np.float64),
            feat[:, :, 1].astype(np.float64),
            feat[:, :, 2].astype(np.float64),
            labels)


def read_block(path, start, stop):
    """Read row range [start, stop) and split into particle arrays + labels."""
    with tables.open_file(path, "r") as h5:
        arr = values_node(h5)[start:stop]
    return _split_row_block(arr)


def read_chunks(path=DEFAULT_PATH, chunk_size=10_000, max_events=None):
    """Yield (pt2d, eta2d, phi2d, labels) chunks of padded particle arrays."""
    with tables.open_file(path, "r") as h5:
        node = values_node(h5)
        total = node.shape[0]
        if max_events is not None:
            total = min(total, max_events)
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            yield _split_row_block(node[start:stop])


def inspect(path=DEFAULT_PATH):
    """Print structure and a few raw particle 4-vectors (build-order step 1)."""
    with tables.open_file(path, "r") as h5:
        node = values_node(h5)
        total, ncols = int(node.shape[0]), int(node.shape[1])
        print(f"[inspect] rows={total}  columns={ncols}  (expected 2101)")
        head = node[:3]
        labels = node[:, -1]
    print(f"[inspect] first labels: {head[:, -1]}")
    ev0 = head[0, :-1].reshape(N_PARTICLES, 3)
    n_real = int((ev0[:, 0] > 0).sum())
    print(f"[inspect] event 0: {n_real} particles with pT>0")
    print("[inspect] first 5 particles (pT, eta, phi):")
    for p in ev0[:5]:
        print(f"          {p[0]:8.3f}  {p[1]:7.3f}  {p[2]:7.3f}")
    n_sig = int(labels.sum())
    print(f"[inspect] total signal events: {n_sig} / {total} "
          f"({100 * n_sig / total:.2f}%)")


if __name__ == "__main__":
    p = download()
    inspect(p)
