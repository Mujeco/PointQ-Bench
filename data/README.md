# PointQ-Bench Data, v1.0

The cloud package contains **3,083 point clouds** with these exact directory names:

| Dataset directory | Clouds |
| --- | ---: |
| LiDARNet | 369 |
| T23D-CompBench_ply | 600 |
| co3d | 499 |
| modelnet40c_processed | 710 |
| scanobjectnn | 300 |
| sjtu_processed | 344 |
| wpc_processed | 261 |
| **Total** | **3,083** |

The separate six-view package contains **18,498 PNGs**, six per cloud. For
`dataset/category/sample.ply`, the images are
`dataset/category/sample/1.png` through `6.png`, under a separate image root.
Expert annotations, question tables, and model outputs are separate assets,
not included in these archives; their formal release freeze is still pending.

## Manifest and Download

`manifest.csv` preserves the existing zero-padded `index` IDs (`0001`-`3083`)
and exact `pc_relative_path` values used by benchmark records. The latter are
the audit's `relative_path`, without any local root or wrapper directory.
`dataset` is the first path component, `bytes` is uncompressed file size, and
`sha256` covers the original cloud bytes. Do not renumber or flatten paths.
`release_manifest.json` records complete archive sizes/hashes and ordered parts.

**Public download:** [Baidu Netdisk, PointQ-Bench-v1.0](https://pan.baidu.com/s/1oHoVxMzDVNkyiV40wphWzw?pwd=4vi1).
Extraction code: **`4vi1`**. The share page's permanent expiry and nine-file
folder listing were verified; a complete remote redownload has **not** been
verified. The folder contains four cloud byte parts, one screenshots ZIP,
two checksum files, the merge helper, and a download guide.

The cloud ZIP is split into **four raw byte parts**, not four independent ZIPs.
Download `.zip.001` through `.zip.004` into a new download folder. Put
`scripts/data/merge_pointcloud_parts.py` and
`scripts/data/PointQ-Bench_PARTS_SHA256SUMS_v1.0.txt` in that same folder, then run:

```sh
python merge_pointcloud_parts.py
# Or check all parts without creating the merged ZIP:
python merge_pointcloud_parts.py --verify-only
```

The unchanged helper concatenates `.001`, `.002`, `.003`, `.004` in order and
checks each part plus the complete ZIP size/SHA-256. It refuses an existing
output. After a failed merge, it can leave a `.merge.part` file; retry in a
fresh folder. **Only after concatenation and verification, unzip the merged
cloud ZIP.** Unzip the screenshots ZIP separately. Extract outside this repo;
do not add clouds, PNGs, ZIPs, or downloaded parts to Git.

`scripts/data/PointQ-Bench_SHA256SUMS_v1.0.txt` covers the two complete ZIPs.
On systems providing `sha256sum`, run `sha256sum -c` with that file in the
download folder containing both complete ZIPs. The merge helper needs Python
3.8+; the verifier below needs Python 3.8+ and no third-party packages.

## Verify Extracted Clouds

From the repository root, using your extracted root (which directly contains
the seven dataset directories):

```sh
python scripts/verify_data.py --cloud-root /path/to/clouds
python scripts/verify_data.py --zip /path/to/PointQ-Bench_pointclouds_v1.0.zip
python scripts/verify_data.py --cloud-root /path/to/clouds --zip /path/to/PointQ-Bench_pointclouds_v1.0.zip
# Lightweight checks only, NOT content-hash verification:
python scripts/verify_data.py --cloud-root /path/to/clouds --inventory-only
python scripts/verify_data.py --zip /path/to/PointQ-Bench_pointclouds_v1.0.zip --zip-members-only
python -m unittest discover -s scripts/tests -v
```

Default verification checks every cloud size/SHA-256 and exact membership.
ZIP mode also reads each cloud with ZIP CRC checking. Missing/extra files,
unexpected directories, unsafe paths, duplicate/case-colliding names,
symlinks/reparse points, special files, and read errors fail with exit status 1
and a JSON error report. Usage errors exit 2. No extraction or source writes
are performed. `--manifest` accepts a fixture or alternative manifest with
the same schema; exactness is relative to that manifest. This CLI verifies
clouds, not PNG content or complete ZIP-container hashes.

Maintainers can reproduce the metadata with
`scripts/data/prepare_release_metadata.py --audit-dir AUDITS --output-dir NEW_DATA --download-tools VERIFIED_HELPERS --tools-output-dir NEW_HELPERS`.
It validates the audit agreement and pinned small helpers, exports only public
fields, refuses existing outputs, and never reads or copies cloud/PNG payloads.

## Verification and Rights

The 2026-10-08 local audit matched SHA-256 for all 3,083 clouds. Both clean
release archives were fully read locally with CRC/SHA-256 checks: 3,083 clouds
and 18,498 images. Part and whole hashes in this release come from those
audits. **Remote download-and-reverify has not been tested.** This preparation
does not upload data, create the public share link, or rerun the large audits;
the release owner supplied the verified permanent share link above.

The authors explicitly approved redistribution of all clouds and images,
including public Baidu sharing. Data terms remain source-dependent; this is
not a new blanket data license. Self-owned code and site are MIT licensed
(see the repository `LICENSE`); MIT does not relicense the source datasets.
