"""Export sanitized metadata from approved audits; never read large payloads."""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from verify_data import FIELDS, path_error, validate_manifest


COUNTS = {
    "LiDARNet": 369,
    "T23D-CompBench_ply": 600,
    "co3d": 499,
    "modelnet40c_processed": 710,
    "scanobjectnn": 300,
    "sjtu_processed": 344,
    "wpc_processed": 261,
}
TOOL_SHA256 = {
    "merge_pointcloud_parts.py": "6fa96d619fcdc27302d5427b12265243bfc15ed4fe33c0ce529c3ab0b03bab34",
    "PointQ-Bench_SHA256SUMS_v1.0.txt": "018e1639a3efb01221fcd8e5abd2e0deff1ab81ce4a6dde879db881fe0bdda4f",
    "PointQ-Bench_PARTS_SHA256SUMS_v1.0.txt": "f50f936ab19c609cfcfc84d7064b276e7d33c76fff7ed62e88bcaa8f7582a80e",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_payload_metadata(record):
    require(type(record["bytes"]) is int and record["bytes"] > 0, "Archive/part bytes must be a positive integer")
    require(isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "Archive/part sha256 must be 64 lowercase hex digits")


def archive_record(report, expected_count, name):
    require(report["valid"] and report["payload_valid"] and report["data_only"], "Invalid archive audit")
    require(report["expected_records"] == report["payload_records"] == report["matched_sha256"] == expected_count, "Incomplete archive verification")
    require(not any(report[key] for key in ("errors", "duplicate_names", "unsafe_names", "other_files")), "Archive audit has errors")
    require(report["archive_prefix"] == "", "Release archive must have no wrapper directory")
    require(Path(report["path"].replace("\\", "/")).name == name, "Unexpected archive name")
    record = {"name": name, "bytes": report["bytes"], "sha256": report["archive_sha256"], "members": expected_count}
    validate_payload_metadata(record)
    return record


def prepare(audit, destination, download_tools, helper_destination=None):
    with (audit / "cloud_manifest_20261008.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    confirmation = read_json(audit / "cloud_confirmation_20261008.json")
    require(len(rows) == confirmation["catalog_rows"] == confirmation["local_clouds"] == 3083, "Expected 3083 cloud records")
    require(confirmation["dataset_counts"] == COUNTS, "Dataset counts differ")
    require(not any(confirmation[key] for key in ("missing", "extra", "casefold_collisions", "changed_hashes", "read_errors", "annotation_missing", "annotation_extra", "annotation_duplicates")), "Cloud confirmation has errors")
    require(confirmation["sha256_matches_baseline"] == 3083, "Incomplete baseline hashes")
    require(all(not table["index_path_mismatches"] and table["records"] == table["unique_paths"] == 3083 for table in confirmation["csv_checks"].values()), "Table identity mismatch")
    frozen = confirmation["records"]
    require(len(frozen) == len(rows), "Confirmation record count differs")
    fields = ("index", "relative_path", "bytes", "sha256")
    require([{key: str(row[key]) for key in fields} for row in rows] == [{key: str(row[key]) for key in fields} for row in frozen], "CSV and confirmation disagree")
    require([row["index"] for row in rows] == [f"{i:04d}" for i in range(1, 3084)], "Existing logical IDs changed")
    require(Counter(row["relative_path"].split("/")[0] for row in rows) == COUNTS, "Manifest counts differ")
    require(all(not path_error(row["relative_path"]) for row in rows), "Unsafe sample path")
    require(len({row["relative_path"].casefold() for row in rows}) == 3083, "Duplicate/case-colliding sample path")
    public = [{"index": row["index"], "pc_relative_path": row["relative_path"], "dataset": row["relative_path"].split("/")[0], "bytes": int(row["bytes"]), "sha256": row["sha256"]} for row in rows]
    require(sum(row["bytes"] for row in public) == confirmation["total_bytes"], "Cloud byte total differs")
    manifest_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(manifest_buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(public)
    manifest_text = manifest_buffer.getvalue()
    _, errors = validate_manifest(io.StringIO(manifest_text))
    require(not errors, "Public manifest failed validation: " + repr(errors))

    clouds = archive_record(read_json(audit / "baidu_upload_clean_clouds_readback_20261008.json"), 3083, "PointQ-Bench_pointclouds_v1.0.zip")
    images = archive_record(read_json(audit / "baidu_upload_clean_images_readback_20261008.json"), 18498, "PointQ-Bench_screenshots_mv6_v1.0.zip")
    split = read_json(audit / "baidu_upload_cloud_parts_20261008.json")
    require(split["valid"] and split["source_bytes"] == clouds["bytes"] and split["source_sha256"] == clouds["sha256"], "Split source differs")
    parts = [{key: part[key] for key in ("name", "bytes", "sha256")} for part in split["parts"]]
    require([part["name"] for part in parts] == [clouds["name"] + f".{i:03d}" for i in range(1, 5)], "Unexpected split names/order")
    for part in parts:
        validate_payload_metadata(part)
    require(sum(part["bytes"] for part in parts) == clouds["bytes"], "Part byte total differs")
    clouds["parts"] = parts
    clouds["assembly"] = "Concatenate .001, .002, .003, .004 in byte order, then unzip the merged ZIP."
    images["views_per_cloud"] = 6
    release = {
        "schema_version": 1,
        "release_version": "1.0",
        "audit_date": "2026-10-08",
        "cloud_manifest": "manifest.csv",
        "cloud_count": 3083,
        "cloud_uncompressed_bytes": confirmation["total_bytes"],
        "dataset_counts": COUNTS,
        "archives": {"clouds": clouds, "images": images},
        "public_download": {
            "provider": "Baidu Netdisk",
            "url": "https://pan.baidu.com/s/1oHoVxMzDVNkyiV40wphWzw?pwd=4vi1",
            "extraction_code": "4vi1",
            "folder": "PointQ-Bench-v1.0",
            "expected_file_count": 9,
            "expiry": "permanent",
            "share_page_expiry_verified": True,
            "status": "Public share supplied by release owner; permanent expiry and nine-file listing verified on share page. Full remote redownload not tested.",
        },
        "verification": {"local_cloud_sha256_matched": 3083, "local_clean_cloud_zip_crc_sha256_matched": 3083, "local_clean_image_zip_crc_sha256_matched": 18498, "remote_redownload_tested": False},
        "separate_assets": {"expert_annotations": "pending_formal_freeze", "question_tables": "pending_formal_freeze", "model_outputs": "pending_formal_freeze"},
        "rights": {"data": "Source-dependent terms; authors explicitly approved redistribution of all 3083 clouds and 18498 images and public Baidu sharing. No blanket data license is asserted.", "code_license": "MIT for self-owned code and site; not a blanket data license"},
    }
    release_text = json.dumps(release, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    # Cache verified helper bytes so later source changes cannot invalidate output.
    helper_payloads = {}
    for name, expected in TOOL_SHA256.items():
        payload = (download_tools / name).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected, "Download helper SHA-256 differs: " + name)
        helper_payloads[name] = payload
    whole_lines = helper_payloads["PointQ-Bench_SHA256SUMS_v1.0.txt"].decode("ascii").splitlines()
    part_lines = helper_payloads["PointQ-Bench_PARTS_SHA256SUMS_v1.0.txt"].decode("ascii").splitlines()
    require(whole_lines == [f"{entry['sha256']}  {entry['name']}" for entry in (clouds, images)], "Whole checksums disagree with audits")
    require(part_lines == [f"{part['sha256']}  {part['name']}" for part in parts], "Part checksums disagree with audits")
    helper_destination = helper_destination or Path(__file__).absolute().parent
    targets = [destination / "manifest.csv", destination / "release_manifest.json"] + [helper_destination / name for name in TOOL_SHA256]
    require(not any(path.exists() or path.is_symlink() for path in targets), "Refusing to overwrite generated files/helpers")
    for folder in (destination, helper_destination):
        for component in (folder,) + tuple(folder.parents):
            require(not (component.exists() or component.is_symlink()) or component.is_dir(), "Output directory conflicts with an existing file: " + str(component))
    # All schema, metadata, helper and collision checks precede public writes.
    destination.mkdir(parents=True, exist_ok=True)
    helper_destination.mkdir(parents=True, exist_ok=True)
    with targets[0].open("x", encoding="utf-8", newline="") as stream:
        stream.write(manifest_text)
    with targets[1].open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(release_text)
    for name, payload in helper_payloads.items():
        with (helper_destination / name).open("xb") as stream:
            stream.write(payload)
    print("Exported 3083 sanitized records and release metadata; copied three verified helpers. No payloads read or copied.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-tools", type=Path, required=True)
    parser.add_argument("--tools-output-dir", type=Path, help="Default: this script's directory; existing helpers are never replaced")
    args = parser.parse_args()
    prepare(args.audit_dir, args.output_dir, args.download_tools, args.tools_output_dir)


if __name__ == "__main__":
    main()
