"""Verify exact PointQ-Bench cloud membership and SHA-256 using only the stdlib."""

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path


FIELDS = ("index", "pc_relative_path", "dataset", "bytes", "sha256")
DEFAULT_MANIFEST = Path(__file__).absolute().parents[1] / "data" / "manifest.csv"
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    prefix + str(number) for prefix in ("COM", "LPT") for number in range(1, 10)
}


@dataclass(frozen=True)
class Record:
    index: str
    pc_relative_path: str
    dataset: str
    bytes: int
    sha256: str


def path_error(name):
    """Check the raw spelling, before Path can normalize dangerous components."""
    if not name or any(ord(char) < 32 or ord(char) == 127 for char in name):
        return "empty path or control character"
    if any(char in name for char in '\\<>:"|?*'):
        return "backslash, drive/stream name, or nonportable character"
    for component in name.split("/"):
        if component in ("", ".", ".."):
            return "absolute, empty, dot, or traversal component"
        if component != component.strip() or component.endswith("."):
            return "leading/trailing whitespace or trailing dot"
        if component.split(".")[0].upper() in RESERVED:
            return "reserved device name"
    return None


def validate_manifest(stream):
    errors = []
    records = {}
    indices = set()
    folded = {}
    try:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            return {}, ["Manifest header must be " + ",".join(FIELDS)]
        for line, row in enumerate(reader, 2):
            label = "Manifest line " + str(line)
            if None in row or any(value is None for value in row.values()):
                errors.append(label + ": wrong number of fields")
                continue
            name = row["pc_relative_path"]
            problem = path_error(name)
            if problem:
                errors.append(label + ": unsafe path " + repr(name) + ": " + problem)
            index = row["index"]
            if not re.fullmatch(r"[0-9]{4}", index) or index == "0000":
                errors.append(label + ": index must be a positive four-digit ID")
            if index in indices:
                errors.append(label + ": duplicate index " + index)
            indices.add(index)
            if name.casefold() in folded:
                errors.append(label + ": duplicate/case-colliding path " + repr(name))
            folded[name.casefold()] = name
            if "/" not in name or row["dataset"] != name.split("/")[0]:
                errors.append(label + ": dataset must match the first path component")
            if not re.fullmatch(r"[0-9]+", row["bytes"]):
                errors.append(label + ": bytes must be a nonnegative decimal integer")
                continue
            if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                errors.append(label + ": sha256 must be 64 lowercase hex digits")
            records[name] = Record(index, name, row["dataset"], int(row["bytes"]), row["sha256"])
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append("Cannot read manifest: " + str(exc))
    if not records:
        errors.append("Manifest must contain at least one record")
    return records, errors


def load_manifest(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return validate_manifest(stream)
    except (OSError, UnicodeError, csv.Error) as exc:
        return {}, ["Cannot read manifest: " + str(exc)]


def is_link(info):
    # Junctions/reparse points on Windows must not bypass the no-symlink policy.
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def fingerprint(info):
    common = info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
    # Windows path stat and handle stat do not agree on ctime semantics.
    return common if os.name == "nt" else common + (info.st_ctime_ns,)


def check_root(path):
    path = Path(path).absolute()
    for component in reversed((path,) + tuple(path.parents)):
        if is_link(component.lstat()):
            raise ValueError("Symlink/reparse point in input path: " + str(component))
    return path


def digest_stream(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def allowed_directories(records):
    result = set()
    for name in records:
        components = name.split("/")
        result.update("/".join(components[:i]) for i in range(1, len(components)))
    return result


def new_report(mode):
    return {"mode": mode, "files_seen": 0, "matched_sizes": 0, "matched_sha256": 0, "errors": []}


def verify_tree(root, records, inventory_only=False):
    report = new_report("inventory_and_sizes_only" if inventory_only else "content_sha256")
    errors = report["errors"]
    seen = set()
    folded = {}
    directories = allowed_directories(records)
    try:
        root = check_root(root)
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise ValueError("Cloud root is not a directory")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        report["valid"] = False
        return report
    pending = [root]
    while pending:
        folder = pending.pop()
        try:
            # Recheck each directory so a substituted link is not followed.
            info = folder.lstat()
            if is_link(info) or not stat.S_ISDIR(info.st_mode):
                raise ValueError("Directory became a symlink or non-directory: " + str(folder))
            with os.scandir(folder) as entries:
                entries = sorted(entries, key=lambda entry: entry.name)
        except (OSError, ValueError) as exc:
            errors.append("Cannot scan directory: " + str(exc))
            continue
        for entry in entries:
            path = Path(entry.path)
            name = path.relative_to(root).as_posix()
            problem = path_error(name)
            if problem:
                errors.append("Unsafe path " + repr(name) + ": " + problem)
                continue
            if name.casefold() in folded:
                errors.append("Duplicate/case-colliding path: " + name)
            folded[name.casefold()] = name
            try:
                # Windows DirEntry.stat can omit inode/device identity.
                before = path.lstat()
                if is_link(before):
                    errors.append("Symlink/reparse point: " + name)
                    continue
                if stat.S_ISDIR(before.st_mode):
                    if name not in directories:
                        errors.append("Extra directory: " + name)
                    pending.append(path)
                    continue
                if not stat.S_ISREG(before.st_mode):
                    errors.append("Non-regular file: " + name)
                    continue
                seen.add(name)
                report["files_seen"] += 1
                if name not in records:
                    errors.append("Extra file: " + name)
                    continue
                expected = records[name]
                if before.st_size != expected.bytes:
                    errors.append("Size mismatch: " + name)
                else:
                    report["matched_sizes"] += 1
                if inventory_only:
                    continue
                flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                with os.fdopen(os.open(path, flags), "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if not stat.S_ISREG(opened.st_mode) or fingerprint(before) != fingerprint(opened):
                        raise ValueError("File changed before reading")
                    size, digest = digest_stream(stream)
                    after_read = os.fstat(stream.fileno())
                after = path.lstat()
                if is_link(after) or fingerprint(before) != fingerprint(after) or fingerprint(opened) != fingerprint(after_read):
                    raise ValueError("File changed during reading")
                if size != expected.bytes or digest != expected.sha256:
                    errors.append("Content SHA-256/size mismatch: " + name)
                else:
                    report["matched_sha256"] += 1
            except (OSError, ValueError) as exc:
                errors.append("Cannot verify " + name + ": " + str(exc))
    errors.extend("Missing file: " + name for name in sorted(set(records) - seen))
    report["valid"] = not errors
    return report


def verify_zip(path, records, members_only=False):
    report = new_report("zip_membership_and_sizes_only" if members_only else "zip_crc_and_sha256")
    errors = report["errors"]
    seen = set()
    names = set()
    folded = {}
    directories = allowed_directories(records)
    try:
        path = check_root(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("ZIP must be a regular file")
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                raw = member.orig_filename
                name = raw[:-1] if raw.endswith("/") else raw
                problem = path_error(name)
                if problem or raw != member.filename:
                    errors.append("Unsafe ZIP path: " + repr(raw))
                    continue
                if name in names or name.casefold() in folded:
                    errors.append("Duplicate/case-colliding ZIP path: " + name)
                names.add(name)
                folded[name.casefold()] = name
                mode = (member.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if kind == stat.S_IFLNK:
                    errors.append("Symlink ZIP member: " + name)
                    continue
                if member.is_dir():
                    if kind not in (0, stat.S_IFDIR) or name not in directories:
                        errors.append("Extra or non-directory ZIP entry: " + name)
                    continue
                if kind not in (0, stat.S_IFREG):
                    errors.append("Non-regular ZIP member: " + name)
                    continue
                seen.add(name)
                report["files_seen"] += 1
                if name not in records:
                    errors.append("Extra ZIP file: " + name)
                    continue
                expected = records[name]
                if member.file_size != expected.bytes:
                    errors.append("ZIP size mismatch: " + name)
                else:
                    report["matched_sizes"] += 1
                if members_only:
                    continue
                try:
                    with archive.open(member) as stream:
                        size, digest = digest_stream(stream)
                    if size != expected.bytes or digest != expected.sha256:
                        errors.append("ZIP content SHA-256/size mismatch: " + name)
                    else:
                        report["matched_sha256"] += 1
                except (OSError, RuntimeError, ValueError, EOFError, NotImplementedError, zipfile.BadZipFile, zlib.error) as exc:
                    errors.append("Cannot read ZIP member " + name + ": " + str(exc))
        if fingerprint(before) != fingerprint(path.lstat()) or is_link(path.lstat()):
            errors.append("ZIP changed during verification")
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        errors.append("Cannot verify ZIP: " + str(exc))
    errors.extend("Missing ZIP file: " + name for name in sorted(set(records) - seen))
    report["valid"] = not errors
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cloud-root", type=Path, help="Extracted root containing the seven dataset directories")
    parser.add_argument("--zip", type=Path, help="Optional merged cloud ZIP (never extracted by this tool)")
    parser.add_argument("--inventory-only", action="store_true", help="Tree membership/sizes only; does NOT verify SHA-256")
    parser.add_argument("--zip-members-only", action="store_true", help="ZIP membership/sizes only; does NOT read CRC/SHA-256")
    args = parser.parse_args(argv)
    if not args.cloud_root and not args.zip:
        parser.error("provide --cloud-root and/or --zip")
    if args.inventory_only and not args.cloud_root:
        parser.error("--inventory-only requires --cloud-root")
    if args.zip_members_only and not args.zip:
        parser.error("--zip-members-only requires --zip")
    records, errors = load_manifest(args.manifest)
    result = {"expected_records": len(records), "manifest_errors": errors}
    if not errors:
        if args.cloud_root:
            result["clouds"] = verify_tree(args.cloud_root, records, args.inventory_only)
        if args.zip:
            result["zip"] = verify_zip(args.zip, records, args.zip_members_only)
    result["valid"] = not errors and all(
        value["valid"] for key, value in result.items() if key in ("clouds", "zip")
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
