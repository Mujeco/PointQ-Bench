"""Small synthetic fixtures only; never access original cloud/image payloads."""

import contextlib
import csv
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(SCRIPTS))
import verify_data as verifier
from data import merge_pointcloud_parts as merger
from data.prepare_release_metadata import COUNTS, TOOL_SHA256
from data import prepare_release_metadata as metadata


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "clouds"
        self.root.mkdir()
        self.payloads = {"dataset/class/a.ply": b"fixture cloud A\n", "dataset/class/b.ply": b"fixture cloud B\n"}
        self.rows = [
            {"index": f"{i:04d}", "pc_relative_path": name, "dataset": "dataset", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for i, (name, payload) in enumerate(self.payloads.items(), 1)
        ]
        self.manifest = self.base / "manifest.csv"
        self.write_manifest(self.rows)
        self.records, errors = verifier.load_manifest(self.manifest)
        self.assertEqual(errors, [])
        for name, payload in self.payloads.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        self.archive = self.base / "clouds.zip"

    def write_manifest(self, rows, fields=verifier.FIELDS):
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_zip(self, extras=(), payloads=None, directories=()):
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as archive:
            for directory in directories:
                archive.writestr(directory, b"")
            for name, payload in (self.payloads if payloads is None else payloads).items():
                archive.writestr(name, payload)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for name, payload in extras:
                    archive.writestr(name, payload)

    def assert_error(self, report, fragment):
        self.assertFalse(report["valid"], report)
        self.assertTrue(any(fragment in error for error in report["errors"]), report)

    def test_tree_hashes_and_read_only_behavior(self):
        before = {name: (self.root / name).read_bytes() for name in self.payloads}
        result = verifier.verify_tree(self.root, self.records)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["matched_sha256"], 2)
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in self.payloads})

    def test_tree_collects_missing_extra_and_hash_errors(self):
        (self.root / "dataset/class/a.ply").unlink()
        (self.root / "dataset/class/b.ply").write_bytes(b"fixture cloud X\n")
        (self.root / "notes.txt").write_bytes(b"not a cloud")
        result = verifier.verify_tree(self.root, self.records)
        for fragment in ("Missing file", "Extra file", "Content SHA-256"):
            self.assert_error(result, fragment)

    def test_tree_size_mismatch(self):
        (self.root / "dataset/class/a.ply").write_bytes(b"short")
        self.assert_error(verifier.verify_tree(self.root, self.records), "Size mismatch")

    def test_tree_inventory_only_does_not_claim_hashes(self):
        (self.root / "dataset/class/a.ply").write_bytes(b"fixture cloud X\n")
        with mock.patch.object(verifier, "digest_stream", side_effect=AssertionError("must not read payload")):
            result = verifier.verify_tree(self.root, self.records, inventory_only=True)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["matched_sha256"], 0)
        self.assertEqual(result["mode"], "inventory_and_sizes_only")

    def test_unexpected_empty_directory_is_not_ignored(self):
        (self.root / "unexpected").mkdir()
        self.assert_error(verifier.verify_tree(self.root, self.records), "Extra directory")

    def test_missing_root(self):
        self.assertFalse(verifier.verify_tree(self.base / "absent", self.records)["valid"])

    def test_scan_error_is_not_silently_dropped(self):
        with mock.patch.object(verifier.os, "scandir", side_effect=PermissionError("fixture denied")):
            result = verifier.verify_tree(self.root, self.records)
        self.assert_error(result, "Cannot scan directory")

    def test_content_read_error_is_reported(self):
        with mock.patch.object(verifier, "digest_stream", side_effect=OSError("fixture read failed")):
            result = verifier.verify_tree(self.root, self.records)
        self.assert_error(result, "fixture read failed")

    def test_file_change_during_read_is_reported(self):
        real_digest = verifier.digest_stream

        def change_after_read(stream):
            result = real_digest(stream)
            target = self.root / "dataset/class/a.ply"
            info = target.stat()
            os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
            return result

        with mock.patch.object(verifier, "digest_stream", side_effect=change_after_read):
            result = verifier.verify_tree(self.root, self.records)
        self.assert_error(result, "changed during reading")

    def test_link_detection_including_windows_reparse_points(self):
        self.assertTrue(verifier.is_link(type("Info", (), {"st_mode": stat.S_IFLNK, "st_file_attributes": 0})()))
        with mock.patch.object(verifier.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024, create=True):
            self.assertTrue(verifier.is_link(type("Info", (), {"st_mode": stat.S_IFDIR, "st_file_attributes": 1024})()))

    def test_simulated_linked_directory_is_not_traversed(self):
        directory = self.root / "dataset"
        inode = directory.lstat().st_ino
        real_is_link = verifier.is_link
        with mock.patch.object(verifier, "is_link", side_effect=lambda info: info.st_ino == inode or real_is_link(info)):
            result = verifier.verify_tree(self.root, self.records)
        self.assert_error(result, "Symlink/reparse point")
        self.assertEqual(result["files_seen"], 0)

    def test_simulated_linked_root_is_rejected(self):
        inode = self.root.lstat().st_ino
        real_is_link = verifier.is_link
        with mock.patch.object(verifier, "is_link", side_effect=lambda info: info.st_ino == inode or real_is_link(info)):
            self.assert_error(verifier.verify_tree(self.root, self.records), "Symlink/reparse point")

    def test_real_file_and_directory_symlinks_rejected(self):
        target = self.root / "dataset/class/a.ply"
        link = self.root / "link.ply"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("OS does not permit creation of symlinks: " + str(exc))
        self.assert_error(verifier.verify_tree(self.root, self.records), "Symlink/reparse point")
        linked_root = self.base / "linked-root"
        linked_root.symlink_to(self.root, target_is_directory=True)
        self.assert_error(verifier.verify_tree(linked_root, self.records), "Symlink/reparse point")

    def test_zip_content_and_valid_explicit_directories(self):
        self.write_zip(directories=("dataset/", "dataset/class/"))
        result = verifier.verify_zip(self.archive, self.records)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["matched_sha256"], 2)

    def test_zip_missing_extra_and_hash_mismatch(self):
        self.write_zip(payloads={"dataset/class/b.ply": b"fixture cloud X\n"}, extras=(("notes.txt", b"extra"),))
        result = verifier.verify_zip(self.archive, self.records)
        for fragment in ("Missing ZIP file", "Extra ZIP file", "ZIP content SHA-256"):
            self.assert_error(result, fragment)

    def test_zip_sizes_in_members_only_mode(self):
        self.write_zip(payloads={"dataset/class/a.ply": b"short", "dataset/class/b.ply": self.payloads["dataset/class/b.ply"]})
        self.assert_error(verifier.verify_zip(self.archive, self.records, members_only=True), "ZIP size mismatch")

    def test_zip_members_only_does_not_claim_content_checks(self):
        self.write_zip()
        with mock.patch.object(verifier.zipfile.ZipFile, "open", side_effect=AssertionError("must not read payload")):
            result = verifier.verify_zip(self.archive, self.records, members_only=True)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["matched_sha256"], 0)
        self.assertEqual(result["mode"], "zip_membership_and_sizes_only")

    def test_zip_duplicate_and_case_colliding_paths(self):
        for name in ("dataset/class/a.ply", "DATASET/class/a.ply", "dataset/"):
            with self.subTest(name=name):
                directories = ("dataset/",) if name == "dataset/" else ()
                self.write_zip(extras=((name, b"duplicate"),), directories=directories)
                self.assert_error(verifier.verify_zip(self.archive, self.records), "Duplicate/case-colliding")

    def test_zip_unsafe_paths(self):
        for name in ("../escape.ply", "/absolute.ply", "C:/drive.ply", "dataset\\backslash.ply", "dataset//a.ply", "dataset/./a.ply", "dataset/a.ply:stream", "dataset/NUL.ply", "dataset/trailing."):
            with self.subTest(name=name):
                self.write_zip(extras=((name, b"bad"),))
                if "\\" in name:
                    # Windows zipfile writers normalize backslashes; patch the
                    # equal-length raw ZIP names to exercise unsafe input.
                    raw = self.archive.read_bytes()
                    self.archive.write_bytes(raw.replace(name.replace("\\", "/").encode("ascii"), name.encode("ascii")))
                self.assert_error(verifier.verify_zip(self.archive, self.records), "Unsafe ZIP path")

    def test_zip_symlink_and_special_members(self):
        for mode, expected in ((stat.S_IFLNK, "Symlink ZIP member"), (stat.S_IFIFO, "Non-regular ZIP member")):
            with self.subTest(mode=mode):
                self.write_zip()
                info = zipfile.ZipInfo("dataset/class/link.ply")
                info.create_system = 3
                info.external_attr = (mode | 0o777) << 16
                with zipfile.ZipFile(self.archive, "a") as archive:
                    archive.writestr(info, b"a.ply")
                self.assert_error(verifier.verify_zip(self.archive, self.records), expected)

    def test_zip_extra_empty_directory(self):
        self.write_zip(directories=("unused/",))
        self.assert_error(verifier.verify_zip(self.archive, self.records), "Extra or non-directory")

    def test_zip_corruption_crc_failure(self):
        self.write_zip()
        raw = self.archive.read_bytes()
        raw = raw.replace(b"fixture cloud A\n", b"fixture cloud X\n", 1)
        self.archive.write_bytes(raw)
        self.assert_error(verifier.verify_zip(self.archive, self.records), "Cannot read ZIP member")

    def test_invalid_zip_container(self):
        self.archive.write_bytes(b"not a zip")
        self.assert_error(verifier.verify_zip(self.archive, self.records), "Cannot verify ZIP")

    def test_zip_read_errors_remain_visible(self):
        self.write_zip()
        with mock.patch.object(verifier, "digest_stream", side_effect=EOFError("fixture truncated stream")):
            result = verifier.verify_zip(self.archive, self.records)
        self.assert_error(result, "fixture truncated stream")

    def test_manifest_rejects_duplicates_and_bad_fields(self):
        for field, value, fragment in (
            ("index", "0002", "duplicate index"),
            ("pc_relative_path", "dataset/class/b.ply", "duplicate/case-colliding"),
            ("pc_relative_path", "DATASET/class/b.ply", "duplicate/case-colliding"),
            ("pc_relative_path", "../a.ply", "unsafe path"),
            ("dataset", "wrong", "dataset must match"),
            ("bytes", "-1", "bytes must be"),
            ("sha256", "not-a-hash", "sha256 must be"),
            ("index", "1", "four-digit"),
        ):
            with self.subTest(field=field, value=value):
                rows = [dict(row) for row in self.rows]
                rows[0][field] = value
                self.write_manifest(rows)
                _, errors = verifier.load_manifest(self.manifest)
                self.assertTrue(any(fragment in error for error in errors), errors)

    def test_manifest_schema_empty_and_malformed_rows(self):
        for raw in ("wrong,header\n", ",".join(verifier.FIELDS) + "\n", ",".join(verifier.FIELDS) + "\n0001,dataset/a.ply\n", ",".join(verifier.FIELDS) + "\n0001,dataset/a.ply,dataset,3,hash,unexpected\n"):
            with self.subTest(raw=raw):
                self.manifest.write_text(raw, encoding="utf-8")
                self.assertTrue(verifier.load_manifest(self.manifest)[1])

    def test_manifest_io_error(self):
        self.assertTrue(verifier.load_manifest(self.base / "absent.csv")[1])

    def test_cli_success_and_failure_exit_codes(self):
        command = [sys.executable, "-B", str(SCRIPTS / "verify_data.py"), "--manifest", str(self.manifest), "--cloud-root", str(self.root)]
        success = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(json.loads(success.stdout)["clouds"]["matched_sha256"], 2)
        (self.root / "extra.txt").write_bytes(b"extra")
        failure = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(failure.returncode, 1, failure.stderr)
        self.assertFalse(json.loads(failure.stdout)["valid"])

    def test_cli_malformed_manifest_fails_before_touching_data(self):
        self.manifest.write_text("bad header\n", encoding="utf-8")
        with mock.patch.object(verifier, "verify_tree", side_effect=AssertionError("must not scan")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = verifier.main(["--manifest", str(self.manifest), "--cloud-root", str(self.root)])
        self.assertEqual(status, 1)
        self.assertTrue(json.loads(output.getvalue())["manifest_errors"])

    def test_cli_usage_errors(self):
        for args in ([], ["--zip", "missing.zip", "--inventory-only"], ["--cloud-root", "missing", "--zip-members-only"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    verifier.main(args)
                self.assertEqual(raised.exception.code, 2)


class MergeHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parts = [b"small ", b"synthetic ", b"archive ", b"fixture"]
        self.whole = b"".join(self.parts)
        lines = []
        for i, payload in enumerate(self.parts, 1):
            name = merger.NAME + f".{i:03d}"
            (self.root / name).write_bytes(payload)
            lines.append(hashlib.sha256(payload).hexdigest() + "  " + name)
        (self.root / "PointQ-Bench_PARTS_SHA256SUMS_v1.0.txt").write_text("\n".join(lines) + "\n", encoding="ascii")

    def run_helper(self, *arguments):
        with mock.patch.object(merger, "__file__", str(self.root / "merge_pointcloud_parts.py")), mock.patch.object(merger, "BYTES", len(self.whole)), mock.patch.object(merger, "SHA256", hashlib.sha256(self.whole).hexdigest()), mock.patch.object(sys, "argv", ["merge_pointcloud_parts.py", *arguments]), contextlib.redirect_stdout(io.StringIO()):
            merger.main()

    def test_byte_concatenation_and_unchanged_inputs(self):
        self.run_helper()
        self.assertEqual((self.root / merger.NAME).read_bytes(), self.whole)
        for i, payload in enumerate(self.parts, 1):
            self.assertEqual((self.root / (merger.NAME + f".{i:03d}")).read_bytes(), payload)

    def test_verify_only_does_not_write_output(self):
        self.run_helper("--verify-only")
        self.assertFalse((self.root / merger.NAME).exists())
        self.assertFalse((self.root / (merger.NAME + ".merge.part")).exists())

    def test_existing_output_is_preserved(self):
        (self.root / merger.NAME).write_bytes(b"do not replace")
        with self.assertRaises(FileExistsError):
            self.run_helper()
        self.assertEqual((self.root / merger.NAME).read_bytes(), b"do not replace")

    def test_missing_or_tampered_part_fails(self):
        part = self.root / (merger.NAME + ".001")
        part.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Part SHA-256 mismatch"):
            self.run_helper("--verify-only")
        part.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_helper("--verify-only")


class PublicMetadataTests(unittest.TestCase):
    def test_manifest_exact_counts_and_no_private_fields(self):
        manifest = SCRIPTS.parent / "data" / "manifest.csv"
        records, errors = verifier.load_manifest(manifest)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 3083)
        self.assertEqual([record.index for record in records.values()], [f"{i:04d}" for i in range(1, 3084)])
        self.assertEqual({dataset: sum(record.dataset == dataset for record in records.values()) for dataset in COUNTS}, COUNTS)
        self.assertEqual(sum(record.bytes for record in records.values()), 22957849147)
        text = manifest.read_text(encoding="utf-8")
        for private in ("local_path", "mtime_ns", "matches_20260926", "E:", "/data/home/", "C:"):
            self.assertNotIn(private, text)

    def test_release_metadata_and_copied_checksum_text_agree(self):
        release = json.loads((SCRIPTS.parent / "data" / "release_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(release["public_download"]["url"], "https://pan.baidu.com/s/1oHoVxMzDVNkyiV40wphWzw?pwd=4vi1")
        self.assertEqual(release["public_download"]["extraction_code"], "4vi1")
        self.assertEqual(release["public_download"]["folder"], "PointQ-Bench-v1.0")
        self.assertEqual(release["public_download"]["expected_file_count"], 9)
        self.assertEqual(release["public_download"]["expiry"], "permanent")
        self.assertEqual(release["rights"]["code_license"], "MIT for self-owned code and site; not a blanket data license")
        self.assertIs(release["verification"]["remote_redownload_tested"], False)
        self.assertEqual(release["archives"]["images"]["members"], 3083 * 6)
        clouds = release["archives"]["clouds"]
        self.assertEqual(sum(part["bytes"] for part in clouds["parts"]), clouds["bytes"])
        helpers = SCRIPTS / "data"
        for name, expected in TOOL_SHA256.items():
            self.assertEqual(hashlib.sha256((helpers / name).read_bytes()).hexdigest(), expected)
        self.assertEqual((helpers / "PointQ-Bench_PARTS_SHA256SUMS_v1.0.txt").read_text(encoding="ascii").splitlines(), [f"{part['sha256']}  {part['name']}" for part in clouds["parts"]])
        self.assertEqual((helpers / "PointQ-Bench_SHA256SUMS_v1.0.txt").read_text(encoding="ascii").splitlines(), [f"{entry['sha256']}  {entry['name']}" for entry in (clouds, release["archives"]["images"])])


class MetadataExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audit = self.root / "audit"
        self.audit.mkdir()
        self.destination = self.root / "public-data"
        self.helper_destination = self.root / "public-helpers"
        with (SCRIPTS.parent / "data" / "manifest.csv").open(encoding="utf-8", newline="") as stream:
            public = list(csv.DictReader(stream))
        self.rows = [{"index": row["index"], "relative_path": row["pc_relative_path"], "bytes": row["bytes"], "sha256": row["sha256"], "local_path": "PRIVATE_SENTINEL", "mtime_ns": 123} for row in public]
        with (self.audit / "cloud_manifest_20261008.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)
        self.confirmation = {
            "catalog_rows": 3083, "local_clouds": 3083, "dataset_counts": COUNTS,
            "missing": [], "extra": [], "casefold_collisions": 0, "changed_hashes": [],
            "read_errors": [], "annotation_missing": [], "annotation_extra": [], "annotation_duplicates": {},
            "sha256_matches_baseline": 3083, "records": self.rows,
            "csv_checks": {"fixture.csv": {"index_path_mismatches": [], "records": 3083, "unique_paths": 3083}},
            "total_bytes": sum(int(row["bytes"]) for row in self.rows),
        }
        self.save_json("cloud_confirmation_20261008.json", self.confirmation)
        self.release = json.loads((SCRIPTS.parent / "data" / "release_manifest.json").read_text(encoding="utf-8"))
        for kind, entry in self.release["archives"].items():
            report = {
                "valid": True, "payload_valid": True, "data_only": True,
                "path": "PRIVATE_SENTINEL/" + entry["name"],
                "expected_records": entry["members"], "payload_records": entry["members"], "matched_sha256": entry["members"],
                "errors": [], "duplicate_names": [], "unsafe_names": [], "other_files": [], "archive_prefix": "",
                "bytes": entry["bytes"], "archive_sha256": entry["sha256"],
            }
            self.save_json(f"baidu_upload_clean_{kind}_readback_20261008.json", report)
        clouds = self.release["archives"]["clouds"]
        self.save_json("baidu_upload_cloud_parts_20261008.json", {"valid": True, "source_bytes": clouds["bytes"], "source_sha256": clouds["sha256"], "parts": clouds["parts"]})

    def save_json(self, name, data):
        (self.audit / name).write_text(json.dumps(data), encoding="utf-8")

    def export(self, source_helpers=None):
        with contextlib.redirect_stdout(io.StringIO()):
            metadata.prepare(self.audit, self.destination, source_helpers or SCRIPTS / "data", self.helper_destination)

    def set_matching_hash(self, value):
        self.rows[0]["sha256"] = value
        with (self.audit / "cloud_manifest_20261008.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)
        self.save_json("cloud_confirmation_20261008.json", self.confirmation)

    def test_matching_invalid_hash_leaves_no_artifacts_and_allows_retry(self):
        original = self.rows[0]["sha256"]
        self.set_matching_hash("not-a-sha256")
        with self.assertRaisesRegex(ValueError, "Public manifest failed validation"):
            self.export()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.helper_destination.exists())
        self.set_matching_hash(original)
        self.export()
        self.assertEqual((self.destination / "manifest.csv").read_bytes(), (SCRIPTS.parent / "data" / "manifest.csv").read_bytes())

    def test_invalid_schema_preserves_existing_output_trees(self):
        self.export()
        (self.destination / "keep.txt").write_bytes(b"legitimate unrelated output")
        before = {str(path.relative_to(self.root)): path.read_bytes() for folder in (self.destination, self.helper_destination) for path in folder.rglob("*") if path.is_file()}
        self.set_matching_hash("not-a-sha256")
        with self.assertRaisesRegex(ValueError, "Public manifest failed validation"):
            self.export()
        after = {str(path.relative_to(self.root)): path.read_bytes() for folder in (self.destination, self.helper_destination) for path in folder.rglob("*") if path.is_file()}
        self.assertEqual(after, before)

    def test_invalid_archive_metadata_leaves_no_artifacts(self):
        name = "baidu_upload_clean_images_readback_20261008.json"
        report = json.loads((self.audit / name).read_text(encoding="utf-8"))
        for field, value, error in (
            ("bytes", -1, "positive integer"),
            ("bytes", True, "positive integer"),
            ("bytes", float("nan"), "positive integer"),
            ("archive_sha256", "not-a-sha256", "64 lowercase hex digits"),
        ):
            with self.subTest(field=field, value=value):
                invalid = dict(report)
                invalid[field] = value
                self.save_json(name, invalid)
                with self.assertRaisesRegex(ValueError, error):
                    self.export()
                self.assertFalse(self.destination.exists())
                self.assertFalse(self.helper_destination.exists())

    def test_release_serialization_failure_leaves_no_artifacts(self):
        with mock.patch.object(metadata.json, "dumps", side_effect=ValueError("fixture serialization failure")):
            with self.assertRaisesRegex(ValueError, "fixture serialization failure"):
                self.export()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.helper_destination.exists())

    def test_existing_helper_is_preserved_without_creating_public_data(self):
        self.helper_destination.mkdir()
        name = next(iter(TOOL_SHA256))
        target = self.helper_destination / name
        target.write_bytes(b"legitimate existing helper")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.export()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.helper_destination.iterdir()), [target])
        self.assertEqual(target.read_bytes(), b"legitimate existing helper")

    def test_helper_parent_file_conflict_does_not_create_public_data(self):
        blocked = self.root / "blocked"
        blocked.write_bytes(b"legitimate existing file")
        self.helper_destination = blocked / "helpers"
        with self.assertRaisesRegex(ValueError, "Output directory conflicts"):
            self.export()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.helper_destination.exists())
        self.assertEqual(blocked.read_bytes(), b"legitimate existing file")

    def test_reproduce_public_outputs_and_preserve_private_inputs(self):
        before = {path.name: path.read_bytes() for path in self.audit.iterdir()}
        self.export()
        self.assertEqual((self.destination / "manifest.csv").read_bytes(), (SCRIPTS.parent / "data" / "manifest.csv").read_bytes())
        self.assertEqual(json.loads((self.destination / "release_manifest.json").read_text(encoding="utf-8")), self.release)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.audit.iterdir()})
        for path in self.destination.iterdir():
            self.assertNotIn("PRIVATE_SENTINEL", path.read_text(encoding="utf-8"))
        for name in TOOL_SHA256:
            self.assertEqual((self.helper_destination / name).read_bytes(), (SCRIPTS / "data" / name).read_bytes())

    def test_export_refuses_existing_outputs(self):
        self.export()
        before = (self.destination / "manifest.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.export()
        self.assertEqual((self.destination / "manifest.csv").read_bytes(), before)

    def test_export_rejects_audit_disagreement_before_writing(self):
        self.confirmation["records"][0]["sha256"] = "0" * 64
        self.save_json("cloud_confirmation_20261008.json", self.confirmation)
        with self.assertRaisesRegex(ValueError, "CSV and confirmation disagree"):
            self.export()
        self.assertFalse(self.destination.exists())

    def test_export_rejects_unverified_helper_before_writing(self):
        unverified = self.root / "unverified"
        unverified.mkdir()
        (unverified / "merge_pointcloud_parts.py").write_bytes(b"changed helper")
        with self.assertRaisesRegex(ValueError, "Download helper SHA-256 differs"):
            self.export(unverified)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.helper_destination.exists())


if __name__ == "__main__":
    unittest.main()
