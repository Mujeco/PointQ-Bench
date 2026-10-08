"""Offline contract checks for the dependency-free project page."""

import json
import hashlib
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


class SiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (DOCS / "index.html").read_text(encoding="utf-8")
        cls.css = (DOCS / "styles.css").read_text(encoding="utf-8")
        cls.page = PageParser()
        cls.page.feed(cls.html)
        cls.visible_text = re.sub(r"\s+", " ", " ".join(cls.page.text))

    def test_ids_unique(self):
        ids = [attrs["id"] for _, attrs in self.page.nodes if "id" in attrs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_anchor_targets_exist(self):
        ids = {attrs["id"] for _, attrs in self.page.nodes if "id" in attrs}
        for tag, attrs in self.page.nodes:
            if tag == "a" and attrs.get("href", "").startswith("#"):
                fragment = attrs["href"][1:]
                if fragment:
                    self.assertIn(fragment, ids)

    def test_local_assets_exist(self):
        for tag, attrs in self.page.nodes:
            for key in ["href", "src"]:
                value = attrs.get(key, "")
                if value and not value.startswith("#") and not urlsplit(value).scheme:
                    self.assertTrue((DOCS / value).is_file(), (tag, key, value))
        for value in re.findall(r'url\(["\']?([^"\')]+)', self.css):
            self.assertTrue((DOCS / value).is_file(), value)

    def test_accessible_images_and_buttons(self):
        for tag, attrs in self.page.nodes:
            if tag == "img":
                self.assertIn("alt", attrs)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn("prefers-reduced-motion", self.css)
        self.assertIn('class="skip-link"', self.html)

    def test_data_link_and_code_match_manifest(self):
        manifest = json.loads((ROOT / "data" / "release_manifest.json").read_text(encoding="utf-8"))
        self.assertIn("1oHoVxMzDVNkyiV40wphWzw", json.dumps(manifest))
        self.assertIn("4vi1", json.dumps(manifest))
        links = [attrs["href"] for tag, attrs in self.page.nodes if tag == "a" and "pan.baidu.com" in attrs.get("href", "")]
        self.assertEqual(len(links), 2)
        self.assertTrue(all(link == "https://pan.baidu.com/s/1oHoVxMzDVNkyiV40wphWzw?pwd=4vi1" for link in links))

    def test_paper_case_is_not_live_inference(self):
        self.assertIn("not an evaluation input", self.visible_text)
        self.assertIn("not a live model prediction", self.visible_text)
        self.assertIn("not included in this initial package", (ROOT / "README.md").read_text(encoding="utf-8"))

    def test_paper_case_assets_match_extracted_originals(self):
        originals = {
            "ji-ge-pointcloud.jpg": "ea9a8faa2f41ec9a7f660543740acb2f6f3048064c9aa9abad7129ca42abbfe6",
            "ji-ge-reference.jpg": "99de6c726a383590f09c22f59d56aeac3347ae123176814afcbcd432da73937a",
        }
        for name, digest in originals.items():
            self.assertEqual(hashlib.sha256((DOCS / "assets" / name).read_bytes()).hexdigest(), digest)

    def test_paper_case_controls_match_images(self):
        buttons = {attrs["data-case-view"]: attrs for tag, attrs in self.page.nodes if tag == "button" and "data-case-view" in attrs}
        images = {attrs["data-case-image"]: attrs for tag, attrs in self.page.nodes if tag == "img" and "data-case-image" in attrs}
        self.assertEqual(set(buttons), {"pointcloud", "reference"})
        self.assertEqual(set(images), set(buttons))
        self.assertEqual(buttons["pointcloud"]["aria-pressed"], "true")
        self.assertEqual(buttons["reference"]["aria-pressed"], "false")
        self.assertNotIn("hidden", images["pointcloud"])
        self.assertIn("hidden", images["reference"])
        self.assertNotIn("chicken-canvas", self.html)
        self.assertNotIn("rotate-toggle", self.html)

    def test_no_external_scripts_or_fonts(self):
        for tag, attrs in self.page.nodes:
            if tag == "script":
                self.assertFalse(urlsplit(attrs.get("src", "")).scheme)
        self.assertNotIn("fonts.googleapis", self.css)
        self.assertIn("SpaceGrotesk-OFL.txt", (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"))

    def test_authors_and_citation(self):
        self.assertIn("10.1145/3767308.3836267", self.html)
        for name in ["Duanchu Wang", "Cheng Li", "Junjie Yang", "Jing Huang", "Zihang Cheng", "Zhi Gao", "Bohong Zhu", "Di Wang"]:
            self.assertIn(name, self.visible_text)
        self.assertEqual(sum(1 for tag, _ in self.page.nodes if tag == "h1"), 1)


if __name__ == "__main__":
    unittest.main()
