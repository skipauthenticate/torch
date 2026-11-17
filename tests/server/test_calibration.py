"""The calibration torch_server uses by default: each pinned model file's own folder, none for any other file.

    python -m unittest discover -s tests/server -p test_calibration.py -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
from torch_server.__main__ import CALIBRATIONS, calibration_for  # noqa: E402


class Calibration(unittest.TestCase):
    def test_each_pinned_file_has_its_own_folder(self):
        self.assertEqual(len(set(CALIBRATIONS.values())), len(CALIBRATIONS))
        for shard1, folder in CALIBRATIONS.items():
            path = calibration_for(f"/models/{shard1}")
            self.assertEqual(path, ROOT / "calibration" / folder)
            for name in ("calib-g.txt", "calib-g-spec.txt", "draft-vocab.txt"):
                self.assertTrue((path / name).is_file(), f"{folder}/{name}")

    def test_other_files_have_none(self):
        self.assertIsNone(calibration_for("/models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf"))
        self.assertIsNone(calibration_for("model.gguf"))


if __name__ == "__main__":
    unittest.main()
