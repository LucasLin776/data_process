"""验证SNR、标签、来源隔离和写出质量，不用于证明真实识别性能。"""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy.signal import freqz

from augment import (
    Source,
    apply_equalizer,
    event_clip,
    export_markers,
    generate,
    load_config,
    mix_at_snr,
    read_audio,
    read_markers,
    random_eq_settings,
    renumber_existing,
    sample_equalizer,
    sha256,
)


class MixingTests(unittest.TestCase):
    def test_eq_band_gains_alignment_and_no_circular_wrap(self):
        cfg = {"sample_rate": 32000, "random_eq": {"enabled": True}}
        taps, log = sample_equalizer(cfg, np.random.default_rng(3))
        _, response = freqz(taps, worN=np.array([100., 1500., 10000.]), fs=32000)
        actual = 20 * np.log10(np.abs(response))
        np.testing.assert_allclose(actual, list(log["gain_db"].values()), atol=.05)
        self.assertEqual(log["delay_compensated_samples"], 256)
        self.assertEqual(log["maximum_temporal_spread_s"], .008)
        impulse = np.zeros(32000)
        impulse[12000] = 1.
        transformed = apply_equalizer(impulse, taps)
        self.assertEqual(len(transformed), len(impulse))
        self.assertEqual(np.argmax(np.abs(transformed)), 12000)
        np.testing.assert_allclose(transformed[:11000], 0, atol=1e-14)
        np.testing.assert_allclose(transformed[13000:], 0, atol=1e-14)

    def test_flat_eq_and_disabled_eq_preserve_signal(self):
        x = np.random.default_rng(10).normal(0, .1, 16000)
        cfg = {"sample_rate": 32000, "random_eq": {"enabled": True, "gain_db": [0, 0]}}
        taps, _ = sample_equalizer(cfg, np.random.default_rng(1))
        np.testing.assert_allclose(apply_equalizer(x, taps), x, atol=1e-14)
        taps, log = sample_equalizer({"sample_rate": 32000}, np.random.default_rng(1))
        self.assertIsNone(taps)
        self.assertFalse(log["applied"])
        self.assertIs(apply_equalizer(x, taps), x)

    def test_eq_pair_shares_response_and_achieves_snr_after_filtering(self):
        rng = np.random.default_rng(8)
        bg, fg = rng.normal(0, .2, 16000), rng.normal(0, .1, 4000)
        taps, _ = sample_equalizer({"sample_rate": 32000, "random_eq": {"enabled": True}}, rng)
        for snr in (-5, 0, 10):
            positive, negative, log = mix_at_snr(bg, fg, 5000, 100, 3900, snr, .95, -3, eq_taps=taps)
            np.testing.assert_allclose(negative, apply_equalizer(bg, taps) * log["common_output_gain"])
            event = positive - negative
            active = slice(5100, 8900)
            actual = 10 * np.log10(np.mean(event[active] ** 2) / np.mean(negative[active] ** 2))
            self.assertAlmostEqual(actual, snr, places=10)
            self.assertTrue(log["snr_measured_after_eq"])
            np.testing.assert_allclose(positive[:4700], negative[:4700], atol=1e-14)
            np.testing.assert_allclose(positive[9300:], negative[9300:], atol=1e-14)
            self.assertLessEqual(max(abs(positive).max(), abs(negative).max()), .95)

    def test_eq_invalid_gains_frequencies_and_taps_rejected(self):
        cases = [[], {"enabled": "yes"}, {"gain_db": [-7, 3]},
                 {"gain_db": [0, float("nan")]}, {"crossovers_hz": [4000, 500]},
                 {"crossovers_hz": [500, 15000]}, {"transition_fraction": 0},
                 {"fir_taps": 512}, {"source_variants": 0}]
        for raw in cases:
            cfg = {"sample_rate": 32000,
                   "random_eq": {"enabled": True, **raw} if isinstance(raw, dict) else raw}
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                random_eq_settings(cfg)

    def test_snr_and_paired_background_share_gain(self):
        rng = np.random.default_rng(5)
        bg = rng.normal(0, .3, 16000)
        fg = rng.normal(0, .2, 4000)
        for snr in (-5, 0, 10):
            positive, negative, log = mix_at_snr(bg, fg, 5000, 100, 3900, snr, .95, -3)
            np.testing.assert_allclose(negative, bg * log["common_output_gain"])
            event = positive - negative
            span = slice(5100, 8900)
            actual = 10 * np.log10(np.mean(event[span] ** 2) / np.mean(negative[span] ** 2))
            self.assertAlmostEqual(actual, snr, places=10)
            np.testing.assert_array_equal(positive[:5000], negative[:5000])
            np.testing.assert_array_equal(positive[9000:], negative[9000:])
            self.assertLessEqual(max(abs(positive).max(), abs(negative).max()), .95)

    def test_no_clipping_even_when_sources_add_at_same_peak(self):
        bg, fg = np.ones(1000) * .9, np.ones(100) * .9
        positive, negative, log = mix_at_snr(bg, fg, 200, 0, 100, 10, .95, 0)
        self.assertAlmostEqual(max(positive), .95)
        self.assertLess(max(negative), .95)
        self.assertLess(log["headroom_gain"], 1)

    def test_silence_and_invalid_placement_rejected(self):
        for start, bg in [(0, np.zeros(1000)), (950, np.ones(1000))]:
            with self.assertRaises(ValueError):
                mix_at_snr(bg, np.ones(100), start, 0, 100, 0, .95, 0)

    def test_fades_do_not_change_annotated_event(self):
        sr = 1000
        source = Source("fg", "foreground", Path("unused"), np.ones(sr), {},
                        [{"start_s": .2, "end_s": .4}], None)
        clip, left, right = event_clip(source, 0, {"sample_rate": sr, "context_s": .02, "fade_s": .05})
        np.testing.assert_array_equal(clip[left:right], np.ones(200))
        self.assertEqual(clip[0], 0)
        self.assertEqual(clip[-1], 0)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_config(self):
        sr = 8000
        rng = np.random.default_rng(4)
        fg = np.zeros(sr * 2)
        for left, right in [(3200, 5200), (8000, 9600)]:
            fg[left:right] = rng.normal(0, .12, right - left)
        sf.write(self.root / "fg.wav", fg, sr, subtype="PCM_16")
        sf.write(self.root / "bg.wav", rng.normal(0, .05, sr * 2), sr, subtype="PCM_16")
        sf.write(self.root / "interference.wav", rng.normal(0, .08, sr * 2), sr, subtype="PCM_16")
        cfg = {
            "version": 1, "sample_rate": sr, "duration_s": 2, "channel": 0, "seed": 7,
            "group_id": "one_training_component", "snr_db": [0, 10], "repeats_per_snr": 1,
            "output_gain_db": [-6, 0], "peak_limit": .95, "context_s": .02, "fade_s": .005,
            "foregrounds": [{"id": "bolt", "audio": "fg.wav", "reviewed": True,
                             "events": [{"start_s": .4, "end_s": .65}, {"start_s": 1., "end_s": 1.2}]}],
            "backgrounds": [{"id": "bg", "audio": "bg.wav", "reviewed": True}],
            "hard_negatives": [{"id": "other", "audio": "interference.wav", "reviewed": True}],
        }
        path = self.root / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return cfg, path

    def test_user_audition_format_and_multiple_events_roundtrip(self):
        path = self.root / "标记.csv"
        path.write_text("Name\tStart\tDuration\tTime Format\tType\tDescription\n"
                        "bolt_drop\t0:12.808\t0:04.764\tdecimal\tCue\tfirst\n"
                        "bolt_drop\t0:24.000\t0:01.000\tdecimal\tCue\tsecond\n", encoding="utf-8-sig")
        events = read_markers(path, 30)
        self.assertEqual(events, [{"start_s": 12.808, "end_s": 17.572}, {"start_s": 24., "end_s": 25.}])
        out = self.root / "roundtrip.csv"
        export_markers(out, events)
        self.assertEqual(read_markers(out, 30), events)

    def test_point_empty_overlap_wrong_name_and_outside_markers_rejected(self):
        header = "Name\tStart\tDuration\tTime Format\n"
        bodies = ["", "bolt_drop\t1\t0\tdecimal\n", "wrong\t1\t1\tdecimal\n",
                  "bolt_drop\t29\t2\tdecimal\n",
                  "bolt_drop\t1\t2\tdecimal\nbolt_drop\t2\t2\tdecimal\n"]
        for i, body in enumerate(bodies):
            path = self.root / f"bad_{i}.csv"
            path.write_text(header + body, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_markers(path, 30)

    def test_generate_multiple_events_labels_quality_sources_and_reproducibility(self):
        _, config = self.make_config()
        before = {p: sha256(p) for p in self.root.glob("*.wav")}
        first, second = self.root / "out1", self.root / "out2"
        summary = generate(config, first)
        self.assertEqual(summary["total_recordings"], 11)
        self.assertEqual(summary["positive_recordings"], 5)
        self.assertEqual(summary["paired_negative_recordings"], 4)
        self.assertFalse(summary["output_near_full_scale_samples"])
        self.assertFalse((first / "RUNNING.json").exists())
        rows = [json.loads(line) for line in (first / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["split"] for row in rows}, {"train"})
        self.assertEqual({row["group_id"] for row in rows}, {"one_training_component"})
        self.assertEqual([row["recording_id"] for row in rows], [f"{index:02d}" for index in range(1, 12)])
        for row in rows:
            x, sr = sf.read(first / row["audio"])
            self.assertEqual(len(x), sr * 2)
            self.assertLessEqual(abs(x).max(), .95004)
            for path in row["plots"].values():
                self.assertEqual((first / path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(Path(path).stem, row["recording_id"])
            if row["events"]:
                imported = read_markers(first / row["labels"], 2)
                self.assertEqual(imported, row["events"])
        original = next(row for row in rows if row["derivation_id"] == "original_bolt")
        self.assertEqual(len(original["events"]), 2)
        generate(config, second)
        for path in (first / "audio").glob("*.wav"):
            self.assertEqual(sha256(path), sha256(second / "audio" / path.name))
        self.assertEqual(before, {p: sha256(p) for p in before})
        with self.assertRaises(FileExistsError):
            generate(config, first)

    def test_validation_rejects_unreviewed_nontrain_duplicate_and_short_sources(self):
        cfg, path = self.make_config()
        cases = []
        altered = copy.deepcopy(cfg)
        altered["backgrounds"][0]["reviewed"] = False
        cases.append(altered)
        altered = copy.deepcopy(cfg)
        altered["backgrounds"][0]["split"] = "test"
        cases.append(altered)
        altered = copy.deepcopy(cfg)
        altered["hard_negatives"][0]["audio"] = "bg.wav"
        cases.append(altered)
        altered = copy.deepcopy(cfg)
        altered["duration_s"] = 3
        cases.append(altered)
        for altered in cases:
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_eq_generation_covers_all_roles_preserves_labels_and_is_reproducible(self):
        cfg, config = self.make_config()
        cfg["random_eq"] = {"enabled": True, "crossovers_hz": [400, 1500]}
        config.write_text(json.dumps(cfg), encoding="utf-8")
        first, second, baseline = self.root / "eq1", self.root / "eq2", self.root / "baseline"
        # 绘图已有集成覆盖；本测试隔离EQ、标签、SNR、复现及随机流逻辑。
        with patch("augment.render_audio_plots", return_value={}):
            summary = generate(config, first)
            generate(config, second)
            cfg["random_eq"]["enabled"] = False
            config.write_text(json.dumps(cfg), encoding="utf-8")
            generate(config, baseline)
        self.assertEqual(summary["total_recordings"], 14)
        self.assertEqual(summary["positive_recordings"], 6)
        self.assertEqual(summary["equalized_source_recordings"], 3)
        self.assertEqual(summary["random_eq_augmented_recordings"], 11)

        def load_logs(root):
            return [json.loads(line) for line in (root / "augmentation_log.jsonl").read_text(encoding="utf-8").splitlines()]

        logs = load_logs(first)
        baseline_logs = {row["derivation_id"]: row for row in load_logs(baseline)}
        pairs = {}
        eq_roles = set()
        for row in logs:
            self.assertEqual(sha256(first / row["audio"]), sha256(second / row["audio"]))
            self.assertLessEqual(sf.read(first / row["audio"])[0].max(), .95004)
            details = row["augmentation"]
            if details["kind"].startswith("equalized_source_"):
                eq_roles.add(details["kind"])
                original = baseline_logs["original_" + row["source_ids"][0]]
                self.assertEqual(row["events"], original["events"])
                self.assertTrue(details["random_eq"]["applied"])
            if "pair_id" in details:
                pairs.setdefault(details["pair_id"], []).append(row)
                original = baseline_logs[row["derivation_id"]]
                for key in ("background_offset_s", "foreground_context_position_s", "output_gain_db_requested"):
                    self.assertEqual(details[key], original["augmentation"][key])
                self.assertEqual(row["events"], original["events"])
                self.assertTrue(details["snr_measured_after_eq"])
            if row["events"]:
                self.assertEqual(read_markers(first / row["labels"], 2), row["events"])
        self.assertEqual(eq_roles, {"equalized_source_foreground", "equalized_source_background", "equalized_source_hard_negative"})
        for pair in pairs.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["augmentation"]["random_eq"], pair[1]["augmentation"]["random_eq"])
            self.assertEqual(pair[0]["augmentation"]["common_output_gain"], pair[1]["augmentation"]["common_output_gain"])

    def test_resampling_channel_selection_and_quality_warning(self):
        left = np.ones(16000) * .1
        left[:100] = -1
        right = -left
        path = self.root / "stereo.wav"
        sf.write(path, np.column_stack([left, right]), 16000, subtype="PCM_16")
        x, quality = read_audio(path, 8000, channel=0)
        self.assertEqual(len(x), 8000)
        self.assertTrue(quality["warnings"])
        self.assertGreater(np.sqrt(np.mean(x ** 2)), .01)
        with self.assertRaises(ValueError):
            read_audio(path, 8000, channel=2)

    def make_legacy_dataset(self):
        self.make_config()
        root = self.root / "legacy"
        (root / "audio").mkdir(parents=True)
        (root / "labels").mkdir()
        rows = []
        # 故意让旧文件名和目标编号形成交换，验证不会覆盖另一个样本。
        for old_id, source, events in [
            ("02", "fg.wav", [{"start_s": .4, "end_s": .65}]),
            ("01", "bg.wav", []),
        ]:
            audio = f"audio/{old_id}.wav"
            (root / audio).write_bytes((self.root / source).read_bytes())
            labels = f"labels/{old_id}.csv" if events else None
            if labels:
                export_markers(root / labels, events)
            rows.append({"recording_id": old_id, "audio": audio, "labels": labels,
                         "events": events, "group_id": "train_group", "split": "train", "reviewed": True})
        for name in ("train.jsonl", "augmentation_log.jsonl"):
            (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with (root / "quality.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["recording_id", "rms"])
            writer.writeheader()
            writer.writerows({"recording_id": row["recording_id"], "rms": .1} for row in rows)
        (root / "summary.json").write_text(json.dumps({"status": "complete", "total_recordings": 2}), encoding="utf-8")
        return root, rows

    def test_migration_preserves_wav_content_and_updates_all_references(self):
        root, before = self.make_legacy_dataset()
        hashes = [sha256(root / row["audio"]) for row in before]
        result = renumber_existing(root)
        after = [json.loads(line) for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["recording_id"] for row in after], ["01", "02"])
        self.assertEqual([sha256(root / row["audio"]) for row in after], hashes)
        self.assertEqual([sha256(Path(result["backup"]) / row["audio"]) for row in before], hashes)
        self.assertEqual(read_markers(root / "labels/01.csv", 2), before[0]["events"])
        self.assertEqual(len(list((root / "waveforms").glob("*.png"))), 2)
        self.assertEqual(len(list((root / "spectrograms").glob("*.png"))), 2)
        logs = [json.loads(line) for line in (root / "augmentation_log.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(logs, after)
        with (root / "quality.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual([row["recording_id"] for row in csv.DictReader(stream)], ["01", "02"])
        with (root / "renaming_map.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual([row["old_recording_id"] for row in csv.DictReader(stream)], ["02", "01"])
        self.assertEqual(renumber_existing(root)["status"], "unchanged")

    def test_migration_rejects_unlisted_audio_without_modifying_originals(self):
        root, rows = self.make_legacy_dataset()
        hashes = [sha256(root / row["audio"]) for row in rows]
        (root / "audio/extra.wav").write_bytes((self.root / "bg.wav").read_bytes())
        with self.assertRaises(ValueError):
            renumber_existing(root)
        self.assertEqual([sha256(root / row["audio"]) for row in rows], hashes)
        self.assertFalse((self.root / "_backups").exists())

    def test_migration_rejects_file_references_outside_dataset(self):
        root, rows = self.make_legacy_dataset()
        rows[0]["audio"] = "../fg.wav"
        (root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaises(ValueError):
            renumber_existing(root)

    def test_migration_rolls_back_if_metadata_cannot_be_installed(self):
        root, rows = self.make_legacy_dataset()
        hashes = [sha256(root / row["audio"]) for row in rows]
        manifest = (root / "train.jsonl").read_bytes()
        import shutil

        original_copy = shutil.copy2

        def fail_on_manifest(source, destination, **kwargs):
            if Path(source).name == "train.jsonl" and Path(source).parent.name.startswith(".legacy_renumber_"):
                raise OSError("simulated metadata lock")
            return original_copy(source, destination, **kwargs)

        with patch("augment.shutil.copy2", side_effect=fail_on_manifest), self.assertRaises(OSError):
            renumber_existing(root)
        self.assertEqual((root / "train.jsonl").read_bytes(), manifest)
        self.assertEqual([sha256(root / row["audio"]) for row in rows], hashes)
        self.assertFalse((root / "RENUMBERING.json").exists())


if __name__ == "__main__":
    unittest.main()
