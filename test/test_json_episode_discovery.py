import tempfile
import unittest
from pathlib import Path

from unitree_lerobot.utils.convert_unitree_json_to_lerobot import JsonDataset


class JsonEpisodeDiscoveryTest(unittest.TestCase):
    def test_dataset_task_and_episode_roots(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task = root / "conveyor"
            episodes = [task / "episode_0000", task / "episode_0001"]
            for episode in episodes:
                episode.mkdir(parents=True)
                (episode / "data.json").write_text('{"data": []}')
            (task / "notes.txt").write_text("Not an episode")
            (task / "unfinished_episode").mkdir()
            (root / "xr_records" / "empty_task").mkdir(parents=True)

            for raw_dir, expected in [(root, episodes), (task, episodes), (episodes[0], episodes[:1])]:
                with self.subTest(raw_dir=raw_dir):
                    dataset = JsonDataset(raw_dir, "Unitree_G1_Dex1_TorsoYaw")
                    self.assertEqual(dataset.episode_paths, [str(path) for path in expected])
                    self.assertEqual(len(dataset.episodes_data_cached), len(expected))

    def test_nested_recordings_and_episode_assets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            episode = root / "xr_records" / "task" / "episode_0000"
            assets = episode / "colors"
            assets.mkdir(parents=True)
            (episode / "data.json").write_text('{"data": []}')
            (assets / "data.json").write_text("Not episode JSON")
            hidden = root / ".backup"
            hidden.mkdir()
            (hidden / "data.json").write_text("Not episode JSON")

            dataset = JsonDataset(root, "Unitree_G1_Dex1_TorsoYaw")
            self.assertEqual(dataset.episode_paths, [str(episode)])

    def test_empty_or_missing_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            for raw_dir in [Path(temporary_dir), Path(temporary_dir) / "missing"]:
                with self.subTest(raw_dir=raw_dir):
                    with self.assertRaisesRegex(FileNotFoundError, "No episodes containing data.json"):
                        JsonDataset(raw_dir, "Unitree_G1_Dex1_TorsoYaw")


if __name__ == "__main__":
    unittest.main()
