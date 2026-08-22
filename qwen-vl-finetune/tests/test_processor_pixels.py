import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from transformers.image_utils import SizeDict


FINETUNE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINETUNE_ROOT))

from qwenvl.data.data_processor import update_processor_pixels  # noqa: E402


class ProcessorPixelsTest(unittest.TestCase):
    def test_updates_frozen_video_size_dict(self):
        image_size = {"shortest_edge": 16, "longest_edge": 32}
        video_size = SizeDict(
            shortest_edge=64,
            longest_edge=128,
            max_height=720,
            max_width=1280,
        )
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(
                min_pixels=16,
                max_pixels=32,
                size=image_size,
            ),
            video_processor=SimpleNamespace(
                min_pixels=64,
                max_pixels=128,
                min_frames=4,
                max_frames=8,
                fps=1.0,
                size=video_size,
            ),
        )
        data_args = SimpleNamespace(
            min_pixels=256,
            max_pixels=512,
            video_min_pixels=1024,
            video_max_pixels=2048,
            video_min_frames=8,
            video_max_frames=32,
            video_fps=2.0,
        )

        result = update_processor_pixels(processor, data_args)

        self.assertIs(result, processor)
        self.assertEqual(
            processor.image_processor.size,
            {"shortest_edge": 256, "longest_edge": 512},
        )
        self.assertIsNot(processor.video_processor.size, video_size)
        self.assertEqual(processor.video_processor.size.shortest_edge, 1024)
        self.assertEqual(processor.video_processor.size.longest_edge, 2048)
        self.assertEqual(processor.video_processor.size.max_height, 720)
        self.assertEqual(processor.video_processor.size.max_width, 1280)


if __name__ == "__main__":
    unittest.main()
