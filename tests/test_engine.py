import unittest
import numpy as np
from negpy.services.rendering.engine import DarkroomEngine
from negpy.domain.models import WorkspaceConfig


class TestDarkroomEngine(unittest.TestCase):
    def test_pipeline_execution(self):
        """End-to-end pipeline smoke test."""
        engine = DarkroomEngine()
        img = np.random.rand(100, 100, 3).astype(np.float32)
        # Default config should preserve the full image until auto-crop is explicitly enabled.
        settings = WorkspaceConfig()

        res = engine.process(img, settings, source_hash="dummy")

        self.assertEqual(res.shape[:2], (100, 100))
        self.assertLessEqual(np.max(res), 1.0)
        self.assertGreaterEqual(np.min(res), 0.0)

    def test_pipeline_with_offset(self):
        """Engine respects geometry settings."""
        engine = DarkroomEngine()
        img = np.random.rand(200, 200, 3).astype(np.float32)
        # Use explicit auto-crop plus offset to shrink image.
        settings = WorkspaceConfig.from_flat_dict({"auto_crop_enabled": True, "autocrop_offset": 10})

        res = engine.process(img, settings, source_hash="dummy")

        self.assertLess(res.shape[0], 200)
        self.assertLess(res.shape[1], 200)

    def test_engine_caching(self):
        """Check intermediate result caching."""
        engine = DarkroomEngine()
        img = np.random.rand(100, 100, 3).astype(np.float32)
        settings = WorkspaceConfig()

        res1 = engine.process(img, settings, source_hash="file1")
        assert engine.cache.base is not None
        assert engine.cache.exposure is not None
        base_id = id(engine.cache.base.data)

        res2 = engine.process(img, settings, source_hash="file1")
        assert id(engine.cache.base.data) == base_id
        assert np.array_equal(res1, res2)

        img2 = np.random.rand(100, 100, 3).astype(np.float32)
        res3 = engine.process(img2, settings, source_hash="file2")
        assert engine.cache.source_hash == "file2"
        assert not np.array_equal(res1, res3)

    def test_pipeline_produces_metrics(self):
        """Verify pipeline populates expected metrics."""
        from negpy.domain.interfaces import PipelineContext

        engine = DarkroomEngine()
        img = np.random.rand(100, 100, 3).astype(np.float32)
        settings = WorkspaceConfig()
        context = PipelineContext(scale_factor=1.0, original_size=(100, 100))

        engine.process(img, settings, source_hash="test", context=context)

        self.assertIn("normalized_log", context.metrics)
        self.assertIn("log_bounds", context.metrics)


if __name__ == "__main__":
    unittest.main()
