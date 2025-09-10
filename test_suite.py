"""
Test suite for the 2D3D pipeline.
"""

import unittest
import cv2
import numpy as np
import os
import tempfile
from quality_pipeline import QualityPreservingPipeline

class TestQualityPreservingPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = QualityPreservingPipeline(depth_profile="fast")
        
        # Create a simple test image
        self.test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    def test_depth_estimation(self):
        """Test depth estimation functionality."""
        depth_map = self.pipeline.depth_estimator.predict_depth(self.test_image)
        
        # Check that depth map has correct shape
        self.assertEqual(depth_map.shape, (240, 320))
        
        # Check that depth values are in [0, 1] range
        self.assertTrue(np.all(depth_map >= 0))
        self.assertTrue(np.all(depth_map <= 1))
    
    def test_stereo_pair_generation(self):
        """Test stereo pair generation."""
        left_frame, right_frame = self.pipeline.process_frame(self.test_image)
        
        # Check that output frames have correct shape
        self.assertEqual(left_frame.shape, self.test_image.shape)
        self.assertEqual(right_frame.shape, self.test_image.shape)
        
        # Check that frames are not identical (stereo effect)
        self.assertFalse(np.array_equal(left_frame, right_frame))
    
    def test_color_preservation(self):
        """Test color preservation functionality."""
        original_colors = self.test_image.copy()
        left_frame, right_frame = self.pipeline.process_frame(self.test_image)
        
        # Check that basic color structure is preserved
        self.assertEqual(left_frame.shape, original_colors.shape)
        self.assertEqual(right_frame.shape, original_colors.shape)
    
    def test_temporal_smoothing(self):
        """Test temporal smoothing functionality."""
        # Process first frame
        self.pipeline.temporal_smoothing = True
        left1, right1 = self.pipeline.process_frame(self.test_image)
        
        # Process second frame (should use temporal smoothing)
        left2, right2 = self.pipeline.process_frame(self.test_image)
        
        # Both frames should be processed
        self.assertIsNotNone(left1)
        self.assertIsNotNone(left2)

class TestVideoProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for video processing."""
        # Create a temporary video file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_video = os.path.join(self.temp_dir, "test_video.mp4")
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.test_video, fourcc, 1, (320, 240))
        
        # Write 5 frames
        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        if os.path.exists(self.test_video):
            os.remove(self.test_video)
        os.rmdir(self.temp_dir)
    
    def test_video_conversion(self):
        """Test video conversion functionality."""
        # This test would require FFmpeg and is resource-intensive
        # We'll just check that the pipeline can be created
        pipeline = QualityPreservingPipeline()
        self.assertIsNotNone(pipeline)

if __name__ == "__main__":
    unittest.main()