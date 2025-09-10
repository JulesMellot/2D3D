"""
Temporal consistency algorithms for video processing.
"""

import cv2
import numpy as np

class TemporalConsistency:
    def __init__(self, method="optical_flow"):
        """
        Initialize temporal consistency processor.
        
        Args:
            method (str): Consistency method ("optical_flow", "feature_matching")
        """
        self.method = method
        self.previous_frame = None
        self.previous_depth = None
        
    def apply_consistency(self, current_frame, current_depth):
        """
        Apply temporal consistency to depth map.
        
        Args:
            current_frame (np.ndarray): Current video frame
            current_depth (np.ndarray): Current depth map
            
        Returns:
            np.ndarray: Consistent depth map
        """
        if self.previous_frame is None or self.previous_depth is None:
            # First frame, no consistency needed
            self.previous_frame = current_frame.copy()
            self.previous_depth = current_depth.copy()
            return current_depth
        
        if self.method == "optical_flow":
            consistent_depth = self._optical_flow_consistency(
                self.previous_frame, current_frame, 
                self.previous_depth, current_depth
            )
        elif self.method == "feature_matching":
            consistent_depth = self._feature_matching_consistency(
                self.previous_frame, current_frame,
                self.previous_depth, current_depth
            )
        else:
            # No consistency method, return as is
            consistent_depth = current_depth
            
        # Update previous frames
        self.previous_frame = current_frame.copy()
        self.previous_depth = consistent_depth.copy()
        
        return consistent_depth
    
    def _optical_flow_consistency(self, prev_frame, curr_frame, prev_depth, curr_depth):
        """
        Apply optical flow based temporal consistency.
        
        Args:
            prev_frame (np.ndarray): Previous frame
            curr_frame (np.ndarray): Current frame
            prev_depth (np.ndarray): Previous depth map
            curr_depth (np.ndarray): Current depth map
            
        Returns:
            np.ndarray: Consistent depth map
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Warp previous depth map using optical flow
        h, w = prev_depth.shape
        flow_map_x, flow_map_y = np.meshgrid(np.arange(w), np.arange(h))
        flow_map_x = (flow_map_x + flow[..., 0]).astype(np.float32)
        flow_map_y = (flow_map_y + flow[..., 1]).astype(np.float32)
        
        warped_depth = cv2.remap(
            prev_depth, flow_map_x, flow_map_y, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        
        # Blend warped depth with current depth for consistency
        consistency_weight = 0.3  # Adjust this value to control consistency strength
        consistent_depth = (
            consistency_weight * warped_depth + 
            (1 - consistency_weight) * curr_depth
        )
        
        return consistent_depth
    
    def _feature_matching_consistency(self, prev_frame, curr_frame, prev_depth, curr_depth):
        """
        Apply feature matching based temporal consistency.
        
        Args:
            prev_frame (np.ndarray): Previous frame
            curr_frame (np.ndarray): Current frame
            prev_depth (np.ndarray): Previous depth map
            curr_depth (np.ndarray): Current depth map
            
        Returns:
            np.ndarray: Consistent depth map
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB features
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        
        if des1 is None or des2 is None:
            # No features found, return current depth
            return curr_depth
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 4:
            # Not enough matches, return current depth
            return curr_depth
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use top matches for homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        
        try:
            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Warp previous depth using homography
            h, w = prev_depth.shape
            warped_depth = cv2.warpPerspective(
                prev_depth, M, (w, h), flags=cv2.INTER_LINEAR
            )
            
            # Blend for consistency
            consistency_weight = 0.2
            consistent_depth = (
                consistency_weight * warped_depth + 
                (1 - consistency_weight) * curr_depth
            )
            
            return consistent_depth
        except:
            # Homography failed, return current depth
            return curr_depth

# Configuration for temporal consistency
TEMPORAL_CONFIG = {
    "enabled": True,
    "method": "optical_flow",
    "consistency_weight": 0.3,
    "optical_flow_params": {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0
    }
}