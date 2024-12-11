## /main.py (by Cdaprod & Claude)
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip
from dataclasses import dataclass
import asyncio
from typing import List, Tuple, Optional
import logging
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

@dataclass
class FocusPoint:
    x: float  # center x coordinate (normalized 0-1)
    y: float  # center y coordinate (normalized 0-1)
    scale: float  # zoom level (1.0 = no zoom)
    frame: int  # video frame number
    duration: int  # how many frames to hold this focus
    transition_frames: int = 30  # frames to transition from previous point

@dataclass
class VideoConfig:
    width: int = 1080  # output width
    height: int = 1920  # output height (vertical format)
    fps: int = 30
    min_zoom: float = 1.0
    max_zoom: float = 2.5
    min_motion_threshold: float = 0.02
    smoothing_window: int = 15

class SmartScreenFocus:
    def __init__(self, config: VideoConfig = VideoConfig()):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def process_video(self, input_path: str, output_path: str) -> None:
        """Process a screen recording and create a dynamic focus version"""
        try:
            # Load video
            video = VideoFileClip(input_path)
            
            # Detect focus points
            focus_points = await self._detect_focus_points(video)
            
            # Smooth camera movement
            smoothed_points = self._smooth_focus_points(focus_points)
            
            # Create output video with camera movement
            output_video = self._apply_camera_movement(video, smoothed_points)
            
            # Write final video
            output_video.write_videofile(
                output_path,
                fps=self.config.fps,
                codec='libx264',
                audio_codec='aac'
            )
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise

    async def _detect_focus_points(self, video: VideoFileClip) -> List[FocusPoint]:
        """Detect points of interest in the video"""
        focus_points = []
        prev_frame = None
        frame_count = 0
        
        for frame in video.iter_frames():
            # Convert frame to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(gray, prev_frame)
                
                # Detect significant changes
                if self._is_significant_change(diff):
                    # Find region of interest
                    roi = self._find_roi(diff)
                    if roi:
                        x, y, w, h = roi
                        # Calculate focus point
                        focus_point = FocusPoint(
                            x=x / frame.shape[1],
                            y=y / frame.shape[0],
                            scale=self._calculate_optimal_zoom(w, h, frame.shape),
                            frame=frame_count,
                            duration=self.config.fps  # hold for 1 second by default
                        )
                        focus_points.append(focus_point)
            
            prev_frame = gray
            frame_count += 1
        
        return focus_points

    def _is_significant_change(self, diff_frame: np.ndarray) -> bool:
        """Determine if frame difference is significant enough"""
        motion_ratio = np.mean(diff_frame > 30) # threshold for motion detection
        return motion_ratio > self.config.min_motion_threshold

    def _find_roi(self, diff_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find region of interest in difference frame"""
        # Threshold the difference frame
        _, thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)

    def _calculate_optimal_zoom(
        self, 
        roi_width: int, 
        roi_height: int, 
        frame_shape: Tuple[int, int]
    ) -> float:
        """Calculate optimal zoom level for ROI"""
        # Calculate zoom needed to make ROI fill 50% of frame
        width_zoom = frame_shape[1] / (roi_width * 2)
        height_zoom = frame_shape[0] / (roi_height * 2)
        
        # Use smaller zoom to ensure ROI fits
        zoom = min(width_zoom, height_zoom)
        
        # Clamp zoom to configured limits
        return max(min(zoom, self.config.max_zoom), self.config.min_zoom)

    def _smooth_focus_points(self, points: List[FocusPoint]) -> List[FocusPoint]:
        """Apply smoothing to focus point movements"""
        if not points:
            return points
            
        # Extract coordinates and scales
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        scales = [p.scale for p in points]
        
        # Apply Savitzky-Golay filter for smooth camera movement
        window = min(self.config.smoothing_window, len(points))
        if window % 2 == 0:
            window -= 1
        
        if window > 2:  # Need at least 3 points for smoothing
            xs = savgol_filter(xs, window, 3)
            ys = savgol_filter(ys, window, 3)
            scales = savgol_filter(scales, window, 3)
        
        # Recreate smoothed focus points
        smoothed = []
        for i in range(len(points)):
            smoothed.append(FocusPoint(
                x=xs[i],
                y=ys[i],
                scale=scales[i],
                frame=points[i].frame,
                duration=points[i].duration,
                transition_frames=points[i].transition_frames
            ))
        
        return smoothed

    def _apply_camera_movement(
        self, 
        video: VideoFileClip, 
        focus_points: List[FocusPoint]
    ) -> VideoClip:
        """Apply camera movement to video based on focus points"""
        
        def transform_frame(get_frame, t):
            # Get original frame
            frame = get_frame(t)
            
            # Find current focus point
            current_frame = int(t * self.config.fps)
            focus = self._get_focus_for_frame(focus_points, current_frame)
            
            if focus is None:
                return frame
                
            # Calculate transform matrix
            transform = self._calculate_transform(
                frame.shape,
                focus.x,
                focus.y,
                focus.scale
            )
            
            # Apply transform
            transformed = cv2.warpAffine(
                frame,
                transform,
                (frame.shape[1], frame.shape[0]),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            return transformed
        
        # Create new video with transforms
        return VideoClip(transform_frame, duration=video.duration)

    def _get_focus_for_frame(
        self, 
        focus_points: List[FocusPoint], 
        frame: int
    ) -> Optional[FocusPoint]:
        """Get interpolated focus point for given frame"""
        # Find surrounding focus points
        prev_point = None
        next_point = None
        
        for point in focus_points:
            if point.frame <= frame:
                prev_point = point
            if point.frame > frame and next_point is None:
                next_point = point
                break
        
        if prev_point is None:
            return next_point
        if next_point is None:
            return prev_point
            
        # Interpolate between points
        frame_diff = next_point.frame - prev_point.frame
        if frame_diff == 0:
            return prev_point
            
        t = (frame - prev_point.frame) / frame_diff
        
        # Apply smooth easing
        t = self._ease_in_out(t)
        
        return FocusPoint(
            x=self._lerp(prev_point.x, next_point.x, t),
            y=self._lerp(prev_point.y, next_point.y, t),
            scale=self._lerp(prev_point.scale, next_point.scale, t),
            frame=frame,
            duration=1
        )

    def _calculate_transform(
        self, 
        shape: Tuple[int, int], 
        focus_x: float, 
        focus_y: float, 
        scale: float
    ) -> np.ndarray:
        """Calculate transformation matrix for given focus point"""
        # Convert normalized coordinates to pixels
        center_x = int(focus_x * shape[1])
        center_y = int(focus_y * shape[0])
        
        # Create transform matrix
        transform = cv2.getRotationMatrix2D(
            (center_x, center_y),
            0,  # no rotation
            scale
        )
        
        # Adjust translation to center on focus point
        transform[0, 2] += shape[1] / 2 - center_x
        transform[1, 2] += shape[0] / 2 - center_y
        
        return transform

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between two values"""
        return a + (b - a) * t

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Smooth easing function for transitions"""
        return t * t * (3 - 2 * t)

# Example usage
async def main():
    processor = SmartScreenFocus()
    await processor.process_video(
        "input_screen_recording.mp4",
        "output_dynamic_focus.mp4"
    )

if __name__ == "__main__":
    asyncio.run(main())