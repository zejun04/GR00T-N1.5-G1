#!/usr/bin/env python3
"""
GR00T Observation Visualizer

This script connects to the GR00T HTTP server and visualizes the observation images
that are being sent from Isaac Lab simulation environment.

Usage:
    python observation_visualizer.py --host localhost --port 8000
"""

import argparse
import numpy as np
import cv2
import requests
import json_numpy
import time

# Patch json to handle numpy arrays
json_numpy.patch()


class ObservationVisualizer:
    def __init__(self, host: str, port: int, api_token: str = None):
        self.host = host
        self.port = port
        self.api_token = api_token
        self.base_url = f"http://{host}:{port}"
        
        # Camera observation keys we want to visualize
        self.camera_keys = [
            "video.cam_left_high",
            "video.cam_left_wrist", 
            "video.cam_right_wrist"
        ]
        
        # Create display windows
        for cam_key in self.camera_keys:
            # Create a readable window name from the camera key
            window_name = cam_key.replace("video.", "").replace("_", " ").title()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)

    def get_latest_observation(self):
        """
        Get the latest observation from the server by sending a dummy request.
        The server should return the current observation state.
        """
        try:
            # Send a minimal observation to trigger the server to process current state
            dummy_observation = {
                "state.left_arm": np.zeros((1, 7), dtype=np.float32),
                "state.right_arm": np.zeros((1, 7), dtype=np.float32),
                "state.left_hand": np.zeros((1, 6), dtype=np.float32),
                "state.right_hand": np.zeros((1, 6), dtype=np.float32),
                "state.waist": np.zeros((1, 3), dtype=np.float32),
                "annotation.human.action.task_description": ["get observation"]
            }
            
            headers = {}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            
            # The server's /act endpoint expects an observation and returns an action
            # We'll use this to get the current observation state
            response = requests.post(
                f"{self.base_url}/act",
                json={"observation": dummy_observation},
                headers=headers,
                timeout=5.0
            )
            
            if response.status_code == 200:
                # The response contains the action, but we need the observation
                # We'll need to modify the server to return observation or use a different approach
                # For now, we'll assume the server stores the latest observation
                return dummy_observation  # This is a placeholder
            else:
                print(f"Server error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Request error: {e}")
            return None

    def visualize_observation(self, observation):
        """Visualize camera observations"""
        if not observation:
            return 0
            
        frames_displayed = 0
        for cam_key in self.camera_keys:
            if cam_key in observation:
                frame_data = observation[cam_key]
                
                # Process the frame for display
                frame = self.process_frame(frame_data)
                if frame is not None:
                    # Create a readable window name
                    window_name = cam_key.replace("video.", "").replace("_", " ").title()
                    
                    # Add timestamp and camera name
                    timestamp = time.strftime("%H:%M:%S")
                    cv2.putText(frame, f"{window_name} - {timestamp}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow(window_name, frame)
                    frames_displayed += 1
                else:
                    print(f"Failed to process frame for {cam_key}")
            else:
                print(f"Camera {cam_key} not found in observation")
                
        return frames_displayed

    def process_frame(self, frame_data):
        """Process a single frame for display"""
        if not isinstance(frame_data, np.ndarray):
            return None
            
        # Handle different frame formats
        if frame_data.ndim == 4:  # Batch dimension (1, H, W, C)
            frame = frame_data[0]  # Take first frame in batch
        elif frame_data.ndim == 3:  # Single frame (H, W, C)
            frame = frame_data
        else:
            print(f"Unexpected frame dimensions: {frame_data.ndim}")
            return None
        
        # Ensure frame is in correct format
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:  # Normalized [0,1]
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV if needed
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize for consistent display
        frame = cv2.resize(frame, (640, 480))
        
        return frame

    def create_test_observation(self):
        """Create a test observation for visualization when no real data is available"""
        # This creates synthetic camera data for testing
        observation = {}
        
        for cam_key in self.camera_keys:
            # Create a test pattern for each camera
            height, width = 480, 640
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Different patterns for different cameras
            if "left_high" in cam_key:
                # Gradient pattern for left high camera
                for i in range(height):
                    for j in range(width):
                        img[i, j, 0] = int(255 * j / width)  # Red
                        img[i, j, 1] = int(255 * i / height)  # Green
                        img[i, j, 2] = 128  # Blue
            elif "left_wrist" in cam_key:
                # Circle pattern for left wrist camera
                cv2.circle(img, (width//2, height//2), 100, (0, 255, 0), -1)
            elif "right_wrist" in cam_key:
                # Rectangle pattern for right wrist camera
                cv2.rectangle(img, (100, 100), (width-100, height-100), (255, 0, 0), -1)
            
            # Add camera label
            cam_name = cam_key.replace("video.", "").replace("_", " ").title()
            cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add batch dimension
            observation[cam_key] = img[np.newaxis, ...]
        
        return observation

    def run(self):
        """Main visualization loop"""
        print("GR00T Observation Visualizer")
        print("============================")
        print("This script visualizes the observation images sent from Isaac Lab")
        print("to the GR00T model inference server.")
        print()
        print("Controls:")
        print("  'q' - Quit")
        print("  't' - Toggle between test data and server data")
        print("  'r' - Refresh")
        print("============================")
        
        use_test_data = False
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Get observation data
                if use_test_data:
                    observation = self.create_test_observation()
                    source = "TEST DATA"
                else:
                    observation = self.get_latest_observation()
                    source = "SERVER"
                
                # Visualize the observation
                frames_displayed = self.visualize_observation(observation)
                
                # Calculate and display status
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                status = f"Source: {source} | FPS: {fps:.1f} | Cameras: {frames_displayed}/{len(self.camera_keys)} | Frame: {frame_count}"
                print(status, end='\r')
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    use_test_data = not use_test_data
                    print(f"\nSwitched to {'TEST DATA' if use_test_data else 'SERVER DATA'}")
                elif key == ord('r'):
                    print("\nManual refresh")
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            cv2.destroyAllWindows()
            print(f"\nVisualization ended. Total frames: {frame_count}")


def main():
    parser = argparse.ArgumentParser(description="GR00T Observation Visualizer")
    parser.add_argument("--host", type=str, default="localhost", help="GR00T server host")
    parser.add_argument("--port", type=int, default=8000, help="GR00T server port")
    parser.add_argument("--api-token", type=str, default=None, help="API token for authentication")
    
    args = parser.parse_args()
    
    print(f"Connecting to GR00T server at {args.host}:{args.port}")
    visualizer = ObservationVisualizer(args.host, args.port, args.api_token)
    visualizer.run()


if __name__ == "__main__":
    main()