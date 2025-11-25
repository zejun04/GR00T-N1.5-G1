#!/usr/bin/env python3
"""
GR00T Camera Data Visualizer - Fixed Dimension Order

This script fixes the dimension order issue where server expects (640, 480) but we were sending (480, 640).

Usage:
    python camera_visualizer_fixed.py --host localhost --port 8000
"""

import argparse
import numpy as np
import cv2
import requests
import json_numpy
import time

# Patch json to handle numpy arrays
json_numpy.patch()


class FixedCameraVisualizer:
    def __init__(self, host: str, port: int, api_token: str = None):
        self.host = host
        self.port = port
        self.api_token = api_token
        self.base_url = f"http://{host}:{port}"
        
        # Correct dimensions: (width, height) = (640, 480)
        self.expected_width = 640
        self.expected_height = 480
        self.expected_shape = (1, self.expected_height, self.expected_width, 3)  # Batch, Height, Width, Channels
        
        self.camera_configs = [
            ("video.cam_left_high", "Left High Camera"),
            ("video.cam_left_wrist", "Left Wrist Camera"),
            ("video.cam_right_wrist", "Right Wrist Camera")
        ]
        
        # Create display windows
        for cam_key, cam_name in self.camera_configs:
            cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam_name, self.expected_width, self.expected_height)

    def create_correct_observation(self):
        """
        Create observation with correct dimension order.
        Shape should be: (batch_size, height, width, channels) = (1, 480, 640, 3)
        """
        observation = {
            "state.left_arm": np.zeros((1, 7), dtype=np.float32),
            "state.right_arm": np.zeros((1, 7), dtype=np.float32),
            "state.left_hand": np.zeros((1, 6), dtype=np.float32),
            "state.right_hand": np.zeros((1, 6), dtype=np.float32), 
            "state.waist": np.zeros((1, 3), dtype=np.float32),
            "annotation.human.action.task_description": ["visualize cameras"]
        }
        
        # Add camera data with CORRECT dimension order
        for cam_key, cam_name in self.camera_configs:
            # Create image with correct shape: (1, 480, 640, 3)
            # This means: 1 image, 480px height, 640px width, 3 color channels
            observation[cam_key] = self.create_test_image()
        
        return observation

    def create_test_image(self):
        """Create a test image with correct dimensions"""
        # Shape: (1, height, width, channels) = (1, 480, 640, 3)
        img = np.zeros((self.expected_height, self.expected_width, 3), dtype=np.uint8)
        
        # Create a more interesting test pattern
        # Gradient background
        for i in range(self.expected_height):
            for j in range(self.expected_width):
                img[i, j, 0] = int(255 * j / self.expected_width)  # Red
                img[i, j, 1] = int(255 * i / self.expected_height)  # Green
                img[i, j, 2] = 128  # Blue
        
        # Add some shapes
        cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
        cv2.rectangle(img, (150, 150), (250, 200), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(img, f"{self.expected_width}x{self.expected_height}", 
                   (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add batch dimension
        return img[np.newaxis, ...]

    def get_action_from_server(self, observation):
        """Send observation to server and get response"""
        try:
            headers = {}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            
            payload = {"observation": observation}
            
            print(f"Sending observation with shapes:")
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape}")
            
            response = requests.post(
                f"{self.base_url}/act",
                json=payload,
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                print("✓ Server request successful")
                return response.json()
            else:
                print(f"✗ Server error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"✗ Request error: {e}")
            return None

    def display_observation_cameras(self, observation):
        """Display camera frames from the observation we sent"""
        frames_displayed = 0
        
        for cam_key, cam_name in self.camera_configs:
            if cam_key in observation:
                frame_data = observation[cam_key]
                print(f"Displaying {cam_key} with shape: {frame_data.shape}")
                
                frame = self.prepare_frame_for_display(frame_data)
                if frame is not None:
                    # Add camera name and info
                    cv2.putText(
                        frame, f"{cam_name} - Sent to Server", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, f"Shape: {frame_data.shape}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
                    
                    cv2.imshow(cam_name, frame)
                    frames_displayed += 1
            else:
                print(f"Camera {cam_key} not in observation")
                placeholder = self.create_placeholder(f"{cam_name} - No Data")
                cv2.imshow(cam_name, placeholder)
        
        return frames_displayed

    def prepare_frame_for_display(self, frame_data):
        """Prepare a single frame for OpenCV display"""
        if frame_data.ndim == 4:  # Batch dimension
            frame = frame_data[0]  # Take first item in batch
        else:
            frame = frame_data
        
        # Convert to BGR for OpenCV
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame

    def create_placeholder(self, message):
        """Create placeholder image"""
        img = np.zeros((self.expected_height, self.expected_width, 3), dtype=np.uint8)
        cv2.putText(
            img, message, 
            (50, self.expected_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        return img

    def run(self):
        """Main visualization loop"""
        print("=" * 60)
        print("GR00T Camera Visualizer - Fixed Dimension Order")
        print("=" * 60)
        print(f"Expected image shape: {self.expected_shape}")
        print(f"  - Batch size: 1")
        print(f"  - Height: {self.expected_height}px")
        print(f"  - Width: {self.expected_width}px") 
        print(f"  - Channels: 3 (RGB)")
        print("Controls: 'q'=quit, 'r'=refresh, 'n'=new test image")
        print("=" * 60)
        
        current_observation = self.create_correct_observation()
        request_count = 0
        
        try:
            while True:
                # Send to server periodically
                if request_count % 5 == 0:  # Every 5 iterations
                    print(f"\n--- Request #{request_count} ---")
                    response = self.get_action_from_server(current_observation)
                    if response:
                        print(f"Server returned {len(response)} action components")
                
                # Display what we're sending to the server
                frames_displayed = self.display_observation_cameras(current_observation)
                
                status = f"Requests: {request_count} | Displaying: {frames_displayed} cameras | 'q' to quit"
                print(status, end='\r')
                
                # Handle key presses
                key = cv2.waitKey(100) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("\nRefreshing server request...")
                    response = self.get_action_from_server(current_observation)
                elif key == ord('n'):
                    print("\nGenerating new test images...")
                    current_observation = self.create_correct_observation()
                
                request_count += 1
                
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        finally:
            cv2.destroyAllWindows()
            print("\nVisualization stopped")


def main():
    parser = argparse.ArgumentParser(description="GR00T Camera Visualizer with Fixed Dimensions")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--api-token", type=str, default=None, help="API token")
    
    args = parser.parse_args()
    
    print(f"Connecting to: {args.host}:{args.port}")
    visualizer = FixedCameraVisualizer(args.host, args.port, args.api_token)
    visualizer.run()


if __name__ == "__main__":
    main()