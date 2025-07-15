import cv2
import os



def video_to_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Save frame as image
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        
        
        
        cv2.imwrite(frame_filename, frame)
        
        print(f"Saved {frame_filename}")
        frame_count += 1

    cap.release()
    print(f"Done. Total frames saved: {frame_count}")
    
def video_to_frame(video_path):
    # Create output directory if it doesn't exist
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0

    ret, frame = cap.read()
    
    print("frame shape", frame.shape)
    print("frame 1", frame[7][5][2])

    cap.release()

# Example usage
# video_to_frames("your_video.mp4", "output_frames")

if __name__ == "__main__":
    # video_path = "color_output.avi"  # Replace with your video file path
    video_path = "success/color_1_env_0_ep_1.avi"  # Replace with your video file path
    output_dir = "color_1_env_0_ep_1"
    # video_to_frame(video_path)
    video_to_frames(video_path, output_dir)
    