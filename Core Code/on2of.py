import os
import cv2
import numpy as np

# Create the output path to store results
output_path = './onset2offset2014'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Calculate optical flow from a video
def calculate_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the video is successfully opened
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, None

    # Read the first frame
    ret, previous_frame = cap.read()
    if not ret:
        print(f"Failed to read the first frame of the video: {video_path}")
        return None, None

    # Convert the frame to grayscale
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    flow_magnitudes = []
    frames = [previous_frame]

    # Process each frame in the video
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        frames.append(current_frame)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Save the average magnitude of the optical flow
        flow_magnitudes.append(np.mean(magnitude))

        previous_gray = current_gray

    cap.release()
    return frames, flow_magnitudes

# Detect key frames (onset, apex, offset) based on optical flow magnitudes
def detect_key_frames(flow_magnitudes):
    if not flow_magnitudes:
        return None, None, None

    max_magnitude = max(flow_magnitudes)
    # TODO: Modify this parameter to explore fully
    threshold = max_magnitude * 0.3  # Threshold for significant change

    onset_frame = None
    apex_frame = None
    offset_frame = None

    # Detect onset frame (first significant change)
    for i, magnitude in enumerate(flow_magnitudes):
        if magnitude > threshold:
            onset_frame = i
            break

    # Detect apex frame (maximum change)
    apex_frame = flow_magnitudes.index(max_magnitude)

    # Detect offset frame (last significant change)
    for i in range(len(flow_magnitudes)-1, 0, -1):
        if flow_magnitudes[i] > threshold:
            offset_frame = i
            break

    return onset_frame, apex_frame, offset_frame

# Process video and save frames from onset to offset
def process_video(video_path, output_base_path):
    frames, flow_magnitudes = calculate_optical_flow(video_path)
    if frames is None or flow_magnitudes is None:
        print(f"Skipping processing: {video_path}")
        return

    onset_frame, apex_frame, offset_frame = detect_key_frames(flow_magnitudes)
    if onset_frame is None or apex_frame is None or offset_frame is None:
        print(f"Failed to detect suitable onset, apex, or offset frames in video: {video_path}")
        return

    # Create a folder for the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_base_path, video_name)
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path)

    # Save frames from onset to offset
    for i in range(onset_frame, offset_frame + 1):
        frame = frames[i]
        frame_filename = f"{i}.png"
        if i == onset_frame:
            frame_filename = "onset.png"
        elif i == apex_frame:
            frame_filename = "apex.png"
        elif i == offset_frame:
            frame_filename = "offset.png"
        
        frame_path = os.path.join(video_output_path, frame_filename)
        cv2.imwrite(frame_path, frame)

    print(f"Processing complete: {video_name}, onset: {onset_frame}, apex: {apex_frame}, offset: {offset_frame}")

# Traverse the input directory and process each video
input_base_path = './little_2014'
for root, dirs, files in os.walk(input_base_path):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_base_path)
            output_folder_path = os.path.join(output_path, relative_path)

            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # Process each video
            process_video(video_path, output_folder_path)

print("All videos processed!")