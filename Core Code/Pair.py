import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import torch
import torch.nn.functional as F

# Process a video folder to extract key frames and compute optical flow
def process_video_folder(args):
    # Re-import necessary modules in the subprocess
    import os
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    input_folder, output_folder, label, ID, device = args

    # Get the list of frame images, sorted by filename
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))])
    if len(frame_files) == 0:
        print(f"No frame images found in {input_folder}.")
        return []

    # Read frame images
    frames = []
    frame_indices = []
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        frames.append(frame)
        frame_indices.append(frame_file)

    # Ensure there are enough frames
    if len(frames) < 2:
        print(f"Insufficient frames in {input_folder}.")
        return []

    # Convert frames to grayscale and then to tensors
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    gray_tensors = [torch.from_numpy(f).float().to(device) / 255.0 for f in gray_frames]

    # Compute optical flow magnitude between consecutive frames
    flow_magnitudes = []
    for i in range(len(gray_tensors)-1):
        prev = gray_tensors[i].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        next = gray_tensors[i+1].unsqueeze(0).unsqueeze(0)
        flow = compute_optical_flow(prev, next, device)
        magnitude = torch.norm(flow, dim=1)
        mean_magnitude = magnitude.mean().item()
        flow_magnitudes.append(mean_magnitude)

    # Determine indices for onset, apex, and offset frames
    onset_index = 0
    offset_index = len(frames) - 1
    cum_flow = np.cumsum(flow_magnitudes)
    apex_index = np.argmax(cum_flow) + 1  # Because flow_magnitudes starts from index 1

    # Compute anchor optical flow (between apex and onset)
    onset_tensor = gray_tensors[onset_index].unsqueeze(0).unsqueeze(0)
    apex_tensor = gray_tensors[apex_index].unsqueeze(0).unsqueeze(0)
    anchor_flow = compute_optical_flow(onset_tensor, apex_tensor, device)
    anchor_magnitude = torch.norm(anchor_flow, dim=1)

    # Compute optical flow between other frames and onset frame, and compare with anchor
    candidate_indices = list(range(onset_index+1, offset_index+1))
    if apex_index in candidate_indices:
        candidate_indices.remove(apex_index)

    similarities = []
    for idx in candidate_indices:
        frame_tensor = gray_tensors[idx].unsqueeze(0).unsqueeze(0)
        flow = compute_optical_flow(onset_tensor, frame_tensor, device)
        magnitude = torch.norm(flow, dim=1)

        # Compute mean squared error (MSE) between optical flow magnitudes as a similarity measure
        mse = F.mse_loss(anchor_magnitude, magnitude).item()
        sim = -mse
        similarities.append((sim, idx))

    # Sort by similarity and select the top 5 frames
    similarities.sort(reverse=True, key=lambda x: x[0])
    top5_frames = [idx for _, idx in similarities[:5]]

    # Save onset, apex, and top 5 frames to the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Save onset frame, renamed as onset.png
    onset_frame = frames[onset_index]
    onset_output_path = os.path.join(output_folder, 'onset.png')
    cv2.imwrite(onset_output_path, onset_frame)

    # Save apex frame, renamed as apex.png
    apex_frame = frames[apex_index]
    apex_output_path = os.path.join(output_folder, 'apex.png')
    cv2.imwrite(apex_output_path, apex_frame)

    # Save top 5 frames
    csv_entries = []
    for idx in top5_frames:
        frame = frames[idx]
        frame_name = frame_indices[idx]
        output_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_path, frame)
        # Build CSV entry
        csv_entries.append([onset_output_path, output_path, label, ID])

    # Include apex frame in CSV entries
    csv_entries.append([onset_output_path, apex_output_path, label, ID])

    return csv_entries

# Compute optical flow using PyTorch
def compute_optical_flow(prev, next, device):
    """
    Compute optical flow using PyTorch.
    """
    # Use CPU-based optical flow calculation as OpenCV CUDA optical flow is unstable in multiprocessing
    prev_np = (prev.squeeze().cpu().numpy() * 255).astype(np.uint8)
    next_np = (next.squeeze().cpu().numpy() * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev_np, next_np, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(device)
    return flow

# Process all video folders in the dataset
def process_dataset(input_root, output_root, csv_filename, device):
    """
    Process all video folders in the dataset.
    """
    csv_entries = []

    # Collect all video folder information
    video_info_list = []
    for ID in os.listdir(input_root):
        ID_path = os.path.join(input_root, ID)
        if not os.path.isdir(ID_path):
            continue
        for label in os.listdir(ID_path):
            label_path = os.path.join(ID_path, label)
            if not os.path.isdir(label_path):
                continue
            for video_folder in os.listdir(label_path):
                video_input_path = os.path.join(label_path, video_folder)
                if not os.path.isdir(video_input_path):
                    continue
                # Define output path
                video_output_path = os.path.join(output_root, ID, label, video_folder)
                video_info_list.append((video_input_path, video_output_path, label, ID, device))

    # Use multiprocessing to process videos
    num_processes = min(multiprocessing.cpu_count(), 8)  # Adjust the number of processes based on resources
    with Pool(processes=num_processes) as pool:
        results = []
        for entries in tqdm(pool.imap_unordered(process_video_folder, video_info_list), total=len(video_info_list), desc=f"Processing dataset {input_root}"):
            csv_entries.extend(entries)

    # Save CSV file
    csv_df = pd.DataFrame(csv_entries, columns=['onset_frame', 'pseudo_apex_frame', 'label', 'ID'])
    csv_df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Set the computation device
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Process training data
    process_dataset('./BDI-II/CroppedTrain2013', './BDI-II/oaPair_train2013', 'BDI-II_train2013.csv', device)

    # Process test data 2013
    process_dataset('./BDI-II/CroppedTest2013', './BDI-II/oaPair_2013', 'BDI-II_2013test.csv', device)

    # Process test data 2014
    # process_dataset('./BDI-II/CroppedTest2014', './BDI-II/oaPair_2014', 'BDI-II_2014test.csv', device)

    print("All datasets processed!")