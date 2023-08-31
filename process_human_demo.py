import os
import numpy as np
import cv2
# Define the path to the raw human demo video
human_demo_video_path = '/home/thomastian/workspace/mvp_exp_data/human_demo/raw_videos/negative_demo/'
processed_demo_save_path = '/home/thomastian/workspace/mvp_exp_data/human_demo/negative_demo/'
# Count the total number of demos
total_demos = len(os.listdir(human_demo_video_path))
# Define the image size
image_size = 224

# Loof through all the demos
for demo_id in range(total_demos):
    # Define the path to the current demo
    current_demo_path = os.path.join(human_demo_video_path, 'negative_demo_'+str(demo_id+1)+'.mp4')
    # Load all frames in the current demo and uniformly sample 45 frames
    cap = cv2.VideoCapture(current_demo_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Define the sampling interval
    sampling_interval = int(total_frames/45)
    # Define the list to store all the sampled frames
    sampled_frames = []
    # Loop through all the frames
    for frame_id in range(total_frames):
        # Extract the current frame
        ret, frame = cap.read()
        # If the current frame is sampled, then resize it and store it
        if frame_id % sampling_interval == 0:
            # Convert the data type to float32
            frame = frame.astype(np.float32)
            frame = frame[0:0+700, 200:200+700, :]
            frame = cv2.resize(frame, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
            sampled_frames.append(frame)
    # Stack all the sampled frames
    sampled_frames = np.stack(sampled_frames, axis=0)
    # Save the sampled frames
    for frame_id in range(45):
        image_save_path = os.path.join(processed_demo_save_path, str(demo_id))
        # Create the folder if it does not exist
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        image_save_path = os.path.join(image_save_path, str(frame_id)+'.png')
        # save the image to the path
        cv2.imwrite(image_save_path, sampled_frames[frame_id])