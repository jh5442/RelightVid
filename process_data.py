import cv2
import os




def process_video_and_mask(video_path, mask_video_path, save_mask_folder):
    # Create target folder if it doesn't exist
    os.makedirs(save_mask_folder, exist_ok=True)

    # Open main video
    cap_video = cv2.VideoCapture(video_path)
    if not cap_video.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Get frame count from main video
    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_video.release()

    # Open mask video
    cap_mask = cv2.VideoCapture(mask_video_path)
    if not cap_mask.isOpened():
        raise IOError(f"Cannot open mask video: {mask_video_path}")

    # Read and save the same number of frames as main video
    for i in range(total_frames):
        ret, frame = cap_mask.read()
        if not ret:
            print(f"Mask video ended at frame {i}, expected {total_frames} frames.")
            break
        # Save frame as PNG with zero-padded index
        filename = f"frame_{i:05d}.png"
        save_path = os.path.join(save_mask_folder, filename)
        cv2.imwrite(save_path, frame)

    cap_mask.release()
    print(f"Saved {min(total_frames, i+1)} mask frames to {save_mask_folder}")




def reshape_lighting_video(reference_video_path, original_video_path, save_video_path):
    # Open reference video to get width and height
    cap_ref = cv2.VideoCapture(reference_video_path)
    if not cap_ref.isOpened():
        raise IOError(f"Cannot open reference video: {reference_video_path}")

    ref_width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_ref.release()

    # Open original video
    cap_orig = cv2.VideoCapture(original_video_path)
    if not cap_orig.isOpened():
        raise IOError(f"Cannot open original video: {original_video_path}")

    # Get FPS from original video to preserve timing
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (ref_width, ref_height))

    # Read original video frames and resize
    while True:
        ret, frame = cap_orig.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR)
        out.write(resized_frame)

    cap_orig.release()
    out.release()
    print(f"Reshaped video saved to: {save_video_path}")



def trim_video(video_path, save_video_path, keep_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < keep_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Trimmed video saved to {save_video_path} with {frame_count} frames.")


if __name__ == "__main__":
    trim_video(video_path="/home/ubuntu/jin/data/test_03_and_04/test_03_5s.mp4",
               save_video_path="/home/ubuntu/jin/data/test_03_and_04/test_03_16_frames.mp4")

    trim_video(video_path="/home/ubuntu/jin/data/test_03_and_04/test_04_5s.mp4",
               save_video_path="/home/ubuntu/jin/data/test_03_and_04/test_04_16_frames.mp4")

    # process_video_and_mask(video_path="/home/ubuntu/jin/data/test_03_and_04/test_03_5s.mp4",
    #                        mask_video_path="/home/ubuntu/jin/data/test_03_and_04/test_03_mask.mp4",
    #                        save_mask_folder="/home/ubuntu/jin/data/test_03_and_04/test_03_5s_mask_frame")
    #
    # process_video_and_mask(video_path="/home/ubuntu/jin/data/test_03_and_04/test_04_5s.mp4",
    #                        mask_video_path="/home/ubuntu/jin/data/test_03_and_04/test_04_mask.mp4",
    #                        save_mask_folder="/home/ubuntu/jin/data/test_03_and_04/test_04_5s_mask_frame")
    #
    # reshape_lighting_video(reference_video_path="/home/ubuntu/jin/data/test_03_and_04/test_03_5s.mp4",
    #                        original_video_path="/home/ubuntu/jin/code/RelightVid/assets/video_bg/stage_light2.mp4",
    #                        save_video_path="/home/ubuntu/jin/code/RelightVid/assets/video_bg/test_03_stage_light2.mp4")
    #
    # reshape_lighting_video(reference_video_path="/home/ubuntu/jin/data/test_03_and_04/test_04_5s.mp4",
    #                        original_video_path="/home/ubuntu/jin/code/RelightVid/assets/video_bg/stage_light2.mp4",
    #                        save_video_path="/home/ubuntu/jin/code/RelightVid/assets/video_bg/test_04_stage_light2.mp4")
