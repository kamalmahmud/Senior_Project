import cv2
import numpy as np


def cut_clips(video_path, clip_len, stride):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        all_frames.append(gray)

    cap.release()

    if len(all_frames) < clip_len:
        raise ValueError(
            f"Video has {len(all_frames)} frames, "
            f"which is less than clip_len={clip_len}"
        )

    clips = []
    for start in range(0, total_frames - clip_len + 1, stride):
        clip = all_frames[start: start + clip_len]
        clips.append(np.array(clip))
    return clips
