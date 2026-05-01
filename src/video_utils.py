import os
import tempfile

import cv2
from PIL import Image

from src.config import MAX_IMAGE_EDGE, VIDEO_SAMPLE_FRAMES


def sample_video_frames(video_source, frame_count: int = VIDEO_SAMPLE_FRAMES) -> list[Image.Image]:
    """Sample representative RGB frames from an uploaded video file."""
    video_path = _write_temp_video(video_source)

    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Could not open this video.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Could not read frames from this video.")

        frame_indexes = _uniform_frame_indexes(total_frames, frame_count)
        frames = []

        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
            frames.append(image)

        if not frames:
            raise ValueError("No readable frames were found in this video.")

        return frames
    finally:
        capture.release()
        os.remove(video_path)


def summarize_frame_captions(frame_captions: list[str]) -> str:
    """Create a readable video-level caption from sampled frame captions."""
    unique_captions = []
    for caption in frame_captions:
        if caption and caption not in unique_captions:
            unique_captions.append(caption)

    if len(unique_captions) == 1:
        return f"Video summary: {unique_captions[0]}"

    joined_captions = "; then ".join(unique_captions)
    return f"Video summary from sampled frames: {joined_captions}"


def _uniform_frame_indexes(total_frames: int, frame_count: int) -> list[int]:
    if frame_count <= 1 or total_frames == 1:
        return [0]

    usable_count = min(frame_count, total_frames)
    last_index = total_frames - 1
    return [
        round(index * last_index / (usable_count - 1))
        for index in range(usable_count)
    ]


def _write_temp_video(video_source) -> str:
    suffix = os.path.splitext(getattr(video_source, "name", ""))[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(video_source.getbuffer())
        return temp_file.name
