import cv2


def read_video(video_path, max_frames=0):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if max_frames > 0 and count >= max_frames:
            break
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path, fourcc, 24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()