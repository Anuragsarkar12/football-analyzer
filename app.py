import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from utils import read_video


def save_video_mp4(frames, output_path, fps=24):
    """Save frames as H.264 MP4 (browser-compatible)."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

    # Re-encode with ffmpeg for browser compatibility
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        final_path = output_path.replace('.mp4', '_final.mp4')
        os.system(
            f'{ffmpeg_path} -y -i {output_path} -c:v libx264 -preset fast '
            f'-crf 23 -movflags +faststart {final_path} 2>/dev/null'
        )
        if os.path.exists(final_path):
            os.replace(final_path, output_path)
    except Exception:
        pass  # Fall back to mp4v if ffmpeg unavailable

@st.cache_resource
def load_tracker(model_path='models/best.pt'):
    """Cache the YOLO model so it's only loaded once."""
    return Tracker(model_path)


def process_video(video_path, progress_bar, status_text):
    """Run the full analysis pipeline on an uploaded video."""

    # Step 1: Read video
    status_text.text("📖 Reading video frames...")
    progress_bar.progress(5)
    video_frames = read_video(video_path)
    total_frames = len(video_frames)

    if total_frames == 0:
        st.error("Could not read any frames from the video. Check the format.")
        return None

    st.info(f"Loaded {total_frames} frames from video.")

    # Step 2: Object detection & tracking
    status_text.text("🔍 Running YOLO detection & tracking (this is the slow part)...")
    progress_bar.progress(10)
    tracker = load_tracker()
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
    tracker.add_position_to_tracks(tracks)

    progress_bar.progress(40)

    # Step 3: Camera movement estimation
    status_text.text("📷 Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=False
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    progress_bar.progress(55)

    # Step 4: View transformation
    status_text.text("🏟️ Applying perspective transform...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Step 5: Ball interpolation
    status_text.text("⚽ Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    progress_bar.progress(65)

    # Step 6: Speed & distance
    status_text.text("🏃 Computing speed & distance...")
    speed_distance_estimator = SpeedAndDistance_Estimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    progress_bar.progress(70)

    # Step 7: Team assignment
    status_text.text("👕 Assigning teams by jersey color...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = \
                team_assigner.team_colors[team]

    progress_bar.progress(80)

    # Step 8: Ball possession
    status_text.text("🎯 Calculating ball possession...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team']
            )
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)

    team_ball_control = np.array(team_ball_control)

    progress_bar.progress(85)

    # Step 9: Draw annotations
    status_text.text("🎨 Drawing annotations on frames...")
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = camera_movement_estimator.draw_camera_movement(
        output_frames, camera_movement_per_frame
    )
    speed_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    progress_bar.progress(95)

    # Step 10: Encode output video
    status_text.text("💾 Encoding output video (H.264 MP4)...")
    output_path = tempfile.mktemp(suffix='.mp4')
    save_video_mp4(output_frames, output_path)

    progress_bar.progress(100)
    status_text.text("✅ Processing complete!")

    return output_path


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Football Match Analyzer",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Match Video Analyzer")
st.markdown(
    "Upload a football match video clip and get back an annotated version with "
    "**player tracking, team identification, ball possession, speed & distance metrics, "
    "and camera movement compensation.**"
)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    st.markdown(
        "**Model:** YOLOv8 (custom-trained)\n\n"
        "**Tracker:** ByteTrack\n\n"
        "**Team detection:** KMeans jersey clustering"
    )
    st.markdown("---")
    st.warning(
        "⚠️ **Processing time depends on video length.**\n\n"
        "A 10-second clip (~240 frames at 24fps) takes roughly 2–5 minutes "
        "depending on hardware. GPU recommended."
    )
    max_frames = st.number_input(
        "Max frames to process (0 = all)",
        min_value=0, max_value=5000, value=0, step=100,
        help="Limit frame count to speed up processing during testing."
    )

# File uploader
uploaded_file = st.file_uploader(
    "Upload a football match video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Supported: MP4, AVI, MOV, MKV. Keep clips under 30 seconds for reasonable processing time."
)

if uploaded_file is not None:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Show uploaded video preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Input Video")
        st.video(tmp_path)

    # Process button
    if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
        with col2:
            st.subheader("📤 Output Video")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                output_path = process_video(tmp_path, progress_bar, status_text)

                if output_path and os.path.exists(output_path):
                    st.video(output_path)

                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Analyzed Video",
                            data=f.read(),
                            file_name="analyzed_football.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

                    # Cleanup
                    os.unlink(output_path)
                else:
                    st.error("Processing failed — no output was generated.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

    # Cleanup temp input
    # (Don't delete immediately — user might re-process)

else:
    # Show placeholder
    st.markdown("---")
    st.markdown(
        "### How it works\n\n"
        "1. **Upload** a short football match clip (10–30 seconds recommended)\n"
        "2. **Click Analyze** and wait for the pipeline to process\n"
        "3. **Download** the annotated video with all overlays\n\n"
        "The pipeline detects and tracks every player, referee, and the ball. "
        "It identifies teams by jersey color, estimates camera movement, "
        "transforms coordinates to real-world pitch positions, and computes "
        "per-player speed and distance covered."
    )