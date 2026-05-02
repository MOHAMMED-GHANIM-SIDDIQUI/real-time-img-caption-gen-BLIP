import streamlit as st

from src.audio import create_caption_audio
from src.captioning import load_captioner
from src.config import MAX_VIDEO_SIZE_MB, VIDEO_SAMPLE_FRAMES
from src.image_utils import load_image
from src.video_utils import sample_video_frames, summarize_frame_captions

try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = ""
except ImportError as error:
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = str(error)


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}
MAX_IMAGE_SIZE_MB = 10
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ],
}


st.set_page_config(
    page_title="AI Caption Studio",
    page_icon="🎥",
    layout="wide",
)


# ---------------------------------------------------------------------------
# CSS Section
# ---------------------------------------------------------------------------
def inject_global_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --bg-1: #060816;
            --bg-2: #11103a;
            --bg-3: #35135f;
            --primary: #8b5cf6;
            --primary-2: #6366f1;
            --cyan: #22d3ee;
            --pink: #ec4899;
            --text: #f8fbff;
            --muted: rgba(226, 232, 240, 0.72);
            --glass: rgba(255, 255, 255, 0.085);
            --glass-strong: rgba(255, 255, 255, 0.13);
            --border: rgba(255, 255, 255, 0.18);
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 12% 12%, rgba(34, 211, 238, 0.20), transparent 28%),
                radial-gradient(circle at 86% 8%, rgba(236, 72, 153, 0.20), transparent 30%),
                linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(5, 8, 22, 0.72);
            border-right: 1px solid var(--border);
            backdrop-filter: blur(20px);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }

        .hero {
            text-align: center;
            padding: 2.8rem 1.2rem 2.1rem;
            animation: fadeUp 0.7s ease both;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 1rem;
            padding: 0.46rem 0.82rem;
            border-radius: 999px;
            border: 1px solid rgba(34, 211, 238, 0.34);
            background: rgba(34, 211, 238, 0.09);
            color: #b8f3ff;
            font-size: 0.84rem;
            font-weight: 700;
            box-shadow: 0 0 26px rgba(34, 211, 238, 0.10);
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(3rem, 8vw, 6.7rem);
            line-height: 0.92;
            letter-spacing: 0;
            font-weight: 900;
            background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 34%, #67e8f9 65%, #f9a8d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero p {
            max-width: 760px;
            margin: 1.05rem auto 0;
            color: var(--muted);
            font-size: clamp(1.02rem, 2vw, 1.32rem);
            line-height: 1.75;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1.8rem auto 0;
            max-width: 820px;
        }

        .metric-pill {
            padding: 0.8rem 1rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
            color: rgba(248, 251, 255, 0.9);
            font-weight: 700;
        }

        .glass-card {
            position: relative;
            margin: 1.05rem 0;
            padding: 1.15rem;
            border: 1px solid var(--border);
            border-radius: 28px;
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.055));
            box-shadow: var(--shadow);
            backdrop-filter: blur(24px);
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
            animation: fadeUp 0.55s ease both;
            overflow: hidden;
        }

        .glass-card::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            border-radius: inherit;
            background: linear-gradient(135deg, rgba(255,255,255,0.22), transparent 36%, rgba(34,211,238,0.11));
            opacity: 0.65;
        }

        .glass-card:hover {
            transform: translateY(-3px);
            border-color: rgba(103, 232, 249, 0.42);
            box-shadow: 0 28px 90px rgba(0, 0, 0, 0.50), 0 0 38px rgba(99, 102, 241, 0.16);
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0 0 0.35rem;
            font-size: 1.38rem;
            font-weight: 850;
            color: var(--text);
        }

        .section-subtitle {
            margin: 0 0 0.9rem;
            color: var(--muted);
            line-height: 1.55;
        }

        .sidebar-panel {
            margin: 0.75rem 0 1rem;
            padding: 1rem;
            border-radius: 22px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.075);
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
        }

        .sidebar-title {
            margin: 0 0 0.7rem;
            font-weight: 850;
            font-size: 1rem;
            color: #ffffff;
        }

        .limit-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.58rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--muted);
            font-size: 0.9rem;
        }

        .limit-row:last-child {
            border-bottom: 0;
        }

        .limit-value {
            color: #c4b5fd;
            font-weight: 800;
            white-space: nowrap;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            padding: 0.35rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 1.1rem;
            border-radius: 999px;
            color: rgba(248, 251, 255, 0.72);
            font-weight: 800;
            transition: all 180ms ease;
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.88), rgba(236, 72, 153, 0.72));
            box-shadow: 0 12px 34px rgba(99, 102, 241, 0.34);
        }

        .stButton > button {
            width: 100%;
            min-height: 3rem;
            border: 0;
            border-radius: 999px;
            color: #ffffff;
            font-weight: 850;
            background: linear-gradient(135deg, var(--primary-2), var(--primary) 48%, var(--pink));
            box-shadow: 0 16px 42px rgba(139, 92, 246, 0.35), 0 0 24px rgba(34, 211, 238, 0.12);
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.01);
            filter: brightness(1.08);
            box-shadow: 0 20px 55px rgba(139, 92, 246, 0.48), 0 0 32px rgba(34, 211, 238, 0.22);
        }

        [data-testid="stFileUploader"] {
            border: 1px dashed rgba(103, 232, 249, 0.46);
            border-radius: 24px;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.06);
            transition: border-color 180ms ease, background 180ms ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: rgba(236, 72, 153, 0.62);
            background: rgba(255, 255, 255, 0.085);
        }

        [data-testid="stImage"], [data-testid="stVideo"], .stAudio {
            border-radius: 22px;
            overflow: hidden;
        }

        div[data-testid="stImage"] img {
            border-radius: 22px;
            transition: transform 180ms ease, filter 180ms ease;
        }

        div[data-testid="stImage"] img:hover {
            transform: scale(1.015);
            filter: saturate(1.08);
        }

        .result-card {
            margin: 1.1rem 0;
            padding: 1.35rem;
            border-radius: 28px;
            border: 1px solid rgba(103, 232, 249, 0.28);
            background:
                linear-gradient(145deg, rgba(34, 211, 238, 0.12), rgba(139, 92, 246, 0.12)),
                rgba(255, 255, 255, 0.07);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.38), 0 0 34px rgba(34, 211, 238, 0.10);
            animation: slideIn 0.42s ease both;
        }

        .result-label {
            margin: 0 0 0.55rem;
            color: #a5f3fc;
            font-weight: 850;
            letter-spacing: 0;
            text-transform: uppercase;
            font-size: 0.78rem;
        }

        .caption-text {
            margin: 0;
            color: #ffffff;
            font-size: clamp(1.18rem, 2vw, 1.58rem);
            line-height: 1.6;
            font-weight: 700;
        }

        .alert-card {
            margin: 1rem 0;
            padding: 1rem 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(248, 113, 113, 0.36);
            background: linear-gradient(145deg, rgba(248, 113, 113, 0.16), rgba(236, 72, 153, 0.10));
            color: #fecaca;
            box-shadow: 0 18px 55px rgba(127, 29, 29, 0.28);
            animation: fadeUp 0.3s ease both;
            font-weight: 650;
        }

        .info-card {
            margin: 1rem 0;
            padding: 1rem 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(34, 211, 238, 0.28);
            background: linear-gradient(145deg, rgba(34, 211, 238, 0.12), rgba(99, 102, 241, 0.10));
            color: #cffafe;
            box-shadow: 0 18px 55px rgba(8, 47, 73, 0.24);
            animation: fadeUp 0.3s ease both;
            font-weight: 650;
        }

        .step-flow {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.8rem 0 1rem;
        }

        .step {
            padding: 0.8rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.13);
            background: rgba(255, 255, 255, 0.065);
            color: rgba(248, 251, 255, 0.88);
            font-weight: 750;
        }

        .helper-text {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.55;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, var(--cyan), var(--primary), var(--pink));
        }

        footer, #MainMenu {
            visibility: hidden;
        }

        [data-testid="stHeader"] {
            background: transparent;
            color: var(--text);
        }

        [data-testid="collapsedControl"] {
            color: var(--text);
            z-index: 999999;
        }

        .footer {
            margin-top: 2.3rem;
            text-align: center;
            color: rgba(226, 232, 240, 0.52);
            font-size: 0.88rem;
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(18px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(12px) scale(0.985);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding-top: 1.5rem;
            }

            .hero-grid, .step-flow {
                grid-template-columns: 1fr;
            }

            .glass-card {
                padding: 0.95rem;
                border-radius: 22px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------
def open_card(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title">{title}</div>
            {f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def render_alert(message: str, icon: str = "⚠️"):
    st.markdown(
        f'<div class="alert-card">{icon} {message}</div>',
        unsafe_allow_html=True,
    )


def render_info(message: str, icon: str = "✨"):
    st.markdown(
        f'<div class="info-card">{icon} {message}</div>',
        unsafe_allow_html=True,
    )


def render_step_flow(steps: list[str]):
    step_markup = "".join(f'<div class="step">{step}</div>' for step in steps)
    st.markdown(f'<div class="step-flow">{step_markup}</div>', unsafe_allow_html=True)


def render_frame_grid(frames):
    columns = st.columns(min(len(frames), 3))
    for index, frame in enumerate(frames):
        with columns[index % len(columns)]:
            st.image(frame, caption=f"Frame {index + 1}", use_container_width=True)


@st.cache_resource(show_spinner="Loading captioning model...")
def get_cached_captioner():
    return load_captioner()


def validate_media_source(media_source) -> str | None:
    """Return a user-facing validation message when media input is invalid."""
    if media_source is None:
        return None

    if hasattr(media_source, "mode"):
        return None

    media_type = getattr(media_source, "type", None)
    media_size = getattr(media_source, "size", 0)

    if media_type in ALLOWED_IMAGE_TYPES:
        max_size_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
        if media_size > max_size_bytes:
            return f"Please choose an image smaller than {MAX_IMAGE_SIZE_MB} MB."
        return None

    if media_type in ALLOWED_VIDEO_TYPES:
        max_size_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
        if media_size > max_size_bytes:
            return f"Please choose a video smaller than {MAX_VIDEO_SIZE_MB} MB."
        return None

    return "Please upload a JPG, PNG, MP4, MOV, or AVI file."


def get_media_kind(media_source) -> str | None:
    if hasattr(media_source, "mode"):
        return "image"

    media_type = getattr(media_source, "type", "")
    if media_type in ALLOWED_IMAGE_TYPES:
        return "image"
    if media_type in ALLOWED_VIDEO_TYPES:
        return "video"
    return None


# ---------------------------------------------------------------------------
# UI Layout Sections
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <section class="hero">
            <div class="hero-badge">✨ Multimodal AI Workspace</div>
            <h1>🎥 AI Caption Studio</h1>
            <p>Turn visuals into intelligent descriptions instantly with image, video, live-camera, and voice narration support.</p>
            <div class="hero-grid">
                <div class="metric-pill">🖼️ Image Captioning</div>
                <div class="metric-pill">🎬 Video Summaries</div>
                <div class="metric-pill">🔊 Voice Narration</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> bool:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-panel">
                <p class="sidebar-title">⚙️ Features</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        enable_audio = st.toggle("🔊 Voice Narration", value=True)

        st.markdown(
            f"""
            <div class="sidebar-panel">
                <p class="sidebar-title">📏 Limits</p>
                <div class="limit-row"><span>🖼️ Images</span><span class="limit-value">JPG, PNG</span></div>
                <div class="limit-row"><span>🎬 Videos</span><span class="limit-value">MP4, MOV, AVI</span></div>
                <div class="limit-row"><span>Image size</span><span class="limit-value">{MAX_IMAGE_SIZE_MB} MB</span></div>
                <div class="limit-row"><span>Video size</span><span class="limit-value">{MAX_VIDEO_SIZE_MB} MB</span></div>
                <div class="limit-row"><span>Frame samples</span><span class="limit-value">{VIDEO_SAMPLE_FRAMES}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return enable_audio


def render_media_input():
    open_card(
        "Upload or Capture Media",
        "Choose a visual source. Captioning begins automatically after the media is ready.",
    )
    upload_tab, camera_tab, live_tab = st.tabs(["⬆️ Upload", "📷 Camera Photo", "🎥 Live Camera"])

    with upload_tab:
        st.markdown(
            '<div class="helper-text">Drop in an image or short video. Supported formats: JPG, PNG, MP4, MOV, AVI.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose an image or video",
            type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
            help="Use a clear image or a short video clip for best results.",
            label_visibility="collapsed",
        )

    with camera_tab:
        st.markdown(
            '<div class="helper-text">Capture a clear image for best results. The model will caption the still photo automatically.</div>',
            unsafe_allow_html=True,
        )
        captured_file = st.camera_input("Take a photo", label_visibility="collapsed")

    with live_tab:
        live_frame = render_live_camera()

    close_card()
    return uploaded_file or captured_file or live_frame


class LatestFrameProcessor(VideoProcessorBase if WEBRTC_AVAILABLE else object):
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        self.latest_frame = frame.to_image()
        return frame


def render_live_camera():
    st.markdown(
        '<div class="helper-text">Start the live camera, then capture the current frame when the scene looks right.</div>',
        unsafe_allow_html=True,
    )

    if not WEBRTC_AVAILABLE:
        render_alert(
            "Live camera support could not load streamlit-webrtc. "
            f"Cloud import error: {WEBRTC_IMPORT_ERROR or 'unknown error'}. "
            "Reboot the Streamlit app after the latest requirements install finishes."
        )
        return None

    context = webrtc_streamer(
        key="live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LatestFrameProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
    )

    if not context.video_processor:
        render_info("Start the live camera, then capture the current frame.")
        return None

    if st.button("📸 Capture & Caption Frame", use_container_width=True):
        frame = context.video_processor.latest_frame
        if frame is None:
            render_alert("No live frame is available yet. Wait a moment and try again.")
            return None
        return frame.convert("RGB")

    return None


def render_caption_output(caption: str):
    st.markdown(
        f"""
        <div class="result-card">
            <p class="result-label">🧠 Generated Caption</p>
            <p class="caption-text">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_audio_output(caption: str):
    with st.spinner("🔊 Creating voice narration..."):
        audio_bytes = create_caption_audio(caption)

    st.markdown(
        """
        <div class="result-card">
            <p class="result-label">🔊 Voice Narration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.audio(audio_bytes, format="audio/mp3")


# ---------------------------------------------------------------------------
# Processing UI Components
# ---------------------------------------------------------------------------
def caption_image(media_source) -> str:
    image = media_source if hasattr(media_source, "mode") else load_image(media_source)
    open_card("Preview", "Your selected image is ready for analysis.")
    st.image(image, use_container_width=True)
    close_card()

    captioner = get_cached_captioner()
    with st.spinner("🧠 AI is analyzing your image..."):
        return captioner.generate_caption(image)


def caption_video(media_source) -> str:
    open_card("Video Preview", "The clip will be sampled into representative frames before captioning.")
    st.video(media_source)
    close_card()

    render_step_flow(["1. 🎬 Preview video", "2. 🧩 Extract frames", "3. 🧠 Caption sequence"])

    with st.spinner("🎬 Understanding video frames..."):
        frames = sample_video_frames(media_source)

    open_card("Sampled Frames", "These frames are used to build the final video summary.")
    render_frame_grid(frames)
    close_card()

    captioner = get_cached_captioner()
    frame_captions = []
    progress = st.progress(0, text="🧠 Captioning sampled frame 0 of 0")

    for index, frame in enumerate(frames, start=1):
        frame_captions.append(captioner.generate_caption(frame))
        progress.progress(
            index / len(frames),
            text=f"🧠 Captioning sampled frame {index} of {len(frames)}",
        )

    progress.empty()
    return summarize_frame_captions(frame_captions)


def render_result(media_source, enable_audio: bool):
    media_kind = get_media_kind(media_source)

    try:
        if media_kind == "image":
            caption = caption_image(media_source)
        elif media_kind == "video":
            caption = caption_video(media_source)
        else:
            render_alert("Unsupported media type.", icon="❌")
            return
    except Exception as error:
        render_alert(f"Caption generation failed: {error}", icon="❌")
        return

    render_caption_output(caption)

    if not enable_audio:
        return

    try:
        render_audio_output(caption)
    except Exception as error:
        render_alert(f"Caption generated, but voice output failed: {error}")


def main():
    inject_global_styles()
    render_header()
    enable_audio = render_sidebar()

    media_source = render_media_input()
    validation_error = validate_media_source(media_source)
    if validation_error:
        render_alert(validation_error, icon="⚠️")
        return

    if media_source is None:
        render_info("Upload an image, upload a short video, or take a live/camera photo to begin.")
        st.markdown('<div class="footer">Created by MOHAMMED GHANIM SIDDIQUI</div>', unsafe_allow_html=True)
        return

    render_result(media_source, enable_audio)

    st.markdown('<div class="footer">Created by MOHAMMED GHANIM SIDDIQUI · AI Caption Studio</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
