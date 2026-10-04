import os
import tempfile
import hashlib

import streamlit as st
from faster_whisper import WhisperModel


# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="Voice to Text",
    page_icon="🎙️",
    layout="centered"
)


# ---------------- Languages ----------------

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Bengali": "bn",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Urdu": "ur",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Japanese": "ja",
}


# ---------------- Session State ----------------

if "result" not in st.session_state:
    st.session_state.result = None


# ---------------- Load Model ----------------

@st.cache_resource
def load_model(size):
    return WhisperModel(
        size,
        device="cpu",
        compute_type="int8"
    )


# ---------------- SRT Function ----------------

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def to_srt(segments):
    lines = []

    for i, segment in enumerate(segments, start=1):

        start = format_time(segment["start"])
        end = format_time(segment["end"])

        lines.append(
            f"{i}\n"
            f"{start} --> {end}\n"
            f"{segment['text']}\n"
        )

    return "\n".join(lines)


# ---------------- Title ----------------

st.title("🎙️ Voice to Text")

st.write(
    "Upload an audio file and convert speech into text "
    "using Whisper."
)


# ---------------- Sidebar ----------------

st.sidebar.header("⚙️ Settings")

lang_name = st.sidebar.selectbox(
    "🌐 Language",
    list(LANGUAGES.keys())
)


model_size = st.sidebar.selectbox(
    "🤖 Whisper Model",
    ["tiny", "base", "small"],
    index=1
)


task = st.sidebar.radio(
    "📋 Task",
    [
        "Transcribe",
        "Translate to English"
    ]
)


st.sidebar.info(
    "💡 Larger models usually provide better accuracy "
    "but require more time and memory."
)


# ---------------- Audio Upload ----------------

audio = st.file_uploader(
    "🎵 Upload Audio",
    type=[
        "wav",
        "mp3",
        "m4a",
        "ogg",
        "flac",
        "webm"
    ]
)


# ---------------- Process Audio ----------------

if audio is not None:

    st.audio(
        audio,
        format=audio.type
    )

    # Create unique ID for the uploaded audio
    audio_id = hashlib.md5(
        audio.getvalue()
    ).hexdigest()

    # ---------------- Transcribe Button ----------------

    if st.button(
        "🎙️ Convert to Text",
        type="primary"
    ):

        suffix = (
            os.path.splitext(
                getattr(audio, "name", "") or ""
            )[1]
            or ".wav"
        )

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(
                audio.getvalue()
            )

            path = tmp.name

        try:

            # Load Whisper model
            model = load_model(model_size)

            with st.spinner(
                "🎧 Transcribing... "
                "(longer audio takes longer)"
            ):

                # Transcribe audio
                seg_iter, info = model.transcribe(
                    path,
                    language=LANGUAGES[lang_name],
                    task=(
                        "translate"
                        if task.startswith("Translate")
                        else "transcribe"
                    ),
                    beam_size=5,
                    vad_filter=True
                )

                # Convert generator to list
                segments = [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    }
                    for segment in seg_iter
                ]

        except Exception as e:

            st.error(
                f"❌ Transcription failed: {e}"
            )

            st.stop()

        finally:

            # Delete temporary file
            if os.path.exists(path):
                os.remove(path)


        # ---------------- Save Result ----------------

        st.session_state.result = {

            "audio_id": audio_id,

            "text": " ".join(
                segment["text"]
                for segment in segments
            ),

            "segments": segments,

            "language": info.language,

            "prob": info.language_probability,

            "duration": info.duration
        }


    # ---------------- Show Result ----------------

    result = st.session_state.result

    if (
        result
        and result["audio_id"] == audio_id
    ):

        st.success(
            f"Detected language: "
            f"**{result['language']}** "
            f"({result['prob'] * 100:.0f}% confidence) "
            f"• Duration: "
            f"{result['duration']:.1f}s"
        )


        # ---------------- Transcript ----------------

        st.subheader("📝 Transcript")

        st.text_area(
            "Text",
            result["text"],
            height=200
        )


        # ---------------- Timestamps ----------------

        with st.expander(
            "⏱️ Timestamps"
        ):

            for segment in result["segments"]:

                st.write(
                    f"`[{segment['start']:6.1f}s → "
                    f"{segment['end']:6.1f}s]` "
                    f"{segment['text']}"
                )


        # ---------------- Download Buttons ----------------

        st.subheader(
            "⬇️ Download"
        )

        c1, c2 = st.columns(2)


        with c1:

            st.download_button(
                "⬇️ Download .txt",
                result["text"],
                "transcript.txt",
                "text/plain"
            )


        with c2:

            st.download_button(
                "⬇️ Download .srt",
                to_srt(result["segments"]),
                "transcript.srt",
                "text/plain"
            )


else:

    st.info(
        "👆 Upload an audio file to begin."
    )
