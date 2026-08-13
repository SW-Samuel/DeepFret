import base64
import gc
import json
import os
import subprocess
import tempfile
from io import BytesIO

import librosa
import matplotlib

matplotlib.use("Agg")  # Backend não-interativo para evitar falhas em servidores
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch

# Configuração da página Streamlit
st.set_page_config(
    page_title="DEEPFRET",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização customizada
st.markdown(
    """
    <style>
        .stApp { background-color: #0f1117; color: #f1f5f9; }
        .css-1d3b10b { background-color: #161922; }
        
        /* Oculta as barras de rolagem nativas do Streamlit/Browser */
        ::-webkit-scrollbar {
            width: 0px;
            height: 0px;
            background: transparent;
        }
        html, body, [data-testid="stAppViewContainer"] {
            overflow: -moz-scrollbars-none;
            -ms-overflow-style: none;
            scrollbar-width: none;
        }
        
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            background-color: #1e293b;
            color: #f8fafc;
            border: 1px solid #334155;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            background-color: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 1. TEMPLATES DE ACORDES & BASE DE DADOS
# ---------------------------------------------------------
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_LABELS = []
CHORD_TEMPLATES = []

for i, root in enumerate(PITCH_NAMES):
    maj_t = np.zeros(12)
    maj_t[i] = 1.5
    maj_t[(i + 4) % 12] = 1.2
    maj_t[(i + 7) % 12] = 0.8
    CHORD_LABELS.append(root)
    CHORD_TEMPLATES.append(maj_t / np.linalg.norm(maj_t))

    min_t = np.zeros(12)
    min_t[i] = 1.5
    min_t[(i + 3) % 12] = 1.2
    min_t[(i + 7) % 12] = 0.8
    CHORD_LABELS.append(f"{root}m")
    CHORD_TEMPLATES.append(min_t / np.linalg.norm(min_t))

CHORD_TEMPLATES = np.array(CHORD_TEMPLATES)

CHORD_DB = {
    "C": {
        "positions": {2: (1, 1), 4: (2, 2), 5: (3, 3)},
        "top": {1: "O", 3: "O", 6: "X"},
        "base_fret": 1,
    },
    "C#": {
        "positions": {2: (3, 4), 3: (3, 3), 4: (3, 2)},
        "top": {6: "X"},
        "base_fret": 4,
        "barre": (1, 1, 5),
    },
    "D": {
        "positions": {1: (2, 2), 2: (3, 3), 3: (2, 1)},
        "top": {4: "O", 5: "X", 6: "X"},
        "base_fret": 1,
    },
    "D#": {
        "positions": {2: (3, 4), 3: (3, 3), 4: (3, 2)},
        "top": {6: "X"},
        "base_fret": 6,
        "barre": (1, 1, 5),
    },
    "E": {
        "positions": {3: (1, 1), 4: (2, 3), 5: (2, 2)},
        "top": {1: "O", 2: "O", 6: "O"},
        "base_fret": 1,
    },
    "F": {
        "positions": {3: (2, 2), 4: (3, 4), 5: (3, 3)},
        "top": {},
        "base_fret": 1,
        "barre": (1, 1, 6),
    },
    "F#": {
        "positions": {3: (2, 2), 4: (3, 4), 5: (3, 3)},
        "top": {},
        "base_fret": 2,
        "barre": (1, 1, 6),
    },
    "G": {
        "positions": {1: (3, 3), 5: (2, 1), 6: (3, 2)},
        "top": {2: "O", 3: "O", 4: "O"},
        "base_fret": 1,
    },
    "G#": {
        "positions": {3: (2, 2), 4: (3, 4), 5: (3, 3)},
        "top": {},
        "base_fret": 4,
        "barre": (1, 1, 6),
    },
    "A": {
        "positions": {2: (2, 3), 3: (2, 2), 4: (2, 1)},
        "top": {1: "O", 5: "O", 6: "X"},
        "base_fret": 1,
    },
    "A#": {
        "positions": {2: (3, 4), 3: (3, 3), 4: (3, 2)},
        "top": {6: "X"},
        "base_fret": 1,
        "barre": (1, 1, 5),
    },
    "B": {
        "positions": {2: (3, 4), 3: (3, 3), 4: (3, 2)},
        "top": {6: "X"},
        "base_fret": 2,
        "barre": (1, 1, 5),
    },
    "Cm": {
        "positions": {2: (2, 2), 3: (3, 4), 4: (3, 3)},
        "top": {6: "X"},
        "base_fret": 3,
        "barre": (1, 1, 5),
    },
    "C#m": {
        "positions": {2: (2, 2), 3: (3, 4), 4: (3, 3)},
        "top": {6: "X"},
        "base_fret": 4,
        "barre": (1, 1, 5),
    },
    "Dm": {
        "positions": {1: (1, 1), 2: (3, 3), 3: (2, 2)},
        "top": {4: "O", 5: "X", 6: "X"},
        "base_fret": 1,
    },
    "D#m": {
        "positions": {2: (2, 2), 3: (3, 4), 4: (3, 3)},
        "top": {6: "X"},
        "base_fret": 6,
        "barre": (1, 1, 5),
    },
    "Em": {
        "positions": {4: (2, 3), 5: (2, 2)},
        "top": {1: "O", 2: "O", 3: "O", 6: "O"},
        "base_fret": 1,
    },
    "Fm": {
        "positions": {4: (3, 4), 5: (3, 3)},
        "top": {},
        "base_fret": 1,
        "barre": (1, 1, 6),
    },
    "F#m": {
        "positions": {4: (3, 4), 5: (3, 3)},
        "top": {},
        "base_fret": 2,
        "barre": (1, 1, 6),
    },
    "Gm": {
        "positions": {4: (3, 4), 5: (3, 3)},
        "top": {},
        "base_fret": 3,
        "barre": (1, 1, 6),
    },
    "G#m": {
        "positions": {4: (3, 4), 5: (3, 3)},
        "top": {},
        "base_fret": 4,
        "barre": (1, 1, 6),
    },
    "Am": {
        "positions": {2: (1, 1), 3: (2, 3), 4: (2, 2)},
        "top": {1: "O", 5: "O", 6: "X"},
        "base_fret": 1,
    },
    "A#m": {
        "positions": {2: (2, 2), 3: (3, 4), 4: (3, 3)},
        "top": {6: "X"},
        "base_fret": 1,
        "barre": (1, 1, 5),
    },
    "Bm": {
        "positions": {2: (2, 2), 3: (3, 4), 4: (3, 3)},
        "top": {6: "X"},
        "base_fret": 2,
        "barre": (1, 1, 5),
    },
}


# ---------------------------------------------------------
# GERADORES & UTILITÁRIOS
# ---------------------------------------------------------
def convert_to_compressed_mp3_b64(input_audio_path):
    """Converte o áudio para MP3 otimizado. Se falhar, retorna o áudio original com o MIME correto."""
    out_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_audio_path,
        "-vn",
        "-ar",
        "22050",
        "-ac",
        "1",
        "-b:a",
        "96k",
        out_mp3,
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        with open(out_mp3, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
        return b64_str, "audio/mp3"
    except Exception:
        ext = os.path.splitext(input_audio_path)[1].lower().replace(".", "")
        mime = f"audio/{'mpeg' if ext == 'mp3' else ext}"
        with open(input_audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime
    finally:
        if os.path.exists(out_mp3):
            try:
                os.remove(out_mp3)
            except Exception:
                pass


def generate_text_chord_sheet(timeline, song_title="Música"):
    lines = [
        f"CIFRA POR BATIDA (COMPASSOS 4/4) - {song_title.upper()}",
        "=" * 50,
        "",
    ]
    measure = []
    comp_num = 1

    for item in timeline:
        measure.append(f"{item['chord']:<3}")
        if len(measure) == 4:
            measure_str = " | ".join(measure)
            lines.append(f"Comp. {comp_num:03d}:  | {measure_str} |")
            measure = []
            comp_num += 1

    if measure:
        measure_str = " | ".join(measure)
        lines.append(f"Comp. {comp_num:03d}:  | {measure_str} |")

    return "\n".join(lines)


@st.cache_data
def get_chord_image_base64(chord_name):
    if chord_name not in CHORD_DB:
        return ""

    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    data = CHORD_DB[chord_name]

    # Desenhar cordas (1 a 6)
    for string in range(1, 7):
        ax.axvline(x=string, color="#e2e8f0", linewidth=2)

    # Desenhar trastes
    for fret in range(6):
        ax.axhline(y=-fret, color="#64748b", linewidth=2)

    base_fret = data.get("base_fret", 1)
    if base_fret == 1:
        ax.axhline(y=0, color="#f8fafc", linewidth=7)

    # Numeração das casas
    for fret_i in range(1, 6):
        actual_fret = base_fret + fret_i - 1
        is_start_fret = base_fret > 1 and fret_i == 1
        color = "#f59e0b" if is_start_fret else "#94a3b8"

        ax.text(
            0.1,
            -fret_i + 0.5,
            f"{actual_fret}ª",
            va="center",
            ha="right",
            fontsize=12,
            color=color,
            weight="bold",
        )

    if base_fret > 1:
        ax.text(
            3.5,
            0.65,
            f"Início: {base_fret}ª Casa",
            ha="center",
            va="center",
            fontsize=11,
            color="#f59e0b",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#1e293b",
                edgecolor="#f59e0b",
                lw=1.5,
            ),
        )

    # Marcadores de cordas soltas (O) e abafadas (X)
    if "top" in data:
        for string, symbol in data["top"].items():
            sym_color = "#ef4444" if symbol == "X" else "#10b981"
            ax.text(
                7 - string,
                0.45,
                symbol,
                ha="center",
                va="center",
                fontsize=13,
                color=sym_color,
                weight="bold",
            )

    if "barre" in data:
        fret, start_str, end_str = data["barre"]

        rect = patches.FancyBboxPatch(
            ((7 - end_str) - 0.28, -fret + 0.2),
            (end_str - start_str) + 0.56,
            0.6,
            boxstyle="round,pad=0,rounding_size=0.15",
            linewidth=0,
            facecolor="#3b82f6",
            zorder=2,
        )
        ax.add_patch(rect)

    # Posições dos dedos
    for string, (fret, finger) in data["positions"].items():
        x_pos = 7 - string
        y_pos = -fret + 0.5
        circle = patches.Circle((x_pos, y_pos), 0.35, color="#3b82f6", zorder=3)
        ax.add_patch(circle)
        ax.text(
            x_pos,
            y_pos,
            str(finger),
            color="white",
            weight="bold",
            ha="center",
            va="center",
            zorder=4,
            fontsize=11,
        )

    ax.set_xlim(-0.4, 6.8)
    ax.set_ylim(-5.5, 1.1)
    ax.axis("off")
    ax.set_title(chord_name, fontsize=20, weight="bold", color="#38bdf8", y=-0.15)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=True, dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------
# PROCESSAMENTO DE ÁUDIO & IA
# ---------------------------------------------------------
def separate_audio_demucs(audio_path):
    output_dir = tempfile.mkdtemp()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Libera a memória RAM não utilizada antes da inferência
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Otimização para Streamlit Cloud: htdemucs_ft, segment de 4s, 1 thread
    cmd = [
        "demucs",
        "--two-stems=vocals",
        "-n",
        "htdemucs_ft",
        "-d",
        device,
        "-j",
        "1",
        "--segment",
        "4",
        "-o",
        output_dir,
        audio_path,
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        clean_path = os.path.join(
            output_dir, "htdemucs_ft", filename, "no_vocals.wav"
        )
        if os.path.exists(clean_path):
            return clean_path
    except Exception:
        st.sidebar.warning("⚠️ Demucs indisponível. Processando áudio original.")

    return audio_path


def estimate_key_priors(chroma_mean):
    maj_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    min_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.34, 3.17, 3.28]
    )

    maj_profile /= np.linalg.norm(maj_profile)
    min_profile /= np.linalg.norm(min_profile)

    best_score = -1
    best_key_idx = 0
    best_is_minor = False

    for root in range(12):
        maj_shifted = np.roll(maj_profile, root)
        min_shifted = np.roll(min_profile, root)

        score_maj = np.dot(chroma_mean, maj_shifted)
        score_min = np.dot(chroma_mean, min_shifted)

        if score_maj > best_score:
            best_score = score_maj
            best_key_idx = root
            best_is_minor = False
        if score_min > best_score:
            best_score = score_min
            best_key_idx = root
            best_is_minor = True

    if not best_is_minor:
        scale = [(best_key_idx + step) % 12 for step in [0, 2, 4, 5, 7, 9, 11]]
    else:
        scale = [(best_key_idx + step) % 12 for step in [0, 2, 3, 5, 7, 8, 10]]

    priors = np.ones(len(CHORD_LABELS)) * 0.35

    for idx, label in enumerate(CHORD_LABELS):
        root_name = label[:-1] if label.endswith("m") else label
        r_idx = PITCH_NAMES.index(root_name)
        if r_idx in scale:
            priors[idx] = 1.0

    return priors


def process_audio_beat_by_beat(clean_audio_path):
    y, sr = librosa.load(clean_audio_path, sr=22050)
    hop_length = 512
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    if len(beats) < 10:
        beat_times = np.arange(0, duration, 0.5)
    else:
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    if len(beat_times) == 0 or beat_times[0] > 0.0:
        beat_times = np.insert(beat_times, 0, 0.0)
    if beat_times[-1] < duration:
        beat_times = np.append(beat_times, duration)

    beat_times = np.unique(beat_times)
    beat_times.sort()

    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    beat_frames = np.unique(beat_frames)
    beat_frames.sort()

    y_harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.note_to_hz("C1"),
    )
    chroma = librosa.util.normalize(chroma, axis=0)

    # Garantir limites válidos dentro do tamanho do chroma
    beat_frames = beat_frames[beat_frames < chroma.shape[1]]
    if len(beat_frames) == 0 or beat_frames[0] != 0:
        beat_frames = np.insert(beat_frames, 0, 0)
    if beat_frames[-1] != chroma.shape[1]:
        beat_frames = np.append(beat_frames, chroma.shape[1])

    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    chroma_mean = np.mean(chroma_sync, axis=1)
    if np.linalg.norm(chroma_mean) > 0:
        chroma_mean /= np.linalg.norm(chroma_mean)
        diatonic_priors = estimate_key_priors(chroma_mean)
    else:
        diatonic_priors = np.ones(len(CHORD_LABELS))

    num_frames = chroma_sync.shape[1]
    num_chords = len(CHORD_LABELS)
    prob_emissions = np.zeros((num_chords, num_frames))

    for i in range(num_chords):
        prob_emissions[i, :] = (
            np.dot(CHORD_TEMPLATES[i], chroma_sync) * diatonic_priors[i]
        )

    prob_emissions = np.exp(prob_emissions * 12)
    prob_emissions /= prob_emissions.sum(axis=0, keepdims=True)

    sim_matrix = np.dot(CHORD_TEMPLATES, CHORD_TEMPLATES.T)
    transition_matrix = np.exp(sim_matrix * 3.5)
    np.fill_diagonal(transition_matrix, transition_matrix.diagonal() * 6.0)
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    states = librosa.sequence.viterbi(prob_emissions, transition_matrix)

    raw_chords = [CHORD_LABELS[s] for s in states]
    smoothed_chords = raw_chords.copy()
    for i in range(1, len(smoothed_chords) - 1):
        if (
            smoothed_chords[i - 1] == smoothed_chords[i + 1]
            and smoothed_chords[i] != smoothed_chords[i - 1]
        ):
            smoothed_chords[i] = smoothed_chords[i - 1]

    frame_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    beat_timeline = []
    for i in range(len(smoothed_chords)):
        start_time = frame_times[i]
        end_time = frame_times[i + 1] if (i + 1) < len(frame_times) else duration

        beat_timeline.append(
            {
                "beat_idx": i,
                "chord": smoothed_chords[i],
                "start": round(float(start_time), 2),
                "end": round(float(end_time), 2),
            }
        )

    return beat_timeline


# ---------------------------------------------------------
# SIDEBAR DE CONFIGURAÇÕES
# ---------------------------------------------------------
with st.sidebar:
    st.title("DEEPFRET")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Selecione a música (MP3/WAV)",
        type=["mp3", "wav", "ogg"],
        key="audio_file_uploader",
    )
    st.markdown("---")
    st.info(
        "💡 **Dica de Treino:** Utilize o controle de velocidade no player para desacelerar o áudio sem alterar o tom!"
    )

# ---------------------------------------------------------
# CORPO PRINCIPAL DO APP
# ---------------------------------------------------------
if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if (
        "beat_timeline" not in st.session_state
        or st.session_state.get("file_key") != file_key
    ):
        bytes_data = uploaded_file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(bytes_data)
            tmp_path = tmp.name

        try:
            with st.spinner("🤖 Isolando instrumental com Demucs AI..."):
                clean_audio_path = separate_audio_demucs(tmp_path)

            with st.spinner("🎵 Mapeando batidas e extraindo acordes..."):
                st.session_state.beat_timeline = process_audio_beat_by_beat(
                    clean_audio_path
                )

            with st.spinner("⚡ Otimizando áudio para o player em tempo real..."):
                audio_b64, mime_type = convert_to_compressed_mp3_b64(
                    clean_audio_path
                )
                st.session_state.compressed_audio_b64 = audio_b64
                st.session_state.mime_type = mime_type

            st.session_state.file_key = file_key
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    timeline = st.session_state.beat_timeline
    audio_b64 = st.session_state.compressed_audio_b64
    mime_type = st.session_state.mime_type

    if timeline and audio_b64:
        all_chord_images = {
            chord: get_chord_image_base64(chord) for chord in CHORD_DB.keys()
        }

        txt_sheet = generate_text_chord_sheet(timeline, uploaded_file.name)
        json_sheet = json.dumps(timeline, indent=2)

        # HTML Engine Interativo
        html_code = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <script src="https://cdn.tailwindcss.com"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <style>
                body {{ background-color: #0f1117; color: #f8fafc; font-family: system-ui, sans-serif; margin: 0; padding: 0; }}
                .beat-box {{ transition: all 0.15s ease-in-out; }}
                .beat-box.active {{
                    background-color: #2563eb !important;
                    color: #ffffff !important;
                    transform: scale(1.08);
                    box-shadow: 0 0 15px rgba(37, 99, 235, 0.6);
                    z-index: 20;
                }}
                .beat-box.chord-change {{ border-top: 3px solid #38bdf8; }}
            </style>
        </head>
        <body class="p-2">
            <div class="max-w-6xl mx-auto flex flex-col gap-4">
                
                <!-- Header Control Panel -->
                <div class="bg-[#161922] border border-slate-800 rounded-xl p-5 sticky top-0 z-50 shadow-2xl flex flex-col md:flex-row items-center gap-6">
                    
                    <!-- Diagram Box -->
                    <div class="w-48 h-60 bg-[#0f1117] border border-slate-700 rounded-xl flex items-center justify-center p-2 shrink-0 shadow-inner">
                        <div id="chord-diagram" class="w-full h-full flex items-center justify-center">
                            <span class="text-xs text-slate-500 text-center">Aguardando play...</span>
                        </div>
                    </div>

                    <!-- Player, Tools & Download Buttons -->
                    <div class="flex-1 w-full flex flex-col gap-3">
                        <div class="flex flex-wrap items-center justify-between gap-2">
                            <div class="flex items-center gap-3">
                                <h2 id="chord-title" class="text-xl font-bold text-blue-400">Pronto para tocar</h2>
                                <div id="audio-loading" class="text-xs text-amber-400 flex items-center gap-1.5 animate-pulse">
                                    <i class="fa-solid fa-spinner fa-spin"></i>
                                    <span>Carregando áudio...</span>
                                </div>
                            </div>

                            <div class="flex items-center gap-2">
                                <button onclick="downloadTxt()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 flex items-center gap-1.5 transition">
                                    <i class="fa-solid fa-file-lines text-blue-400"></i> Baixar TXT
                                </button>
                                <button onclick="downloadJson()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 flex items-center gap-1.5 transition">
                                    <i class="fa-solid fa-code text-pink-400"></i> Baixar JSON
                                </button>
                            </div>
                        </div>

                        <!-- Audio Tag -->
                        <audio id="audio-player" class="w-full h-10 rounded-lg" controls preload="auto">
                            <source src="data:{mime_type};base64,{audio_b64}" type="{mime_type}">
                        </audio>

                        <!-- Playback Speed Controls & Transpose -->
                        <div class="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800">
                            
                            <!-- Transpose Buttons -->
                            <div class="flex items-center gap-2 bg-slate-800/80 px-3 py-1 rounded-lg border border-slate-700">
                                <span class="text-xs text-slate-400 mr-1">Tom:</span>
                                <button onclick="transpose(-1)" class="px-2.5 py-0.5 bg-slate-700 hover:bg-slate-600 rounded text-xs font-bold">-1</button>
                                <span id="transpose-val" class="text-xs font-mono text-pink-400 font-bold px-1">0</span>
                                <button onclick="transpose(1)" class="px-2.5 py-0.5 bg-slate-700 hover:bg-slate-600 rounded text-xs font-bold">+1</button>
                            </div>

                            <!-- Speed Control -->
                            <div class="flex items-center gap-2">
                                <i class="fa-solid fa-gauge text-slate-400 text-xs"></i>
                                <span class="text-xs text-slate-400">Velocidade:</span>
                                <div class="flex bg-slate-800 rounded-lg p-0.5 border border-slate-700">
                                    <button onclick="setSpeed(0.5)" class="speed-btn px-2 py-0.5 text-xs rounded hover:bg-slate-700">0.5x</button>
                                    <button onclick="setSpeed(0.75)" class="speed-btn px-2 py-0.5 text-xs rounded hover:bg-slate-700">0.75x</button>
                                    <button onclick="setSpeed(1.0)" class="speed-btn px-2 py-0.5 text-xs rounded bg-blue-600 font-bold">1.0x</button>
                                    <button onclick="setSpeed(1.25)" class="speed-btn px-2 py-0.5 text-xs rounded hover:bg-slate-700">1.25x</button>
                                </div>
                            </div>

                            <div class="flex items-center gap-2">
                                <input type="range" id="speed-slider" min="0.5" max="1.5" step="0.05" value="1.0" class="w-24 accent-blue-500" oninput="updateSpeedSlider(this.value)">
                                <span id="speed-val" class="text-xs font-mono w-10 text-blue-400">1.00x</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Beat Grid -->
                <div class="bg-[#161922] border border-slate-800 rounded-xl p-4">
                    <h3 class="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">Compassos e Batidas</h3>
                    <div id="grid-container" class="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3 max-h-[480px] overflow-y-auto pr-2"></div>
                </div>

            </div>

            <script>
                const originalTimeline = {json.dumps(timeline)};
                let currentTimeline = JSON.parse(JSON.stringify(originalTimeline));
                const chordImages = {json.dumps(all_chord_images)};
                const pitchNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
                
                const rawTxtData = {json.dumps(txt_sheet)};
                const rawJsonData = {json.dumps(json_sheet)};
                const songFileName = "{os.path.splitext(uploaded_file.name)[0]}";

                const audio = document.getElementById('audio-player');
                const loader = document.getElementById('audio-loading');
                const grid = document.getElementById('grid-container');
                const chordDiagram = document.getElementById('chord-diagram');
                const chordTitle = document.getElementById('chord-title');
                const speedVal = document.getElementById('speed-val');
                const speedSlider = document.getElementById('speed-slider');
                let currentTranspose = 0;
                let activeIndex = -1;

                function downloadTxt() {{
                    const blob = new Blob([rawTxtData], {{ type: 'text/plain;charset=utf-8' }});
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(blob);
                    a.download = `cifra_${{songFileName}}.txt`;
                    a.click();
                }}

                function downloadJson() {{
                    const blob = new Blob([rawJsonData], {{ type: 'application/json;charset=utf-8' }});
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(blob);
                    a.download = `sincronizacao_${{songFileName}}.json`;
                    a.click();
                }}

                audio.oncanplaythrough = () => {{ if (loader) loader.style.display = 'none'; }};
                audio.onloadeddata = () => {{ if (loader) loader.style.display = 'none'; }};

                function setSpeed(rate) {{
                    audio.playbackRate = rate;
                    speedSlider.value = rate;
                    speedVal.innerText = rate.toFixed(2) + 'x';
                    document.querySelectorAll('.speed-btn').forEach(btn => {{
                        btn.classList.remove('bg-blue-600', 'font-bold');
                        if (parseFloat(btn.innerText) === rate) {{
                            btn.classList.add('bg-blue-600', 'font-bold');
                        }}
                    }});
                }}

                function updateSpeedSlider(val) {{
                    const rate = parseFloat(val);
                    audio.playbackRate = rate;
                    speedVal.innerText = rate.toFixed(2) + 'x';
                }}

                function shiftChord(chord, steps) {{
                    let isMinor = chord.endsWith('m');
                    let root = isMinor ? chord.slice(0, -1) : chord;
                    let idx = pitchNames.indexOf(root);
                    if (idx === -1) return chord;
                    let newIdx = (idx + steps) % 12;
                    if (newIdx < 0) newIdx += 12;
                    return pitchNames[newIdx] + (isMinor ? 'm' : '');
                }}

                function transpose(steps) {{
                    currentTranspose += steps;
                    document.getElementById('transpose-val').innerText = (currentTranspose > 0 ? '+' : '') + currentTranspose;
                    renderGrid();
                }}

                function renderGrid() {{
                    grid.innerHTML = '';
                    let lastChord = '';

                    currentTimeline.forEach((item, index) => {{
                        item.chord = shiftChord(originalTimeline[index].chord, currentTranspose);

                        if (index % 4 === 0) {{
                            var measure = document.createElement('div');
                            measure.className = 'grid grid-cols-4 gap-1 bg-slate-900/60 p-1.5 rounded-lg border border-slate-800';
                            measure.id = 'measure-' + Math.floor(index / 4);
                            grid.appendChild(measure);
                        }}

                        const beatBox = document.createElement('div');
                        beatBox.className = 'beat-box h-12 bg-[#0f1117] text-slate-400 rounded flex items-center justify-center font-bold text-sm cursor-pointer hover:bg-slate-800 hover:text-white select-none';
                        beatBox.id = 'beat-' + index;
                        beatBox.innerText = item.chord;

                        if (item.chord !== lastChord) {{
                            beatBox.classList.add('chord-change', 'text-blue-400');
                            lastChord = item.chord;
                        }}

                        beatBox.onclick = () => {{
                            audio.currentTime = item.start;
                            audio.play();
                        }};

                        const currentMeasure = grid.lastChild;
                        currentMeasure.appendChild(beatBox);
                    }});
                }}

                renderGrid();

                audio.ontimeupdate = () => {{
                    const curTime = audio.currentTime;
                    const index = currentTimeline.findIndex(b => curTime >= b.start && curTime < b.end);

                    if (index !== -1 && index !== activeIndex) {{
                        if (activeIndex !== -1) {{
                            const prev = document.getElementById('beat-' + activeIndex);
                            if (prev) prev.classList.remove('active');
                        }}

                        const activeBeat = document.getElementById('beat-' + index);
                        if (activeBeat) {{
                            activeBeat.classList.add('active');
                            const rect = activeBeat.getBoundingClientRect();
                            if (rect.top < 180 || rect.bottom > window.innerHeight) {{
                                activeBeat.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                            }}
                        }}

                        const currentChord = currentTimeline[index].chord;
                        chordTitle.innerText = "Tocando: " + currentChord;
                        if (chordImages[currentChord]) {{
                            chordDiagram.innerHTML = `<img src="data:image/png;base64,${{chordImages[currentChord]}}" class="max-h-full max-w-full object-contain" />`;
                        }} else {{
                            chordDiagram.innerHTML = `<span class="text-sm font-bold text-slate-400">${{currentChord}}</span>`;
                        }}

                        activeIndex = index;
                    }}
                }};
            </script>
        </body>
        </html>
        """

        components.html(html_code, height=820, scrolling=True)