import gradio as gr
import os
import shutil

from lipsync import LipSync
from mejorar4 import procesar_amd_rocm  # <-- tu script de mejora

# ---------- CONFIG ----------
WORKDIR = "gradio_tmp"
os.makedirs(WORKDIR, exist_ok=True)

# ---------- LIPSYNC ----------
def crear_lipsync(modelo):
    if modelo == "wav2lip_gan":
        ckpt = "lipsync/checkpoints/wav2lip_gan.pth"
    else:
        ckpt = "lipsync/checkpoints/wav2lip.pth"

    return LipSync(
        model="wav2lip",
        checkpoint_path=ckpt,
        device="cuda",
        nosmooth=True,
        cache_dir="lipsync/cache",
        img_size=96,
        save_cache=True,
    )

# ---------- FUNCIÓN PRINCIPAL ----------
def procesar(
        video_file, 
        audio_file, 
        mejorar, 
        modelo,
        weight, 
        alpha,
        every_n_frames_slider):
    if video_file is None or audio_file is None:
        return None

    lip = crear_lipsync(modelo)
    # limpiar workspace
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)

    import hashlib

    def hash_file(path):
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:8]

    vid_hash = hash_file(video_file)


    video_in = os.path.join(WORKDIR, f"video_{vid_hash}.mp4")
    #video_in = os.path.join(WORKDIR, "input_video.mp4")
    audio_in = os.path.join(WORKDIR, "input_audio.mp3")
    video_lipsync = os.path.join(WORKDIR, "lipsync.mp4")
    video_final = os.path.join(WORKDIR, "final.mp4")

    shutil.copy(video_file, video_in)
    shutil.copy(audio_file, audio_in)

    # ---------- PASO 1: LIPSYNC ----------
    lip.sync(video_in, audio_in, video_lipsync)

    # ---------- PASO 2: MEJORA (OPCIONAL) ----------
    if mejorar:
        procesar_amd_rocm(
            video_lipsync,
            audio_in,
            video_final,
            weight, 
            alpha,
            every_n_frames_slider
        )
        return video_final
    else:
        return video_lipsync

# ---------- GRADIO UI ----------
with gr.Blocks(title="LipSync + Mejora Facial AMD") as demo:
    gr.Markdown("## 🎥 LipSync + Mejora Facial (AMD ROCm)")

    with gr.Row():
        video_input = gr.Video(label="📹 Video original",height=360)
        audio_input = gr.Audio(label="🎵 Audio", type="filepath")
 
    modelo_lipsync = gr.Radio(
        choices=[
            ("Wav2Lip (rápido / estable)", "wav2lip"),
            ("Wav2Lip GAN (más realista)", "wav2lip_gan"),
        ],
        value="wav2lip",
        label="👄 Modelo de LipSync"
    )

    mejorar_chk = gr.Checkbox(
        label="✨ Mejorar video final (GFPGAN)",
        value=True
    )

    weight_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="GFPGAN Weight")
    alpha_slider = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Alpha (mezcla con frame original)")
    every_n_frames_slider = gr.Slider(1, 10, value=3, step=1, label="Aplicar GFPGAN cada N frames")

    btn = gr.Button("🚀 Procesar")

    video_output = gr.Video(label="✅ Resultado final",height=360)

    btn.click(
        procesar,
        inputs=[
            video_input, 
            audio_input, 
            mejorar_chk, 
            modelo_lipsync, 
            weight_slider, 
            alpha_slider,
            every_n_frames_slider],
        outputs=video_output
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
