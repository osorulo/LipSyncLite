import os
import sys
import argparse

# ---------------- CONFIG ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--colab", action="store_true", help="Usar Google Drive para cache persistente")
parser.add_argument("--cache_dir", type=str, default=None, help="Ruta personalizada para cache")
parser.add_argument("--workdir", type=str, default=None, help="Ruta temporal de trabajo")
args, _ = parser.parse_known_args()
# 🔥 CLAVE: limpiar argv para scripts hijos
sys.argv = sys.argv[:1]

WORKDIR = args.workdir or "gradio_tmp"
CACHE_DIR = args.cache_dir or "lipsync/cache"
VOCES_DIR = os.path.join(os.path.dirname(__file__), "voces")

if args.colab:
    BASE_CACHE = "/content/drive/MyDrive/LipSyncLite"
    CACHE_DIR = os.path.join(BASE_CACHE, "cache")
    WORKDIR = os.path.join(BASE_CACHE, "tmp")
    VOCES_DIR = os.path.join(BASE_CACHE, "voces")

print("📂 WORKDIR =", WORKDIR)
print("📦 CACHE_DIR =", CACHE_DIR)
print("📦 VOCES_DIR =", VOCES_DIR)

os.makedirs(WORKDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VOCES_DIR, exist_ok=True)

import shutil
import threading
import atexit
import gradio as gr 
from lipsync import LipSync
# de las funciones locales importamos lo necesario
from f5_socket_client import generar_audio_f5_socket
from f5_tts.socket_server import F5TTSServer

# ---------------- SOCKET SERVER ----------------
f5_socket_server = None
f5_thread = None

def start_f5_socket():
    global f5_socket_server, f5_thread
    try:
        f5_socket_server = F5TTSServer()
        def run_server():
            f5_socket_server.serve_forever()
        f5_thread = threading.Thread(target=run_server, daemon=True)
        f5_thread.start()
        print("✅ F5-TTS socket server iniciado")
    except Exception as e:
        print(f"⚠️ No se pudo iniciar F5-TTS Server: {e}")

def stop_f5_socket():
    global f5_socket_server
    if f5_socket_server:
        print("🛑 Cerrando F5-TTS socket server...")
        f5_socket_server = None

atexit.register(stop_f5_socket)

# ---------------- LIPSYNC HELPER ----------------
def crear_lipsync(modelo, w):
    ckpt = (
        "lipsync/checkpoints/wav2lip_gan.pth"
        if modelo == "wav2lip_gan"
        else "lipsync/checkpoints/wav2lip.pth"
    )
    return LipSync(
        checkpoint_path=ckpt,
        device="cuda",
        nosmooth=False,
        cache_dir=CACHE_DIR,
        save_cache=True,
        weight=w
    )

# ---------------- F5 AUDIO ----------------
def generar_audio_ui(texto, ref_audio, speed):
    if not texto or len(texto.strip()) < 3:
        raise gr.Error("El texto es demasiado corto")

    ref_audio_path = os.path.join(VOCES_DIR, ref_audio)
    if not os.path.exists(ref_audio_path):
        raise gr.Error(f"No existe el archivo de voz: {ref_audio}")
    
    audio_path = generar_audio_f5_socket(
        texto,
        ref_audio=ref_audio_path,
        speed=speed,
        cfg_strength=2.0
    )

    return audio_path, gr.update(value=audio_path, visible=True)

# ---------------- PROCESAR ----------------
import uuid

def procesar(video_file, audio_file, mejorar, modelo, weight,
             n_slider, a_slider, mejorarES_chk, calidez, progress=gr.Progress()):
    if video_file is None or audio_file is None:
        raise gr.Error("Se requiere video y audio.")

    # Crear carpeta si no existe, pero NO borrarla (para evitar OSError Errno 39)
    os.makedirs(WORKDIR, exist_ok=True)
    
    # ID único para esta ejecución
    run_id = str(uuid.uuid4())[:8]
    
    video_in = os.path.join(WORKDIR, f"in_{run_id}.mp4")
    audio_in = os.path.join(WORKDIR, f"audio_{run_id}.wav")
    video_out = os.path.join(WORKDIR, f"out_{run_id}.mp4")

    # Copiar archivos
    shutil.copy(video_file, video_in)
    shutil.copy(audio_file, audio_in)

    lip = crear_lipsync(modelo, weight)
    lip.sync(video_in, audio_in, video_out, mejorar, a_slider, mejorarES_chk, 
             calidez=calidez, progress=progress) # <-- Pasar calidez aquí
    return video_out

# ---------------- UI ----------------
with gr.Blocks(title="LipSync Pro AMD", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 LipSync + TTS Español + Mejoras Visuales")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="📹 Video Original")
            modelo_lipsync = gr.Radio(
                choices=[("Wav2Lip GAN (Mejor boca)", "wav2lip_gan"), ("Wav2Lip (Más estable)", "wav2lip")],
                value="wav2lip_gan",
                label="👄 Modelo"
            )
        
        with gr.Column():
            tts_text = gr.Textbox(label="📝 Texto", lines=3, value="Hola, ¿cómo estás hoy?")
            ref_audio_input = gr.Dropdown(
                label="🎵 Voz de Referencia",
                choices=os.listdir(VOCES_DIR) if os.path.exists(VOCES_DIR) else []
            )
            speed_slider = gr.Slider(0.5, 2.0, value=1.0, label="Velocidad")
            btn_gen_audio = gr.Button("🎙️ Generar Audio TTS")
            audio_upload = gr.Audio(label="🎧 Audio Final (Generado o Subido)", type="filepath")
            audio_download = gr.File(label="⬇️ Descargar Audio", visible=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✨ Post-Procesamiento")
            mejorar_chk = gr.Checkbox(label="GFPGAN (Restaurar Rostro)", value=True)
            mejorarES_chk = gr.Checkbox(label="Real-ESRGAN (Upscale x2)", value=False)
        with gr.Column():
            weight_slider = gr.Slider(0, 1, value=0.5, label="Fuerza GFPGAN")
            a_slider = gr.Slider(0, 1, value=0.15, label="Mezcla (Blend con original)")
            calidez_slider = gr.Slider(0, 0.5, value=0.1, step=0.01, label="Eliminar Azul (Calidez)")

    btn_procesar = gr.Button("🚀 EMPEZAR LIP-SYNC", variant="primary")
    video_output = gr.Video(label="✅ Resultado Final")

    # Eventos
    btn_gen_audio.click(
        generar_audio_ui,
        inputs=[tts_text, ref_audio_input, speed_slider],
        outputs=[audio_upload, audio_download]
    )

    btn_procesar.click(
        procesar,
        inputs=[
            video_input, audio_upload, mejorar_chk, modelo_lipsync,
            weight_slider, gr.Number(value=1, visible=False),
            a_slider, mejorarES_chk, calidez_slider # <-- Añadido aquí
        ],
        outputs=video_output
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    start_f5_socket()
    demo.queue().launch(server_name="0.0.0.0", server_port=7861, share=True)
