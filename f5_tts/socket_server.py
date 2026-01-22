import socket
import json
import os
import tempfile
import traceback
import time

import torch
import soundfile as sf
import numpy as np

from f5_tts.infer.infer_cli import model_cls, model_cfg
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process

from huggingface_hub import snapshot_download
import os

# 1. Solo descargamos los archivos necesarios del repo español
# Ignoramos el .pt gigante para no llenar el Drive/Colab
repo_path = snapshot_download(
    repo_id="jpgallegoar/F5-Spanish", 
    ignore_patterns=["*.pt"] 
)

# 2. Definimos las rutas exactas basándonos en los archivos que acabas de descargar
MODEL_CKPT = os.path.join(repo_path, "model_1250000.safetensors")
VOCAB_PATH = os.path.join(repo_path, "vocab.txt")
CONFIG_PATH = os.path.join(repo_path, "transformer_config.yaml")

print(f"--- MODELO ESPAÑOL CARGADO ---")
print(f"Checkpoint: {MODEL_CKPT}")
print(f"Vocab: {VOCAB_PATH}")
# ================= CONFIG =================
HOST = "127.0.0.1"
PORT = 5555

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCES_DIR = os.path.join(BASE_DIR, "..", "voces")
DEFAULT_REF_AUDIO = os.path.join(VOCES_DIR, "Alejandro.wav")
DEFAULT_REF_TEXT = "La pantalla brilló en medio de la noche. Algoritmos invisibles decidían qué ver, qué leer y qué ignorar. Mientras tanto, alguien dudaba si la máquina aprendía de las personas o si las personas ya pensaban como máquinas."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000

ASR_PIPE = None

# =========================================

class F5TTSServer:
    def __init__(self):
        print("🔹 Cargando modelo F5-TTS...")
        self.model = load_model(
            model_cls=model_cls,
            model_cfg=model_cfg,
            ckpt_path=MODEL_CKPT,
            vocab_file=VOCAB_PATH,
            device=DEVICE,
        )
        print("🔹 Cargando vocoder...")
        self.vocoder = load_vocoder(device=DEVICE)
        print("✅ Modelo y vocoder cargados")

    def init_asr(self):
        global ASR_PIPE
        if ASR_PIPE is None:
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            ASR_PIPE = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                device=DEVICE,
                torch_dtype=dtype,
            )

    def get_ref_text(self, ref_audio_path):
        global ASR_PIPE
        if ASR_PIPE is None:
            self.init_asr()
        result = ASR_PIPE(ref_audio_path, chunk_length_s=30, batch_size=128)
        return result["text"].strip()

    def synthesize(self, ref_audio, ref_text, text, out_path, speed=1.0, cfg_strength=2.0):
        wav, sr, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            model_obj=self.model,
            vocoder=self.vocoder,
            device=DEVICE
        )
        sf.write(out_path, wav, sr)

    def handle_client(self, conn):
        try:
            payload = json.loads(conn.recv(131072).decode("utf-8"))

            text = payload.get("text", "").strip()
            ref_audio = payload.get("ref_audio", "Alejandro.mp3").strip()
            ref_text = payload.get("ref_text", "").strip()
            speed = float(payload.get("speed", 1.0))
            cfg_strength = float(payload.get("cfg_strength", 2.0))

            if len(text) < 3:
                raise ValueError("Texto demasiado corto")

            ref_audio_path = os.path.join(VOCES_DIR, ref_audio)
            if not os.path.exists(ref_audio_path):
                raise FileNotFoundError(f"ref_audio no existe: {ref_audio_path}")

            if not ref_text:
                #print("Generando ref_text automáticamente con Whisper...")
                #ref_text = self.get_ref_text(ref_audio_path)
                ref_text = DEFAULT_REF_TEXT

            os.makedirs("audio_out", exist_ok=True)
            out_wav = os.path.join("audio_out", f"tts_{int(time.time()*1000)}.wav")

            self.synthesize(ref_audio_path, ref_text, text, out_wav, speed, cfg_strength)

            conn.sendall(json.dumps({"status": "ok", "wav": out_wav}).encode("utf-8"))

        except Exception as e:
            conn.sendall(json.dumps({
                "status": "error",
                "error": str(e),
                "trace": traceback.format_exc()
            }).encode("utf-8"))
        finally:
            conn.close()

    def serve_forever(self):
        print(f"🚀 F5-TTS socket escuchando en {HOST}:{PORT}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen(5)
            while True:
                conn, _ = s.accept()
                self.handle_client(conn)


if __name__ == "__main__":
    F5TTSServer().serve_forever()
