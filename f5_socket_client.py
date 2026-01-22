import socket
import json
import os

HOST = "127.0.0.1"
PORT = 5555

def generar_audio_f5_socket(texto, ref_audio=None, speed=1.0, cfg_strength=2.0):
    """
    Genera audio usando F5-TTS vía socket.
    
    Parámetros:
        texto (str): Texto a sintetizar.
        ref_audio (str, opcional): Archivo de voz de referencia en VOCES.
        speed (float, opcional): Velocidad de la voz (1.0 = normal).
        cfg_strength (float, opcional): Fuerza de pausas/silencios.
    
    Retorna:
        str: Ruta al archivo WAV generado.
    """
    payload = {
        "text": texto,
        "speed": speed,
        "cfg_strength": cfg_strength
    }

    if ref_audio:
        payload["ref_audio"] = ref_audio

    # Conexión al socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(json.dumps(payload).encode("utf-8"))

        # Recibimos datos en chunks grandes
        data = b""
        while True:
            chunk = s.recv(131072)
            if not chunk:
                break
            data += chunk

    resp = json.loads(data.decode("utf-8"))
    if resp["status"] != "ok":
        raise RuntimeError(f"Error F5-TTS: {resp['error']}")

    return resp["wav"]


