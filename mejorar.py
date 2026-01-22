import cv2
import torch
import numpy as np
from gfpgan import GFPGANer
from tqdm import tqdm
import subprocess

cv2.setUseOptimized(True)
cv2.setNumThreads(4)
#torch.backends.cudnn.benchmark = True


# ------------------ COLOR MATCH RÁPIDO ------------------
def match_color_fast(src, ref):
    src = src.astype(np.float32)
    ref = ref.astype(np.float32)

    for c in range(3):
        s_mean, s_std = src[..., c].mean(), src[..., c].std() + 1e-6
        r_mean, r_std = ref[..., c].mean(), ref[..., c].std()
        src[..., c] = (src[..., c] - s_mean) * (r_std / s_std) + r_mean

    return np.clip(src, 0, 255).astype(np.uint8)


# ------------------ FUNCIÓN PRINCIPAL ------------------
def procesar_amd_rocm(
    video_borroso,
    audio_original,
    output_final,
    weight=0.25,
    every_n_frames=1,
    alpha=0.9,
):
    print("🔥 ULTRA MODE – AMD ROCm GFPGAN")

    device = torch.device("cuda")

    restorer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        device=device
    )

    # --------- DETECTOR (1 sola vez) ---------
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    cap = cv2.VideoCapture(video_borroso)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("No se pudo leer el video")

    blob = cv2.dnn.blobFromImage(frame0, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()

    box = det[0, 0, 0, 3:7] * np.array([w, h, w, h])
    x1, y1, x2, y2 = box.astype(int)

    x1, y1 = max(0, x1 - 40), max(0, y1 - 60)
    x2, y2 = min(w, x2 + 40), min(h, y2 + 60)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --------- FFMPEG ---------
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-i", audio_original,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_final
    ], stdin=subprocess.PIPE)

    last_enhanced = None
    last_crop = None

    pbar = tqdm(total=total)

    with torch.inference_mode():
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            face = frame[y1:y2, x1:x2]

            if i % every_n_frames == 0 or last_enhanced is None:
                _, _, enhanced = restorer.enhance(face, weight=weight)
                enhanced = match_color_fast(enhanced, face)

                if last_enhanced is not None:
                    enhanced = (
                        0.7 * enhanced +
                        0.3 * last_enhanced
                    ).astype(np.uint8)

                last_enhanced = enhanced
                last_crop = face.copy()
            else:
                enhanced = last_enhanced

            # alpha dinámico según movimiento
            motion = np.mean(np.abs(face.astype(np.int16) - last_crop.astype(np.int16)))
            alpha_dyn = np.clip(0.6 + motion / 50, 0.6, 0.85)

            roi = frame[y1:y2, x1:x2]
            roi[:] = (
                alpha_dyn * enhanced +
                (1 - alpha_dyn) * roi
            ).astype(np.uint8)

            ffmpeg.stdin.write(frame.tobytes())
            pbar.update(1)

    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    pbar.close()

    print(f"✅ FINALIZADO ULTRA RÁPIDO → {output_final}")
