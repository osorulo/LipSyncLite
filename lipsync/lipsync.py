import numpy as np
import cv2
import os
import subprocess
import torch
import tempfile
import pickle
import hashlib
from tqdm import tqdm
from typing import List, Tuple

from lipsync import audio
from lipsync.helpers import read_frames
from lipsync.models import load_model
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class LipSync:
    def __init__(self, **kwargs):
        self.checkpoint_path = kwargs.get('checkpoint_path', '')
        self.static = kwargs.get('static', False)
        self.fps = kwargs.get('fps', 25.0)
        self.pads = kwargs.get('pads', [0, 10, 0, 0]) 
        self.wav2lip_batch_size = kwargs.get('wav2lip_batch_size', 128)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = 96
        self.weight = kwargs.get('weight', 0.5)
        self.box = kwargs.get('box', [-1, -1, -1, -1])
        self.cache_dir = kwargs.get('cache_dir', 'lipsync/cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Detector SSD
        proto = "lipsync/checkpoints/deploy.prototxt"
        model_ssd = "lipsync/checkpoints/res10_300x300_ssd_iter_140000.caffemodel"
        self.face_net = cv2.dnn.readNetFromCaffe(proto, model_ssd)
        self._init_enhancers()

    def _init_enhancers(self):
        model_esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.restorer = RealESRGANer(
            scale=2, model_path="lipsync/checkpoints/RealESRGAN_x2plus.pth",
            model=model_esrgan, half=(self.device == 'cuda'), device=self.device
        )
        self.restorer2 = GFPGANer(
            model_path="lipsync/checkpoints/GFPGANv1.4.pth",
            upscale=1, arch="clean", channel_multiplier=2, device=self.device
        )

    # --- CACHÉ POR CONTENIDO MD5 ---
    def get_video_hash(self, path):
        """Calcula un hash único basado en los primeros 10MB del video (rápido)."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            chunk = f.read(1024 * 1024 * 10) # 10MB son suficientes para identificarlo
            hasher.update(chunk)
        return hasher.hexdigest()

    def _get_face_detections(self, frames, face_path):
        v_hash = self.get_video_hash(face_path)
        cache_path = os.path.join(self.cache_dir, f"{v_hash}.pk")
        
        if os.path.exists(cache_path):
            print(f"📦 CACHÉ ENCONTRADO (Hash: {v_hash}). Saltando detección facial.")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"🔍 NO HAY CACHÉ (Hash: {v_hash}). Detectando rostros...")
        preds = self.detect_faces_in_frames(frames)
        
        # Lógica de rectángulos y padding
        valid_preds = [p for p in preds if p is not None]
        if not valid_preds: raise ValueError("No se detectó rostro.")
        last_known = valid_preds[0]
        h_img, w_img = frames[0].shape[:2]
        results = []

        for i, p in enumerate(preds):
            curr = p if p is not None else last_known
            x1, y1, x2, y2 = curr
            y1, y2 = max(0, y1 - self.pads[0]), min(h_img, y2 + self.pads[1])
            x1, x2 = max(0, x1 - self.pads[2]), min(w_img, x2 + self.pads[3])
            results.append([frames[i][y1:y2, x1:x2], (y1, y2, x1, x2)])
            last_known = curr

        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        return results

    # --- INFERENCIA ---
    def detect_faces_in_frames(self, frames):
        predictions = []
        h, w = frames[0].shape[:2]
        for frame in tqdm(frames, desc="Detectando"):
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            best_conf, best_box = 0.5, None
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > best_conf:
                    box = detections[0, 0, i, 3:7]
                    best_box = (int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h))
                    best_conf = conf
            predictions.append(best_box)
        return predictions

    def datagen(self, frames, mels, face_path):
        face_det_results = self._get_face_detections(frames, face_path)
        for i, m in enumerate(mels):
            idx = 0 if self.static else (i % len(frames))
            face, coords = face_det_results[idx]
            face_resized = cv2.resize(face, (self.img_size, self.img_size))
            yield face_resized, m, frames[idx], coords

    def sync(self, face_path, audio_path, outfile, mejorar=True, a_slider=0.15, mejorarES_chk=False,calidez=0.1, progress=None):
        full_frames, fps = read_frames(face_path)
        audio_wav = self._prepare_audio(audio_path)
        mel = audio.melspectrogram(audio.load_wav(audio_wav, 16000))
        mel_chunks = self._split_mel_chunks(mel, fps)
        
        model = load_model('wav2lip', self.device, self.checkpoint_path)
        temp_avi = self._create_temp_file('avi')
        out = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (full_frames[0].shape[1], full_frames[0].shape[0]))

        # Lógica de Batch simplificada para estabilidad
        total = len(mel_chunks)
        gen = self.datagen(full_frames, mel_chunks, face_path)
        
        for counter, (face_resized, m, full_f, coords) in enumerate(gen):
            img_t = torch.FloatTensor(np.transpose(self._mask_img(face_resized), (2, 0, 1))).unsqueeze(0).to(self.device)
            mel_t = torch.FloatTensor(m).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = model(mel_t, img_t)

            pred_np = (pred.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            final_f = self._post_process(pred_np, full_f, coords, mejorar, a_slider, mejorarES_chk, calidez)
            out.write(final_f)

            if progress is not None and counter % 5 == 0:
                progress(counter / total, desc=f"Sincronizando: {counter}/{total}")

        out.release()
        self._merge_audio_video(audio_wav, temp_avi, outfile)
        return outfile

    def _mask_img(self, img):
        img_masked = img.copy()
        img_masked[self.img_size//2:] = 0
        return np.concatenate((img_masked, img), axis=2) / 255.0

    def _post_process(self, p_img, full_f, coords, mejorar, a_slider, mejorarES_chk, calidez=0.1):
        y1, y2, x1, x2 = coords
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0: return full_f
        
        p_resized = cv2.resize(p_img, (w, h))
        roi = full_f[y1:y2, x1:x2].copy()

        if mejorar:
            # 1. Aplicamos GFPGAN sin forzar el centrado excesivo
            _, _, enhanced = self.restorer2.enhance(p_resized, 
                                                    has_aligned=False, 
                                                    only_center_face=False, 
                                                    paste_back=True,
                                                    weight=self.weight)
            
            # 2. Mezclamos con el ROI original ANTES de la máscara para suavizar la textura
            # Esto evita que el cambio de nitidez sea demasiado brusco
            enhanced = cv2.addWeighted(enhanced, 1.0 - a_slider, roi, a_slider, 0)
        else:
            enhanced = p_resized

        if mejorarES_chk:
            enhanced, _ = self.restorer.enhance(enhanced, outscale=2)
            enhanced = cv2.resize(enhanced, (w, h))

        # 3. MÁSCARA SEAMLESS (Sin recuadro)
        # En lugar de un óvalo rígido, creamos un degradado muy suave
        mask = np.zeros((h, w, 3), dtype=np.float32)
        
        # Dibujamos un elipse blanca (1.0) sobre fondo negro (0.0)
        # Achicamos un poco el radio (0.35 y 0.45) para que no toque los bordes del recorte
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.35), int(h*0.45)), 0, 0, 360, (1.0, 1.0, 1.0), -1)
        
        # El desenfoque (GaussianBlur) es lo que quita el "recuadro"
        # Un valor de 25-35% del ancho del rostro suele ser ideal
        blur_size = int(w * 0.3) | 1 # Asegurar que sea impar
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        # 4. COMPOSICIÓN FINAL
        # Convertimos a float para una mezcla matemática perfecta y evitar el oscurecimiento
        img_float = full_f[y1:y2, x1:x2].astype(np.float32)
        enh_float = enhanced.astype(np.float32)
        
        # Fórmula: (Imagen Mejorada * Máscara) + (Imagen Original * Inversa de la Máscara)
        merged = (enh_float * mask) + (img_float * (1.0 - mask))
        
        full_f[y1:y2, x1:x2] = np.clip(merged, 0, 255).astype(np.uint8)

        return full_f

    def _prepare_audio(self, a_file):
        if not a_file.endswith('.wav'):
            temp = self._create_temp_file('wav')
            subprocess.run(f'ffmpeg -y -i "{a_file}" -loglevel error "{temp}"', shell=True)
            return temp
        return a_file

    def _split_mel_chunks(self, mel, fps):
        chunks = []
        step, mult = 16, 80.0 / fps
        num_f = int(np.ceil(mel.shape[1] / mult))
        for i in range(num_f):
            start = int(i * mult)
            end = start + step
            if end <= mel.shape[1]: chunks.append(mel[:, start:end])
            elif mel.shape[1] >= step: chunks.append(mel[:, -step:])
            else: chunks.append(np.pad(mel, ((0, 0), (0, step - mel.shape[1])), mode='constant'))
        return chunks

    def _create_temp_file(self, ext):
        fd, path = tempfile.mkstemp(suffix=f'.{ext}')
        os.close(fd)
        return path

    def _merge_audio_video(self, audio, video, outfile):
        subprocess.run(f'ffmpeg -y -i "{audio}" -i "{video}" -c:v libx264 -q:v 2 -loglevel error "{outfile}"', shell=True)