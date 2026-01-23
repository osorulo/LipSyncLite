import numpy as np
import hashlib
import cv2
import os
import subprocess
from tqdm import tqdm
import torch
import tempfile
import pickle
from lipsync import audio
from lipsync.helpers import read_frames, get_face_box
from lipsync.models import load_model
from typing import List, Tuple, Union
import gc

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

import threading
import queue


class LipSync:
    """
    Class for lip-syncing videos using the Wav2Lip model.
    """

    # Default parameters
    checkpoint_path: str = ''
    static: bool = False
    fps: float = 25.0
    pads: List[int] = [0, 10, 0, 0]
    wav2lip_batch_size: int = 128
    resize_factor: int = 1
    crop: List[int] = [0, -1, 0, -1]
    box: List[int] = [-1, -1, -1, -1]
    rotate: bool = False
    nosmooth: bool = False
    save_cache: bool = True
    cache_dir: str = tempfile.gettempdir()
    _filepath: str = ''
    img_size: int = 96
    mel_step_size: int = 16
    device: str = 'cpu'
    ffmpeg_loglevel: str = 'verbose'
    model: str = 'wav2lip'
    weight: float = 0.5

    def __init__(self, **kwargs):
        """
        Initializes LipSync with custom parameters.
        """
        device = kwargs.get('device', self.device)
        self.device = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'

        # Update class attributes with provided keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Detector SSD
        proto = "lipsync/checkpoints/deploy.prototxt"
        model_ssd = "lipsync/checkpoints/res10_300x300_ssd_iter_140000.caffemodel"
        self.face_net = cv2.dnn.readNetFromCaffe(proto, model_ssd)

        self._init_enhancers()

    def _init_enhancers(self):
        # ESRGAN para super-resolución
        model_esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.restorer = RealESRGANer(
            scale=2, model_path="lipsync/checkpoints/RealESRGAN_x2plus.pth",
            model=model_esrgan, half=(self.device == 'cuda'), device=self.device
        )
        # GFPGAN para restauración facial
        self.restorer2 = GFPGANer(
            model_path="lipsync/checkpoints/GFPGANv1.4.pth",
            upscale=1, arch="clean", channel_multiplier=2, device=self.device
        )

    @staticmethod
    def get_smoothened_boxes(boxes: np.ndarray, t: int) -> np.ndarray:
        """
        Smoothens bounding boxes over a temporal window.
        """
        for i in range(len(boxes)):
            window_end = min(i + t, len(boxes))
            window = boxes[i:window_end]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def get_video_hash(self, path):
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            hasher.update(f.read(10 * 1024 * 1024)) # 10MB
        hasher.update(str(self.pads).encode())
        return hasher.hexdigest()
    
    def get_cache_filename(self) -> str:
        """
        Generates a filename for caching face detection results.
        """
        v_hash = self.get_video_hash(self._filepath)
        cache_path = os.path.join(self.cache_dir, f"{v_hash}.pk")
        return cache_path

    def get_from_cache(self) -> Union[List, bool]:
        """
        Retrieves face detection results from cache if available.
        """
        if not self.save_cache:
            return False

        cache_filename = self.get_cache_filename()
        if os.path.isfile(cache_filename):
            with open(cache_filename, 'rb') as cached_file:
                return pickle.load(cached_file)

        return False

    def detect_faces_in_frames(self, images, span_por_etapa, progress=None):
        predictions = []
        h, w = images[0].shape[:2]
        total = len(images)

        for i in range(0, total):
            blob = cv2.dnn.blobFromImage(images[i], 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            best_conf = 0
            best_box = None

            for j in range(detections.shape[2]):
                conf = detections[0, 0, j, 2]
                if conf > best_conf and conf > 0.6:
                    box = detections[0, 0, j, 3:7]
                    x1, y1, x2, y2 = (box * np.array([w, h, w, h])).astype(int)
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

            predictions.append(best_box)
            
            if progress is not None:
                self._report_progress(progress, 0.0, span_por_etapa, i / total, f"Face Detection: {i}/{total}")

        return predictions

    def _merge_audio_video(self, audio_file: str, temp_video: str, outfile: str):
        """
        Mezcla audio y video con compresión eficiente H.264.
        """
        # -c:v libx264: Codec estándar de alta compatibilidad.
        # -crf 18: Calidad visualmente sin pérdidas (rango 18-23 es ideal).
        # -preset slow: Mejor compresión a cambio de un poco más de tiempo.
        # -pix_fmt yuv420p: Necesario para que el video se vea en Windows/Móviles.
        
        command = (
            f'ffmpeg -y -i "{temp_video}" -i "{audio_file}" '
            f'-c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p '
            f'-c:a aac -b:a 192k -shortest '
            f'-loglevel {self.ffmpeg_loglevel} "{outfile}"'
        )
        subprocess.run(command, shell=True, check=True)

    def process_face_boxes(self, predictions: List, images: List[np.ndarray]) -> List[List]:
        """
        Process face bounding boxes, apply smoothing, and crop faces.
        """
        pady1, pady2, padx1, padx2 = self.pads
        img_h, img_w = images[0].shape[:2]

        results = []
        for rect, image in zip(predictions, images):
            if rect is None:
                raise ValueError('Face not detected! Ensure all frames contain a face.')
            y1 = max(0, rect[1] - pady1)
            y2 = min(img_h, rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(img_w, rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, t=5)

        cropped_results = []
        for (x1, y1, x2, y2), image in zip(boxes, images):
            face_img = image[int(y1): int(y2), int(x1): int(x2)]
            cropped_results.append([face_img, (int(y1), int(y2), int(x1), int(x2))])

        return cropped_results

    def face_detect(self, images: List[np.ndarray], span, progress=None) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Performs face detection on a list of images.
        """
        cache = self.get_from_cache()
        if cache:
            return cache

        predictions = self.detect_faces_in_frames(images, span, progress)
        cropped_results = self.process_face_boxes(predictions, images)

        # Cache results if enabled
        if self.save_cache:
            with open(self.get_cache_filename(), 'wb') as cached_file:
                pickle.dump(cropped_results, cached_file)

        return cropped_results

    def datagen(self, frames, mels, span, progress=None):

        self._get_face_detections(frames, span, progress)
        
        num_mels = len(mels) 
        batch_size = self.wav2lip_batch_size
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        for i in range(num_mels):
            idx = i % len(frames) if not self.static else 0
            
            face, coords = self._face_det_cache[idx]
            
            face_img = face.copy()
            face_resized = cv2.resize(face_img, (self.img_size, self.img_size))

            img_batch.append(face_resized)
            mel_batch.append(mels[i])
            frame_batch.append(frames[idx].copy())
            coords_batch.append(coords)

            if len(img_batch) >= batch_size:
                yield self._prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            yield self._prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)

    def _get_face_detections(self, frames: List[np.ndarray], span, progress=None) -> List[List]:
        """
        Obtiene o calcula las detecciones de rostro y las guarda en el objeto.
        """
        if self.box[0] == -1:
            results = self.face_detect(frames if not self.static else [frames[0]], span, progress)
        else:
            y1, y2, x1, x2 = self.box
            results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        self._face_det_cache = results 
        return results
        

    def _prepare_batch(
        self,
        img_batch: List[np.ndarray],
        mel_batch: List[np.ndarray],
        frame_batch: List[np.ndarray],
        coords_batch: List[Tuple[int, int, int, int]]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Prepares a batch of images and mel spectrograms for inference.
        """
        img_batch_np = np.asarray(img_batch, dtype=np.uint8)
        mel_batch_np = np.asarray(mel_batch, dtype=np.float32)

        half = self.img_size // 2
        img_masked = img_batch_np.copy()
        img_masked[:, half:] = 0
        img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
        mel_batch_np = mel_batch_np[..., np.newaxis]

        return img_batch_np, mel_batch_np, frame_batch, coords_batch

    @staticmethod
    def create_temp_file(ext: str) -> str:
        """
        Creates a temporary file with a specific extension.
        """
        temp_fd, filename = tempfile.mkstemp()
        os.close(temp_fd)
        return f'{filename}.{ext}'
    
    def create_temp_file(self, ext):
        fd, path = tempfile.mkstemp(suffix=f'.{ext}')
        os.close(fd)
        return path
    
    def sync_wav2lip(self, face: str, audio_file: str, span_por_etapa, progress=None) -> str:
        self._last_face_coords = []
        self._filepath = face
        
        full_frames, fps = self._load_input_face(face)
        audio_file = self._prepare_audio(audio_file)
        mel = self._generate_mel_spectrogram(audio_file)
        
        mel_chunks = self._split_mel_chunks(mel, fps) 
        
        model = self._load_model_for_inference()
        temp_result_avi = self.create_temp_file('avi')
        out = self._prepare_video_writer(temp_result_avi, full_frames[0].shape[:2], fps)

        self._perform_inference(model, full_frames, mel_chunks, out, span_por_etapa, progress=progress)
        out.release()

        coords_path = temp_result_avi + ".coords"
        with open(coords_path, "wb") as f:
            pickle.dump(self._last_face_coords, f)

        return temp_result_avi, audio_file, coords_path

    def sync(self, face_path, audio_path, outfile, mejorar=True, alpha=0.3, mejorarES_chk=False, progress=None):
        steps = 2 + (1 if mejorar else 0) + (1 if mejorarES_chk else 0)
        span = 1.0 / steps

        # 1. Wav2Lip
        raw_video, audio_wav, coords_path = self.sync_wav2lip(face_path, audio_path, span, progress)
        current_video = raw_video

        # 2. GFPGAN (Restauración Facial)
        if mejorar:
            enhanced_face = self.create_temp_file('avi')
            self._enhance_video_gfpgan(current_video, enhanced_face, base=span*2, span=span, alpha=alpha, progress=progress)
            current_video = enhanced_face

        # 3. Real-ESRGAN (Super Resolución de todo el frame)
        if mejorarES_chk:
            upscaled_video = self.create_temp_file('avi')
            self._upscale_video_esrgan(current_video, upscaled_video, progress)
            current_video = upscaled_video

        # 4. Merge Final
        self._merge_audio_video(audio_wav, current_video, outfile)

        for tmp in [raw_video, audio_wav, coords_path]:
            try:
                if os.path.exists(tmp): os.remove(tmp)
            except:
                pass

        return outfile

    def _enhance_video_gfpgan(self, in_video, out_video, base, span, weight=0.5, mejorarES_chk=True, batch_size=12, alpha=0.3,progress=None):

        with open(in_video + ".coords", "rb") as f:
            face_coords = pickle.load(f)
        
        frame_idx = 0

        cap = cv2.VideoCapture(in_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # 1. COLA DE LECTURA (Buffer de frames)
        # Permite que la CPU lea frames mientras la GPU procesa otros.

        frame_queue = queue.Queue(maxsize=batch_size * 2)

        def reader_thread():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_queue.put(frame) # Se detiene si la cola está llena
            frame_queue.put(None) # Señal de fin de video
            cap.release()

        # Iniciar el hilo de lectura
        
        threading.Thread(target=reader_thread, daemon=True).start()

        processed_count = 0
        frames_buffer = []

        while True:
            # 2. OBTENER FRAME DE LA COLA
            frame = frame_queue.get()
            
            if frame is not None:
                frames_buffer.append(frame)

            # 3. PROCESAMIENTO CUANDO EL BATCH ESTÁ LLENO (O FINALIZÓ EL VIDEO)
            if len(frames_buffer) >= batch_size or (frame is None and len(frames_buffer) > 0):
                for i in range(len(frames_buffer)):
                    current_frame = frames_buffer[i]
                    
                    y1, y2, x1, x2 = face_coords[frame_idx]
                    frame_idx += 1

                    roi = current_frame[y1:y2, x1:x2].copy()
                    if roi.size == 0:
                        out.write(current_frame)
                        continue
                            
                    if roi.size > 0:
                        # MEJORA GFPGAN
                        _, _, enhanced = self.restorer2.enhance(
                                roi, 
                                weight=weight
                            )

                        # MÁSCARA SEAMLESS (Fusión suave)
                        # 1. Crear máscara base negra
                        fh, fw = enhanced.shape[:2]
                        mask = np.zeros((fh, fw, 3), dtype=np.float32)

                        # 3. Dibujar la mitad SUPERIOR con el alpha elegido (Ángulos 180 a 360)
                        cv2.ellipse(mask, (fw//2, fh//2), (int(fw*0.35), int(fh*0.45)), 0, 180, 360, (alpha, alpha, alpha), -1)

                        # 4. Dibujar la mitad INFERIOR al 100% (Ángulos 0 a 180)
                        cv2.ellipse(mask, (fw//2, fh//2), (int(fw*0.35), int(fh*0.45)), 0, 0, 180, (1.0, 1.0, 1.0), -1)

                        # 5. Suavizar TODA la máscara para que la transición entre el 0.3 y el 1.0 no sea brusca
                        blur_size = int(fw * 0.15) 
                        if blur_size % 2 == 0: blur_size += 1
                        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

                        # Mezcla final en float32 para evitar artefactos
                        img_float = roi.astype(np.float32)
                        enh_float = enhanced.astype(np.float32)
                        merged = (enh_float * mask) + (img_float * (1.0 - mask))
                                
                        current_frame[y1:y2, x1:x2] = np.clip(merged, 0, 255).astype(np.uint8)
                        # SUPER RESOLUCIÓN ESRGAN
                        #if mejorarES_chk:
                            #current_frame, _ = self.restorer.enhance(current_frame, outscale=2)
                            #enhanced = cv2.resize(enhanced, (x2 - x1, y2 - y1))
                    out.write(current_frame)
                    processed_count += 1

                    if progress is not None and processed_count % 10 == 0:
                        self._report_progress(
                            progress, base, span, processed_count / total, 
                            desc=f"GPU Face Restoration {processed_count}/{total}"
                        )

                frames_buffer = [] 

            if frame is None: 
                break

        out.release()
        # 4. LIMPIEZA DE VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _upscale_video_esrgan(self, in_video, out_video, progress=None):
        cap = cv2.VideoCapture(in_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)

        out = cv2.VideoWriter(
            out_video,
            cv2.VideoWriter_fourcc(*'DIVX'),
            fps,
            (w, h)
        )

        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            up, _ = self.restorer.enhance(frame, outscale=2)
            out.write(up)

            if progress is not None and i % 2 == 0:
                self._report_progress(
                    progress,
                    base=0.85,
                    span=0.15,
                    value=i / total,
                    desc=f"Upscale {i}/{total}"
                )
            i += 1

        cap.release()
        out.release()

         # 4. LIMPIEZA DE VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        

    def _report_progress(self, progress, base, span, value, desc):
        if progress is None:
            return
        try:
            progress(base + span * value, desc=desc)
        except Exception:
            pass

    def _load_input_face(self, face: str) -> Tuple[List[np.ndarray], float]:
        """
        Loads the input face (video or image) and returns frames and fps.
        """
        if not os.path.isfile(face):
            raise ValueError('face argument must be a valid file path.')

        if face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            self.static = True
            full_frames = [cv2.imread(face)]
            fps = self.fps
        else:
            full_frames, fps = read_frames(face)

        return full_frames, fps

    def _prepare_audio(self, audio_file: str) -> str:
        """
        Prepares (extracts) raw audio if not in .wav format.
        """
        if not audio_file.endswith('.wav'):
            wav_filename = self.create_temp_file('wav')
            command = (
                f'ffmpeg -y -i "{audio_file}" -strict -2 "{wav_filename}" '
                f'-loglevel {self.ffmpeg_loglevel}'
            )
            subprocess.run(command, shell=True, check=True)
            audio_file = wav_filename
        return audio_file

    @staticmethod
    def _generate_mel_spectrogram(audio_file: str) -> np.ndarray:
        """
        Generates the mel spectrogram from the given audio file.
        """
        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel).any():
            raise ValueError('Mel contains NaN! Add a small epsilon to the audio and try again.')
        return mel

    def _load_model_for_inference(self) -> torch.nn.Module:
        """
        Loads the lip sync model for inference.
        """
        model = load_model(self.model, self.device, self.checkpoint_path)
        return model

    @staticmethod
    def _prepare_video_writer(filename: str, frame_shape: Tuple[int, int], fps: float) -> cv2.VideoWriter:
        """
        Prepares the VideoWriter for output.
        """
        frame_h, frame_w = frame_shape
        return cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'DIVX'),
            fps,
            (frame_w, frame_h),
        )

    def _perform_inference(
        self,
        model: torch.nn.Module,
        full_frames: List[np.ndarray],
        mel_chunks: List[np.ndarray],
        out: cv2.VideoWriter,
        span_por_etapa,
        progress=None
    ):
        """
        Runs the inference loop: generates data, passes through model, and writes results.
        """
        data_generator = self.datagen(full_frames, mel_chunks, span_por_etapa, progress)
        proximo_base = span_por_etapa
        total_frames = len(mel_chunks)
        processed_counter = [0]

        for (img_batch_np, mel_batch_np, frames, coords) in data_generator:
            img_batch_t = torch.FloatTensor(np.transpose(img_batch_np, (0, 3, 1, 2))).to(self.device)
            mel_batch_t = torch.FloatTensor(np.transpose(mel_batch_np, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = model(mel_batch_t, img_batch_t)

            self._write_predicted_frames(pred, frames, coords, out, total_frames,processed_counter,proximo_base,span_por_etapa,progress=progress)

    def _write_predicted_frames(
        self,
        pred: torch.Tensor,
        frames: List[np.ndarray],
        coords: List[Tuple[int, int, int, int]],
        out: cv2.VideoWriter,
        total_frames,
        counter,
        proximo_base,
        span_por_etapa,
        progress=None
    ):
        """
        Writes the predicted frames using a seamless mask to avoid the "box" effect.
        """
        pred_np = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
        
        for p, f, c in zip(pred_np, frames, coords):
            self._last_face_coords.append(c)
            y1, y2, x1, x2 = c
            h, w = y2 - y1, x2 - x1
            if h <= 0 or w <= 0: continue

            p_resized = cv2.resize(p, (w, h))
            
            # --- LÓGICA DE MÁSCARA SEAMLESS PARA WAV2LIP ---
            # Creamos una máscara para el recorte de la boca/cara
            mask = np.zeros((h, w, 3), dtype=np.float32)
            
            # Dibujamos una elipse que cubra la parte central (donde Wav2Lip es fuerte)
            # y se desvanezca hacia los bordes del cuadro detectado
            cv2.ellipse(mask, (w//2, h//2), (int(w*0.45), int(h*0.48)), 0, 0, 360, (1.0, 1.0, 1.0), -1)
            
            # Aplicamos un desenfoque fuerte a la máscara para suavizar la transición
            blur_size = int(w * 0.25) | 1 # 25% del ancho, siempre impar
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

            # Composición usando Alpha Blending
            img_original = f[y1:y2, x1:x2].astype(np.float32)
            img_predicha = p_resized.astype(np.float32)
            
            # Mezcla: (Predicción * Máscara) + (Original * (1 - Máscara))
            merged = (img_predicha * mask) + (img_original * (1.0 - mask))
            f[y1:y2, x1:x2] = np.clip(merged, 0, 255).astype(np.uint8)
            # -----------------------------------------------

            out.write(f)
            counter[0] += 1
            
            if progress is not None and counter[0] % 5 == 0:
                self._report_progress(
                    progress,
                    base=proximo_base,
                    span=span_por_etapa,
                    value=counter[0] / total_frames,
                    desc=f"Sincronizando... {counter[0]}/{total_frames}"
                )

    def _split_mel_chunks(self, mel: np.ndarray, fps: float) -> List[np.ndarray]:
        """
        Divide el mel en trozos. El número de trozos resultantes 
        determina la duración final del video.
        """
        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            end_idx = start_idx + self.mel_step_size
            if end_idx > mel.shape[1]:
                break
            mel_chunks.append(mel[:, start_idx:end_idx])
            i += 1
        return mel_chunks