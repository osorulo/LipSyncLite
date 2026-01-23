# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys

sys.path.append(f"../../{os.path.dirname(os.path.abspath(__file__))}/third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

_ref_audio_cache = {}

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

import re

def chunk_text_by_period(text, max_chars=250):
    """
    Divide el texto SOLO por final de frase (. ! ?).
    Nunca corta frases, aunque se pase de max_chars.
    Nunca genera chunks de una sola palabra.
    """

    # 1️⃣ separar en frases reales
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Si aún no hay nada, empezar chunk
        if not current:
            current = sentence
            continue

        # Si añadir la frase no supera max_chars → juntar
        if len((current + " " + sentence).encode("utf-8")) <= max_chars:
            current = current + " " + sentence
        else:
            # Guardar chunk actual
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    # 2️⃣ Seguridad extra: nunca devolver chunks de 1 palabra
    fixed_chunks = []
    buffer = ""

    for chunk in chunks:
        if len(chunk.split()) == 1:
            buffer = (buffer + " " + chunk).strip()
        else:
            if buffer:
                fixed_chunks.append(buffer)
                buffer = ""
            fixed_chunks.append(chunk)

    if buffer:
        fixed_chunks.append(buffer)

    return fixed_chunks


def split_long_sentence(sentence, max_frames, ref_audio_len, ref_text, speed=1.0):
    parts = []
    current = ""

    for word in sentence.split():
        test = (current + " " + word).strip()
        est_duration = ref_audio_len + int(
            ref_audio_len
            / max(1, len(ref_text.encode("utf-8")))
            * len(test.encode("utf-8"))
            / speed
        )

        if est_duration <= max_frames:
            current = test
        else:
            if current:
                parts.append(current)
            current = word

    if current:
        parts.append(current)

    return parts

# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device):
    if vocoder_name == "vocos":
        if is_local:
            print(f"Load vocos from local path {local_path}")
            vocoder = Vocos.from_hparams(f"{local_path}/config.yaml")
            state_dict = torch.load(f"{local_path}/pytorch_model.bin", map_location="cpu")
            vocoder.load_state_dict(state_dict)
            vocoder = vocoder.eval().to(device)
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device=device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16 if device == "cuda" and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16 if device == "cuda" and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device)


# load model for inference
def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("tokenizer : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    model.eval()

    # 🔥 AQUÍ VA torch.compile
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile ACTIVADO")
    except Exception as e:
        print("torch.compile NO disponible:", e)

    return model



def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


# In utils_infer.py

def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device=device):
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 15000:
                aseg = aseg[:15000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache
    if audio_hash in _ref_audio_cache:
        # Use cached reference text
        ref_text = _ref_audio_cache[audio_hash]
    else:
        if not ref_text.strip():
            global asr_pipe
            if asr_pipe is None:
                initialize_asr_pipeline(device=device)
            
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = asr_pipe(
                ref_audio,
                chunk_length_s=30,
                batch_size=128,
                generate_kwargs={"task": "transcribe"},
                return_timestamps=False,
            )["text"].strip()
        else:
            show_info("Using custom reference text...")
        # Cache the transcribed text
        _ref_audio_cache[audio_hash] = ref_text

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    socket_callback=None, # 👈 AÑADIMOS ESTO
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    
    gen_text_batches = chunk_text_by_period(
        gen_text,
        max_chars=180
    )

    total_batches = len(gen_text_batches)
    show_info(f"Generating audio in {total_batches} batches...")

    # Creamos una versión personalizada del proceso que avise al socket
    def progress_wrapper(iterable, total=None, **kwargs):
        pbar = progress(iterable, total=total, **kwargs)
        for i, item in enumerate(pbar):
            yield item
            # 👈 AQUÍ MANDAMOS EL PROGRESO REAL
            if socket_callback:
                # Calculamos el % basado en el índice i
                pct = (i + 1) / total_batches
                socket_callback(pct, i + 1, total_batches)
        return

    return infer_batch_process_batched(
        (audio, sr),
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        mel_spec_type=mel_spec_type,
        progress=progress_wrapper, # 👈 PASAMOS EL WRAPPER
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
    )

from tqdm import tqdm

@torch.inference_mode()
def infer_batch_process_batched(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=None,
    target_rms=0.1,
    cross_fade_duration=0.1,
    nfe_step=6,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1.0,
    fix_duration=None,
    device="cuda",
):
    audio, sr = ref_audio

    # ---- mono ----
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # ---- RMS ----
    rms = torch.sqrt((audio ** 2).mean())
    if rms < target_rms:
        audio = audio * target_rms / rms

    # ---- resample ----
    if sr != 24000:
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)

    audio = audio.to(device)
    ref_audio_len = audio.shape[-1] // hop_length

    # ---- seguridad texto ----
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text += " "

    MAX_SEC = 30
    MAX_FRAMES = int(MAX_SEC * 24000 / hop_length)

    waves = []
    spectrograms = []

    total_batches = len(gen_text_batches)

    # =================================================
    # 🔥 BARRA DE PROGRESO REAL
    # =================================================
    pbar = tqdm(
        gen_text_batches,
        desc="🔊 Generando audio",
        unit="batch",
        total=total_batches,
    )

    for i, gen_text in enumerate(pbar, start=1):
        
        pbar.set_postfix({
            "batch": f"{i}/{total_batches}",
            "chars": len(gen_text)
        })

        est_duration = ref_audio_len + int(
            ref_audio_len
            / max(1, len(ref_text.encode("utf-8")))
            * len(gen_text.encode("utf-8"))
            / speed
        )

        if est_duration > MAX_FRAMES:
            MAX_FRAMES = int((MAX_SEC*2) * 24000 / hop_length)

        # 🔑 TEXTO CORRECTO
        full_text = ref_text + gen_text
        text_list = convert_char_to_pinyin([full_text])

        if fix_duration is not None:
            duration = int(fix_duration * 24000 / hop_length)
        else:
            duration = ref_audio_len + int(
                ref_audio_len
                / max(1, len(ref_text.encode("utf-8")))
                * len(gen_text.encode("utf-8"))
                / speed
            )

        duration = min(duration, MAX_FRAMES)
        MAX_FRAMES = 30

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            generated, _ = model_obj.sample(
                cond=audio,
                text=text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        # ---- quitar referencia ----
        generated = generated[:, ref_audio_len:, :].float()
        mel = generated.permute(0, 2, 1)

        # ---- vocoder ----
        if mel_spec_type == "vocos":
            wav = vocoder.decode(mel)[0]
        else:
            wav = vocoder(mel)[0]

        if rms < target_rms:
            wav *= rms / target_rms

        waves.append(wav)
        spectrograms.append(mel[0].cpu().numpy())

    # ---- crossfade ----
    fade_len = int(cross_fade_duration * 24000)
    final = waves[0]

    if fade_len > 0:
        fade_out = torch.linspace(1, 0, fade_len, device=device)
        fade_in = torch.linspace(0, 1, fade_len, device=device)

        for w in waves[1:]:
            overlap = final[-fade_len:] * fade_out + w[:fade_len] * fade_in
            final = torch.cat([final[:-fade_len], overlap, w[fade_len:]])
    else:
        for w in waves[1:]:
            final = torch.cat([final, w])

    return final.cpu().numpy(), 24000, np.concatenate(spectrograms, axis=1)



# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
