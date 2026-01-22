from lipsync import LipSync

lip = LipSync(
    model='wav2lip',
    checkpoint_path='lipsync/checkpoints/wav2lip.pth',  # <-- aquí va la ruta
    device='cuda'
)

lip.sync('video.mp4', 'audio.mp3', 'output.mp4')
