import whisperx
import torch
import gc

# 1. Parameters
audio_file = "../data/cut10m.wav"
model_name = "turbo"
align_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
language = "ru"
hf_token = ""
compute_type = "float16"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 # or 8, 4, etc. depending on your GPU memory

print(f"Using device: {device}")

# 2. Transcribe
print("Loading ASR model...")
model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)

print("Loading audio...")
audio = whisperx.load_audio(audio_file)

print("Transcribing...")
result = model.transcribe(audio, batch_size=batch_size)

print("Transcription complete.")
# Clean up ASR model
del model
gc.collect()
torch.cuda.empty_cache()

# 3. Align
print("Loading alignment model...")
align_model, metadata = whisperx.load_align_model(language_code=language, device=device, model_name=align_model_name)

print("Aligning transcription...")
result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

print("Alignment complete.")
# Clean up alignment model
del align_model
gc.collect()
torch.cuda.empty_cache()

# 4. Diarize
print("Loading diarization pipeline...")
diarization_pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)

print("Performing diarization...")
diarize_segments = diarization_pipeline(audio)

print("Assigning speakers...")
result = whisperx.assign_word_speakers(diarize_segments, result)

print("Diarization complete.")

# 5. Print result
import json
print(json.dumps(result["segments"], indent=2, ensure_ascii=False))
