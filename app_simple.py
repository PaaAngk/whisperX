import streamlit as st
from pathlib import Path
from uuid import uuid4
import json
import time
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
from docx import Document
from io import BytesIO

# ---------- Config (minimal) ----------
STORAGE = Path("storage")
TASKS = STORAGE / "tasks"
STORAGE.mkdir(parents=True, exist_ok=True)
TASKS.mkdir(parents=True, exist_ok=True)
hf_token = ""

# How many background worker processes allowed.
# For single-GPU machines it's safer to set 1. You can override via env var MAX_WORKERS.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1"))

# Keep one persistent executor across reruns
if "executor" not in st.session_state:
    st.session_state.executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
if "current_task" not in st.session_state:
    st.session_state.current_task = None

# ---------- Utility helpers ----------
def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def save_bytes(path: Path, b: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)

def read_status(workdir: Path):
    sf = workdir / "status.json"
    if sf.exists():
        try:
            return json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "unknown (bad status.json)"}
    return None

def write_status(workdir: Path, data):
    (workdir / "status.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def create_docx(text: str) -> BytesIO:
    doc = Document()
    doc.add_paragraph(text)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def format_by_speaker(segments):
    lines = []
    current_speaker = None
    speaker_text = ""

    for segment in segments:
        if "speaker" in segment:
            speaker = segment["speaker"]
            if current_speaker != speaker:
                if current_speaker is not None:
                    lines.append(f"**{current_speaker}:** {speaker_text.strip()}")
                current_speaker = speaker
                speaker_text = ""
            speaker_text += segment["text"].strip() + " "
        else:
            # If there's pending speaker text, write it out
            if current_speaker is not None:
                 lines.append(f"**{current_speaker}:** {speaker_text.strip()}")
                 current_speaker = None
                 speaker_text = ""
            lines.append(segment["text"].strip())


    if current_speaker is not None and speaker_text.strip():
        lines.append(f"**{current_speaker}:** {speaker_text.strip()}")

    return "\n\n".join(lines)


# ---------- Worker that runs in separate process ----------
def worker_transcribe(task_id: str, audio_path: str, workdir: str):
    """
    Runs in a separate process. Uses whisperx as library.
    Writes artifacts into workdir. Returns summary dict.
    """
    import json, os, traceback, gc
    import whisperx
    import torch

    # Set cache directory for all models, so they can be bundled into a container image
    cache_dir = "./cache"
    os.environ['XDG_CACHE_HOME'] = cache_dir
    os.environ['TORCH_HOME'] = cache_dir

    workdir_p = Path(workdir)
    workdir_p.mkdir(parents=True, exist_ok=True)
    status_file = workdir_p / "status.json"

    result_summary = {"task_id": task_id, "status": "failed", "error": None, "artifacts": {}}
    try:
        # 1. Start
        status_file.write_text(json.dumps({"status": "loading_model", "ts": now_iso()}), encoding="utf-8")

        # simple defaults (no UI options)
        model_name = "turbo"
        align_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
        language = "ru"
        compute_type = "float16"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 16

        # 2. Load asr model
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language, download_root=cache_dir)

        status_file.write_text(json.dumps({"status": "loading_audio", "ts": now_iso()}), encoding="utf-8")
        audio = whisperx.load_audio(audio_path)

        status_file.write_text(json.dumps({"status": "transcribing", "ts": now_iso()}), encoding="utf-8")
        asr_result = model.transcribe(audio, batch_size=batch_size)

        # free ASR model
        del model
        gc.collect()
        if device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Align
        status_file.write_text(json.dumps({"status": "loading_align_model", "ts": now_iso()}), encoding="utf-8")
        align_model, align_meta = whisperx.load_align_model(language_code=language, device=device, model_name=align_model_name, model_dir=cache_dir)

        status_file.write_text(json.dumps({"status": "aligning", "ts": now_iso()}), encoding="utf-8")
        aligned = whisperx.align(asr_result["segments"], align_model, align_meta, audio, device, return_char_alignments=False)

        # free align model
        del align_model
        gc.collect()
        if device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Diarize (try best-effort, no hf token required for public models)
        status_file.write_text(json.dumps({"status": "diarizing", "ts": now_iso()}), encoding="utf-8")
        try:
            from whisperx.diarize import DiarizationPipeline
            diarizer = DiarizationPipeline(use_auth_token=hf_token, device=device, cache_dir=cache_dir)
            diar_segments = diarizer(audio)
            final = whisperx.assign_word_speakers(diar_segments, aligned)
        except Exception as e:
            print(e)
            # if diarization fails, continue without speakers
            final = aligned

        # Save artifacts
        status_file.write_text(json.dumps({"status": "saving", "ts": now_iso()}), encoding="utf-8")
        out_json = workdir_p / "transcript.json"
        out_txt = workdir_p / "transcript.txt"

        # write json
        out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

        # build segs for txt
        segs = final.get("segments", [])
        
        # txt
        with out_txt.open("w", encoding="utf-8") as f:
            for s in segs:
                f.write(f"{s.get('text','').strip()}\n")

        # done
        status_file.write_text(json.dumps({"status": "done", "ts": now_iso(), "segments": len(segs)}), encoding="utf-8")

        result_summary["status"] = "done"
        result_summary["artifacts"] = {
            "json": str(out_json),
            "txt": str(out_txt),
            "workdir": str(workdir_p),
        }
    except Exception as e:
        tb = traceback.format_exc()
        status_file.write_text(json.dumps({"status": "failed", "error": str(e), "trace": tb}), encoding="utf-8")
        result_summary["status"] = "failed"
        result_summary["error"] = tb
    finally:
        # best-effort free
        try:
            gc.collect()
            if "torch" in globals():
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

    return result_summary

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Транскрибация аудио", layout="centered")
st.title("Простой транскрибатор речи")

st.write("Загрузите аудиофайл и нажмите **Транскрибировать**.")
uploaded = st.file_uploader("Аудио (wav, mp3, m4a, flac)", type=["wav","mp3","m4a","flac"])

if st.button("Транскрибировать"):
    if not uploaded:
        st.warning("Пожалуйста, сначала загрузите аудиофайл.")
    else:
        # create task
        task_id = str(uuid4())
        workdir = TASKS / task_id
        workdir.mkdir(parents=True, exist_ok=True)
        audio_name = uploaded.name or f"{task_id}.wav"
        audio_path = workdir / audio_name
        save_bytes(audio_path, uploaded.getvalue())
        # write initial status
        write_status(workdir, {"status": "queued", "ts": now_iso()})
        # schedule
        future = st.session_state.executor.submit(worker_transcribe, task_id, str(audio_path), str(workdir))
        st.session_state.current_task = {"id": task_id, "future": future, "workdir": str(workdir)}
        st.session_state.task_running = True
        st.session_state.result = None


if st.session_state.get("task_running"):
    ct = st.session_state.current_task
    workdir = Path(ct["workdir"])
    future = ct["future"]

    progress_bar = st.progress(0)
    status_text = st.empty()

    STATUS_MAP = {
        "queued": ("В очереди...", 5),
        "loading_model": ("Загрузка модели...", 15),
        "loading_audio": ("Обработка аудио...", 30),
        "transcribing": ("Транскрибация...", 50),
        "loading_align_model": ("Загрузка модели выравнивания...", 70),
        "aligning": ("Выравнивание текста...", 80),
        "diarizing": ("Разделение по спикерам...", 90),
        "saving": ("Сохранение результатов...", 95),
        "done": ("Готово!", 100),
        "failed": ("Ошибка!", 100),
    }

    while not future.done():
        status_data = read_status(workdir)
        status_key = status_data.get("status", "queued") if status_data else "queued"
        
        message, percent = STATUS_MAP.get(status_key, ("Неизвестный статус...", 0))
        
        status_text.text(message)
        progress_bar.progress(percent)
        time.sleep(1)

    # Task finished
    try:
        res = future.result(timeout=1)
        st.session_state.result = res
    except Exception as e:
        st.session_state.result = {"status": "failed", "error": str(e)}
    
    st.session_state.task_running = False
    # Force a rerun to display results
    # st.experimental_rerun()


if st.session_state.get("result"):
    res = st.session_state.result
    if res.get("status") == "done":
        st.success("Транскрибация завершена.")
        workdir = Path(res["artifacts"]["workdir"])
        out_json_path = workdir / "transcript.json"
        
        if out_json_path.exists():
            transcript_data = json.loads(out_json_path.read_text(encoding="utf-8"))
            segments = transcript_data.get("segments", [])
            
            full_text = "\n".join([s.get("text", "").strip() for s in segments])
            speaker_text = format_by_speaker(segments)

            st.subheader("Результат транскрибации")
            with st.expander("Показать полный текст"):
                st.text_area("Полный текст", value=full_text, height=300, key="exp_full_text")
            
            with st.expander("Показать текст по спикерам"):
                st.markdown(speaker_text)

            st.subheader("Скачать результаты")
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="Скачать .txt (полный текст)",
                    data=full_text.encode("utf-8"),
                    file_name=f"{st.session_state.current_task['id']}_full.txt",
                    mime="text/plain",
                )
                st.download_button(
                    label="Скачать .docx (полный текст)",
                    data=create_docx(full_text),
                    file_name=f"{st.session_state.current_task['id']}_full.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            
            with col2:
                st.download_button(
                    label="Скачать .txt (по спикерам)",
                    data=speaker_text.encode("utf-8"),
                    file_name=f"{st.session_state.current_task['id']}_speakers.txt",
                    mime="text/plain",
                )
                st.download_button(
                    label="Скачать .docx (по спикерам)",
                    data=create_docx(speaker_text),
                    file_name=f"{st.session_state.current_task['id']}_speakers.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

    else:
        st.error("Задача не удалась. Подробности ниже.")
        st.text(res.get("error") or "Неизвестная ошибка.")

# footer
st.markdown("---")
st.caption("Простое приложение для транскрибации аудио.")
