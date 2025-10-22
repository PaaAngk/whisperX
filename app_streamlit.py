import streamlit as st
import os
from pathlib import Path
from uuid import uuid4
import json
import time
from concurrent.futures import ProcessPoolExecutor
import shutil

# Worker imports are done inside worker function (to avoid loading heavy libs in main process)

# ---------- Configuration ----------
MAX_WORKERS = 5
STORAGE_ROOT = Path("storage")
TASKS_ROOT = STORAGE_ROOT / "tasks"
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
TASKS_ROOT.mkdir(parents=True, exist_ok=True)

# Ensure executor lives across reruns
if "executor" not in st.session_state:
    st.session_state.executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
if "tasks" not in st.session_state:
    st.session_state.tasks = {}  # task_id -> metadata dict

# ---------- Helpers ----------
def now_ts():
    return time.time()

def sec_to_srt_timestamp(s: float) -> str:
    # returns "HH:MM:SS,mmm"
    msec = int(round((s - int(s)) * 1000))
    s_int = int(s)
    hh = s_int // 3600
    mm = (s_int % 3600) // 60
    ss = s_int % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{msec:03d}"

def write_srt(segments, out_path: Path):
    # segments: list of dicts with 'start','end','text'
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = sec_to_srt_timestamp(seg.get("start", 0.0))
            end = sec_to_srt_timestamp(seg.get("end", seg.get("start", 0.0) + 1.0))
            text = seg.get("text", "").strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def write_txt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg.get('start', 0):.2f} - {seg.get('end', 0):.2f}] ")
            f.write(seg.get("text", "").strip() + "\n")

def save_json(result, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# ---------- Worker function (runs in separate process) ----------
def transcribe_worker(task_id: str, audio_path: str, params: dict, workdir: str):
    """
    This function is executed in a separate process.
    It performs the pipeline using whisperx as a library and writes artifacts into workdir.
    """
    import traceback
    import gc
    import whisperx
    import torch

    res = {"task_id": task_id, "status": "failed", "error": None, "artifacts": {}}
    try:
        os.makedirs(workdir, exist_ok=True)
        # 1. Parameters
        model_name = params.get("model_name", "small")
        align_model_name = params.get("align_model_name", None)
        language = params.get("language", None)
        hf_token = params.get("hf_token", None)
        compute_type = params.get("compute_type", "float16")
        device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        batch_size = params.get("batch_size", 16)
        do_diarize = params.get("diarize", True)
        do_align = params.get("align", True)

        # 2. Load model and audio
        # Write progress file
        with open(os.path.join(workdir, "status.json"), "w", encoding="utf-8") as stf:
            json.dump({"status": "loading_model"}, stf)

        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)

        audio = whisperx.load_audio(audio_path)  # returns numpy array or path-specific structure

        with open(os.path.join(workdir, "status.json"), "w", encoding="utf-8") as stf:
            json.dump({"status": "transcribing"}, stf)

        # 3. Transcribe
        result = model.transcribe(audio, batch_size=batch_size)
        # free model to save GPU
        del model
        gc.collect()
        if device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # 4. Align (optional)
        if do_align and align_model_name:
            with open(os.path.join(workdir, "status.json"), "w", encoding="utf-8") as stf:
                json.dump({"status": "loading_align_model"}, stf)
            align_model, metadata = whisperx.load_align_model(language_code=language, device=device, model_name=align_model_name)
            with open(os.path.join(workdir, "status.json"), "w", encoding="utf-8") as stf:
                json.dump({"status": "aligning"}, stf)
            result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
            del align_model
            gc.collect()
            if device.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # 5. Diarize (optional)
        if do_diarize:
            with open(os.path.join(workdir, "status.json"), "w", encoding="utf-8") as stf:
                json.dump({"status": "diarizing"}, stf)
            # whisperx provides a helper
            diarization_pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarization_pipeline(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        # 6. Save artifacts
        with open(os.path.join(workdir, "status.json"), "w", encoding="utf-8") as stf:
            json.dump({"status": "saving_artifacts"}, stf)

        # segments expected in result["segments"]
        segments = result.get("segments", [])
        # save json
        out_json = os.path.join(workdir, "transcript.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        # save srt
        out_srt = os.path.join(workdir, "transcript.srt")
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start = seg.get("start", 0.0)
                end = seg.get("end", start + 1.0)
                start_ts = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{int(start % 60):02d},{int((start - int(start)) * 1000):03d}"
                end_ts = f"{int(end // 3600):02d}:{int((end % 3600) // 60):02d}:{int(end % 60):02d},{int((end - int(end)) * 1000):03d}"
                text = seg.get("text", "").strip()
                f.write(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")
        # save txt
        out_txt = os.path.join(workdir, "transcript.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for seg in segments:
                s = seg.get("start", 0.0)
                e = seg.get("end", 0.0)
                f.write(f"[{s:.2f}-{e:.2f}] {seg.get('text','').strip()}\n")

        res["status"] = "done"
        res["artifacts"] = {"json": out_json, "srt": out_srt, "txt": out_txt, "workdir": workdir}
        # update final status file
        with open(os.path.join(workdir, "status.json"), "w", encoding="utf-8") as stf:
            json.dump({"status": "done"}, stf)

    except Exception as e:
        err = traceback.format_exc()
        res["status"] = "failed"
        res["error"] = err
        # write status
        try:
            with open(os.path.join(workdir, "status.json"), "w", encoding="utf-8") as stf:
                json.dump({"status": "failed", "error": str(e)}, stf)
        except Exception:
            pass
    finally:
        # try to free CUDA if present
        try:
            # del locals()
            gc.collect()
            if 'torch' in globals():
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

    return res

# ---------- Summarization helper ----------
def summarize_text(text: str, method: str = "auto"):
    """
    Try transformers summarizer, fallback to simple extractive (first 3 sentences).
    """
    try:
        # try transformers
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        # choose a light summarization model
        model_name = "sshleifer/distilbart-cnn-12-6"
        device = 0 if (("cuda" in str(st.session_state.get("device", ""))) and ( __import__("torch").cuda.is_available())) else -1
        summarizer = pipeline("summarization", model=model_name, device=device)
        # transformers has max token limits; handle long text by chunking
        max_chunk = 1000  # approximate
        sentences = text.split(". ")
        chunks = []
        cur = []
        cur_len = 0
        for s in sentences:
            cur.append(s)
            cur_len += len(s)
            if cur_len > max_chunk:
                chunks.append(". ".join(cur))
                cur = []
                cur_len = 0
        if cur:
            chunks.append(". ".join(cur))
        results = []
        for ch in chunks:
            out = summarizer(ch, max_length=150, min_length=40, do_sample=False)
            results.append(out[0]["summary_text"].strip())
        return "\n\n".join(results)
    except Exception as e:
        # fallback: take first 3 sentences or shorter
        sents = text.replace("\n", " ").split(". ")
        return ". ".join(sents[:3]).strip() + ("..." if len(sents) > 3 else "")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="WhisperX Meeting MVP", layout="wide")
st.title("WhisperX — Простая транскрибация совещаний (MVP)")

# Sidebar
st.sidebar.header("Настройки (для MVP)")
hf_token = st.sidebar.text_input("HuggingFace Token (необязательно, нужен для diarize gated models)", type="password")
device_choice = st.sidebar.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
st.session_state["device"] = device_choice if device_choice != "auto" else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
st.sidebar.markdown(f"Используемое устройство: **{st.session_state['device']}**")
st.sidebar.info("Приложение использует whisperx как библиотеку. При первом запуске модели будут скачаны — это может занять время.")

# Upload area
st.subheader("1) Загрузите аудиофайл")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Аудиофайл (wav, mp3, m4a, flac)", type=["wav", "mp3", "m4a", "flac"])
    meeting_title = st.text_input("Название встречи", value="Совещание")
with col2:
    st.write("Параметры (просто для инфо):")
    model_name = st.selectbox("Модель (MVP)", ["small", "medium", "large-v3"], index=0)
    compute_type = st.selectbox("compute_type", ["float16", "float32", "int8"], index=0)
    do_align = st.checkbox("Word-level alignment (better timestamps)", value=True)
    do_diarize = st.checkbox("Diarization (speaker labels)", value=True)
    batch_size = st.slider("batch size (GPU memory)", 1, 32, 16)

# Action: Start transcription
if st.button("Транскрибировать"):
    if not uploaded_file:
        st.warning("Пожалуйста, сначала загрузите файл.")
    else:
        task_id = str(uuid4())
        workdir = TASKS_ROOT / task_id
        workdir.mkdir(parents=True, exist_ok=True)
        # save uploaded file
        ext = Path(uploaded_file.name).suffix or ".wav"
        audio_path = workdir / f"original{ext}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # prepare params
        params = {
            "model_name": model_name,
            "align_model_name": "jonatasgrosman/wav2vec2-large-xlsr-53-russian" if do_align else None,
            "language": "ru",
            "hf_token": hf_token,
            "compute_type": compute_type,
            "device": st.session_state["device"],
            "batch_size": batch_size,
            "diarize": do_diarize,
            "align": do_align,
        }
        # register task
        st.session_state.tasks[task_id] = {
            "id": task_id,
            "title": meeting_title,
            "status": "queued",
            "created_at": now_ts(),
            "workdir": str(workdir),
            "audio_path": str(audio_path),
            "params": params,
            "result": None,
            "future": None,
        }
        # submit to executor
        future = st.session_state.executor.submit(transcribe_worker, task_id, str(audio_path), params, str(workdir))
        st.session_state.tasks[task_id]["future"] = future
        st.session_state.tasks[task_id]["status"] = "running"
        st.success(f"Задача {task_id} поставлена в очередь и запущена.")

st.markdown("---")
st.subheader("2) Текущие задачи и история")

# Display tasks list
to_remove = []
for tid, meta in list(st.session_state.tasks.items()):
    cols = st.columns([1, 2, 1, 1])
    with cols[0]:
        st.write(f"**{meta['title']}**")
        st.write(f"`{tid[:8]}`")
    with cols[1]:
        status = meta.get("status", "queued")
        future = meta.get("future", None)
        # update status from future
        if future is not None:
            if future.running():
                status = "running"
            if future.done():
                try:
                    res = future.result(timeout=0)
                    meta["result"] = res
                    status = res.get("status", "done")
                except Exception as e:
                    status = "failed"
                    meta["result"] = {"status": "failed", "error": str(e)}
            meta["status"] = status
        st.write(f"Status: **{status}**")
        # show a small log/status file if exists
        status_file = Path(meta["workdir"]) / "status.json"
        if status_file.exists():
            try:
                sj = json.loads(status_file.read_text(encoding="utf-8"))
                st.write(sj)
            except Exception:
                pass
    with cols[2]:
        # open button
        if st.button("Открыть", key=f"open_{tid}"):
            st.session_state["_open_task"] = tid
            st.experimental_rerun()
    with cols[3]:
        if st.button("Удалить", key=f"del_{tid}"):
            # attempt to cancel if running
            fut = meta.get("future")
            if fut and not fut.done():
                fut.cancel()
            # remove files
            try:
                shutil.rmtree(meta["workdir"])
            except Exception:
                pass
            to_remove.append(tid)
            st.experimental_rerun()

for tid in to_remove:
    st.session_state.tasks.pop(tid, None)

# If user opened a task, show transcript editor
open_task_id = st.session_state.get("_open_task", None)
if open_task_id:
    meta = st.session_state.tasks.get(open_task_id)
    if not meta:
        st.error("Task not found")
    else:
        st.markdown("---")
        st.subheader(f"Транскрипт: {meta['title']}  (`{open_task_id[:8]}`)")
        workdir = Path(meta["workdir"])
        # display artifacts if ready
        transcript_json = workdir / "transcript.json"
        transcript_txt = workdir / "transcript.txt"
        transcript_srt = workdir / "transcript.srt"

        # Refresh button
        if st.button("Обновить"):
            st.experimental_rerun()

        if transcript_json.exists():
            st.success("Результаты готовы.")
            with open(transcript_json, "r", encoding="utf-8") as f:
                result = json.load(f)
            segments = result.get("segments", [])
            # Audio player
            st.audio(meta["audio_path"])

            # Show transcript segments and allow inline editing
            st.info("Вы можете править текст сегментов. Нажмите 'Сохранить правки' после редактирования.")
            edited_segments = []
            for i, seg in enumerate(segments):
                st.markdown(f"**Segment {i+1}** — {seg.get('start',0):.2f}s - {seg.get('end',0):.2f}s — speaker: {seg.get('speaker','-')}")
                new_text = st.text_area(f"text_{i}", value=seg.get("text",""), key=f"textarea_{open_task_id}_{i}", height=80)
                edited_segments.append({"start": seg.get("start", 0.0), "end": seg.get("end", 0.0), "text": new_text, "speaker": seg.get("speaker", None)})

            if st.button("Сохранить правки"):
                # overwrite transcript json and txt/srt
                new_result = result.copy()
                new_result["segments"] = edited_segments
                save_json(new_result, workdir / "transcript.json")
                write_txt(edited_segments, workdir / "transcript.txt")
                write_srt(edited_segments, workdir / "transcript.srt")
                st.success("Правки сохранены.")

            # Summarization
            st.markdown("---")
            st.subheader("Суммаризация")
            sum_mode = st.selectbox("Режим суммаризации", ["Краткая (TL;DR)", "Подробно (minutes)", "Action items"], index=0)
            if st.button("Сгенерировать суммаризацию"):
                # gather full text
                full_text = "\n".join([s["text"] for s in edited_segments])
                with st.spinner("Генерирую суммаризацию..."):
                    summary_text = summarize_text(full_text, method="auto")
                    (workdir / "summary.txt").write_text(summary_text, encoding="utf-8")
                st.success("Суммаризация готова.")
                st.text_area("Summary", value=summary_text, height=200)

            # Export buttons
            st.markdown("---")
            st.subheader("Выгрузка")
            if transcript_json.exists():
                with open(transcript_json, "rb") as f:
                    st.download_button("Скачать JSON", f.read(), file_name=f"{open_task_id}_transcript.json")
            if transcript_srt.exists():
                with open(transcript_srt, "rb") as f:
                    st.download_button("Скачать SRT", f.read(), file_name=f"{open_task_id}.srt")
            if transcript_txt.exists():
                with open(transcript_txt, "rb") as f:
                    st.download_button("Скачать TXT", f.read(), file_name=f"{open_task_id}.txt")
            summary_file = workdir / "summary.txt"
            if summary_file.exists():
                with open(summary_file, "rb") as f:
                    st.download_button("Скачать Summary", f.read(), file_name=f"{open_task_id}_summary.txt")

        else:
            st.info("Результаты пока не готовы. Подождите или нажмите 'Обновить' спустя некоторое время.")
            # show status file
            status_file = workdir / "status.json"
            if status_file.exists():
                try:
                    st.write(json.loads(status_file.read_text(encoding="utf-8")))
                except Exception:
                    st.write("Невозможно прочитать status.json")

# Footer notes
st.markdown("---")
st.caption("MVP: Backend отсутствует, всё выполняется локально (whisperx как библиотека). "
           "Первый запуск скачивает модели — это может занять некоторое время. "
           "Если используется GPU — убедитесь, что venv имеет корректный PyTorch с CUDA.")
