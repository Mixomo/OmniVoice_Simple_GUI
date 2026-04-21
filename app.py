import os
import sys
import json
import glob
import subprocess
import threading
import shutil
import random
import datetime
from pathlib import Path
from typing import Optional
import winsound

import gradio as gr
import numpy as np
import torch
import soundfile as sf
import yaml
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

# OmniVoice imports
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

project_root = Path(__file__).parent.absolute()

# --- Global State ---
current_model: Optional[OmniVoice] = None
current_checkpoint: Optional[str] = None
asr_model: Optional[WhisperModel] = None
training_process: Optional[subprocess.Popen] = None
training_log = ""
extract_process: Optional[subprocess.Popen] = None


_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

OMNIVOICE_MODELS = {
    "OmniVoice (Official)": "k2-fsa/OmniVoice",
    "OmniVoice-bf16": "drbaph/OmniVoice-bf16",
    "OmniVoice-Singing": "ModelsLab/omnivoice-singing"
}

def unload_asr():
    global asr_model
    if asr_model is not None:
        print("Unloading ASR model...", file=sys.stderr)
        del asr_model
        asr_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def unload_omnivoice():
    global current_model, current_checkpoint
    if current_model is not None:
        print("Unloading OmniVoice model...", file=sys.stderr)
        del current_model
        current_model = None
        current_checkpoint = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def play_done_chime():
    chime_path = project_root / "assets" / "inference_training_done.wav"
    if chime_path.exists():
        try:
            winsound.PlaySound(str(chime_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except: pass

def load_omnivoice(checkpoint):
    global current_model, current_checkpoint
    
    # If the requested checkpoint is already loaded, do nothing
    if current_model is not None and current_checkpoint == checkpoint:
        return "Model Already Loaded"
        
    # If a different model is loaded, unload it first
    if current_model is not None:
        unload_omnivoice()
        
    # Ensure ASR is unloaded before loading OmniVoice
    unload_asr()
    
    try:
        # Check if the checkpoint matches one of our known HuggingFace repo IDs
        repo_ids = list(OMNIVOICE_MODELS.values())
        if checkpoint in repo_ids:
            folder_name = checkpoint.replace("/", "--")
            dest_path = project_root / "models" / folder_name
            if not dest_path.exists():
                print(f"Downloading {checkpoint} model to {dest_path}...", file=sys.stderr)
                snapshot_download(repo_id=checkpoint, local_dir=str(dest_path), local_dir_use_symlinks=False)
            checkpoint = str(dest_path)
            
        device = get_best_device()
        m = OmniVoice.from_pretrained(checkpoint, device_map=device, dtype=torch.float16, load_asr=False, attn_implementation="sdpa")
            
        current_model = m
        current_checkpoint = checkpoint
        return "Model Loaded"
    except Exception as e:
        return f"Error loading model: {e}"

def get_best_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

WHISPER_MODELS = {
    "tiny (~1 GB VRAM)": "Systran/faster-whisper-tiny",
    "base (~1 GB VRAM)": "Systran/faster-whisper-base",
    "small (~2 GB VRAM)": "Systran/faster-whisper-small",
    "medium (~5 GB VRAM)": "Systran/faster-whisper-medium",
    "large-v2 (~10 GB VRAM)": "Systran/faster-whisper-large-v2",
    "large-v3 (~10 GB VRAM)": "Systran/faster-whisper-large-v3",
    "distil-large-v3 (~5 GB VRAM)": "Systran/faster-distil-whisper-large-v3"
}

current_asr_name = None

def get_or_load_asr_model(requested_model="large-v3 (~10 GB VRAM)"):
    global asr_model, current_asr_name
    
    repo_id = WHISPER_MODELS.get(requested_model, requested_model)
    
    if asr_model is not None and current_asr_name == repo_id:
        return asr_model
    
    unload_asr()
    unload_omnivoice()
    
    print(f"Loading ASR model ({repo_id})...", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    dest_path = project_root / "models" / repo_id.replace("/", "--")
    if not dest_path.exists():
        snapshot_download(repo_id=repo_id, local_dir=str(dest_path), local_dir_use_symlinks=False)
        
    asr_model = WhisperModel(str(dest_path), device=device, compute_type=compute_type)
    current_asr_name = repo_id
    return asr_model

def get_sample_choices():
    samples_dir = project_root / "samples"
    if not samples_dir.exists():
        samples_dir.mkdir(parents=True, exist_ok=True)
        return []
    choices = [os.path.splitext(f)[0] for f in os.listdir(samples_dir) if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))]
    return ["None"] + sorted(list(set(choices)))

def load_sample(sample_name):
    if not sample_name or sample_name == "None": return None, ""
    samples_dir = project_root / "samples"
    audio_path = None
    for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
        p = samples_dir / f"{sample_name}{ext}"
        if p.exists():
            audio_path = p
            break
    if not audio_path: return None, ""
    
    text = ""
    txt_path = samples_dir / f"{sample_name}.txt"
    json_path = samples_dir / f"{sample_name}.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                text = d.get("Text", d.get("text", ""))
        except: pass
    if not text and txt_path.exists():
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except: pass
    return str(audio_path.absolute()), text

def save_prep_sample(audio_path, sample_name, transcription):
    if not audio_path or not sample_name: return "Error: Missing data."
    samples_dir = project_root / "samples"
    os.makedirs(samples_dir, exist_ok=True)
    ext = os.path.splitext(audio_path)[1] or ".wav"
    dest_audio = samples_dir / f"{sample_name}{ext}"
    dest_txt = samples_dir / f"{sample_name}.txt"
    dest_json = samples_dir / f"{sample_name}.json"
    
    shutil.copy(audio_path, dest_audio)
    with open(dest_txt, "w", encoding="utf-8") as f: f.write(transcription)
    with open(dest_json, "w", encoding="utf-8") as f: json.dump({"text": transcription}, f, ensure_ascii=False)
    
    return f"Sample '{sample_name}' saved successfully (Audio + JSON + TXT)!"

def recognize_audio(audio_path, whisper_model="large-v3 (~10 GB VRAM)"):
    if not audio_path: return ""
    try:
        model = get_or_load_asr_model(whisper_model)
        segments, _ = model.transcribe(audio_path, beam_size=5)
        return "".join([s.text for s in segments]).strip()
    except Exception as e:
        print(f"ASR Error: {e}", file=sys.stderr)
        return ""

def scan_datasets():
    datasets_root = project_root / "data"
    if not datasets_root.exists():
        os.makedirs(datasets_root, exist_ok=True)
        return []
    manifests = []
    for root, dirs, files in os.walk(datasets_root):
        if "data.lst" in files:
            p = os.path.join(root, "data.lst")
            manifests.append(os.path.relpath(p, project_root).replace("\\", "/"))
    return sorted(manifests)


def get_existing_lora_projects():
    out_dir = project_root / "exp"
    if not out_dir.exists():
        os.makedirs(out_dir, exist_ok=True)
        return []
    projects = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    return sorted(projects)

def scan_lora_checkpoints(with_info=False):
    checkpoints = []
    out_dir = project_root / "exp"
    if not out_dir.exists():
        os.makedirs(out_dir, exist_ok=True)
        return []

    # Real scan: only add directories that contain actual weight files
    for root, dirs, files in os.walk(out_dir):
        # We look for safetensors or bin files which indicate a loadable model/adapter
        weight_files = ["model.safetensors", "pytorch_model.bin", "adapter_model.bin", "adapter_model.safetensors"]
        if any(f in files for f in weight_files):
            rel_path = os.path.relpath(root, project_root).replace("\\", "/")
            if with_info:
                # Try to label it based on if it's an adapter or full model
                label = "Trained LoRA" if "adapter_model.bin" in files or "adapter_model.safetensors" in files else ""
                checkpoints.append((rel_path, label))
            else:
                checkpoints.append(rel_path)
    
    return sorted(checkpoints, key=lambda x: x[0] if isinstance(x, tuple) else x)
    
def calculate_dataset_stats(manifest_path):
    if not manifest_path: return 0, 0, 0, 0
    
    manifest_file = project_root / manifest_path
    dataset_dir = manifest_file.parent.parent # data/name
    
    total_count = 0
    total_duration = 0.0
    total_tokens = 0
    
    # Iterate over both train and val extracted shards for accurate stats
    for split in ["train", "val"]:
        txt_dir = dataset_dir / split / "txts"
        if txt_dir.exists():
            for shard_file in txt_dir.glob("*.jsonl"):
                try:
                    with open(shard_file, "r", encoding="utf-8") as f:
                        for line in f:
                            d = json.loads(line)
                            total_count += 1
                            total_duration += float(d.get("audio_duration", 0.0))
                            total_tokens += int(d.get("num_tokens", 0))
                except: pass

    avg_dur = total_duration / total_count if total_count > 0 else 0
    
    # Fallback to raw count if no shards found yet
    if total_count == 0:
        raw_jsonl = dataset_dir / "train_raw.jsonl"
        if raw_jsonl.exists():
            try:
                with open(raw_jsonl, "r", encoding="utf-8") as f:
                    for _ in f: total_count += 1
            except: pass
            
    return total_count, total_duration, avg_dur, total_tokens


def refresh_loras():
    checkpoints_with_info = scan_lora_checkpoints(with_info=True)
    choices = ["None"]
    for ckpt in checkpoints_with_info:
        if isinstance(ckpt, tuple):
            choices.append(ckpt[0])
        else:
            choices.append(ckpt)
    return gr.update(choices=choices, value="None")

def prepare_voxcpm_dataset(source_folder, dataset_name, val_split, batch_size, lang_code="en", whisper_model="large-v3 (~10 GB VRAM)", progress=gr.Progress()):
    global extract_process
    
    if not source_folder or not os.path.exists(source_folder):
        print("Error: Please provide a valid source folder path.", file=sys.stderr)
        return "Error: Please provide a valid source folder path."
    if not dataset_name or dataset_name.strip() == "":
        print("Error: Please provide a target dataset name.", file=sys.stderr)
        return "Error: Please provide a target dataset name."
        
    out_dir = project_root / "data" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "data.jsonl"
    
    print(f"Scanning {source_folder} for audios...", file=sys.stderr)
    valid_exts = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    all_files = os.listdir(source_folder)
    audios = [os.path.join(source_folder, f) for f in all_files if f.lower().endswith(valid_exts)]
    audios = sorted(list(set(audios))) # Extra safety against duplicates
    
    if not audios:
        print("Error: No audio files found in the source folder.", file=sys.stderr)
        return f"Error: No audio files found in the source folder."
    
    print(f"Found {len(audios)} audio files. Auto-transcribing missing scripts via Whisper...", file=sys.stderr)
    
    lines = []
    asr = get_or_load_asr_model()
    
    for i, ap in enumerate(sorted(audios)):
        base = os.path.splitext(os.path.basename(ap))[0]
        progress((i, len(audios)), desc=f"Procesando {base}...")

        text_file = os.path.join(source_folder, f"{base}.txt")
        text = ""
        if os.path.exists(text_file):
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
        else:
            try:
                asr = get_or_load_asr_model(whisper_model)
                segments, _ = asr.transcribe(ap, beam_size=5, language=lang_code)
                text = "".join([s.text for s in segments]).strip()
            except Exception as e:
                print(f"Failed to transcribe {ap}: {e}", file=sys.stderr)
        
        # Audio paths deben ser locales o file:/// si fallan. Guardemos relativos o absolutos usando slash
        audio_path_clean = os.path.abspath(ap).replace("\\", "/")
        lines.append({
            "id": base,
            "audio_path": audio_path_clean,
            "text": text,
            "language_id": lang_code
        })
        
    # Shuffling & Splitting
    random.seed(42)
    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - float(val_split)))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    print(f"Dataset split: {len(train_lines)} train samples, {len(val_lines)} validation samples.", file=sys.stderr)
    
    # Save JSONLs
    train_jsonl = out_dir / "train_raw.jsonl"
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for l in train_lines: f.write(json.dumps(l, ensure_ascii=False) + "\n")
        
    val_jsonl = None
    if val_lines:
        val_jsonl = out_dir / "val_raw.jsonl"
        with open(val_jsonl, "w", encoding="utf-8") as f:
            for l in val_lines: f.write(json.dumps(l, ensure_ascii=False) + "\n")
            
    # Save statistics
    stats = {
        "count": len(train_lines),
        "total_duration": "Scan required or estimated", # Real duration check would require librosa/ffprobe
        "total_files": len(lines)
    }
    with open(out_dir / "stats.json", "w") as f: json.dump(stats, f)
            
    # Sequential extraction tasks
    def run_sequenced_extraction():
        global extract_process
        
        # 1. Train Extraction
        progress(0.7, desc="Extracting Train shards...")
        train_tar = f"data/{dataset_name}/train/audios/shard-%06d.tar"
        train_txt = f"data/{dataset_name}/train/txts/shard-%06d.jsonl"
        os.makedirs(os.path.join(out_dir, "train", "audios"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "train", "txts"), exist_ok=True)
        
        cmd_t = [
            sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
            "--input_jsonl", str(train_jsonl),
            "--tar_output_pattern", train_tar,
            "--jsonl_output_pattern", train_txt,
            "--tokenizer_path", "eustlb/higgs-audio-v2-tokenizer",
            "--shuffle", "True",
            "--loader_workers", "1",
            "--nj_per_gpu", "1"
        ]
        extract_process = subprocess.Popen(cmd_t)
        extract_process.wait()
        
        # 2. Val Extraction (if needed)
        if val_lines and extract_process.returncode == 0:
            progress(0.9, desc="Extracting Val shards...")
            val_tar = f"data/{dataset_name}/val/audios/shard-%06d.tar"
            val_txt = f"data/{dataset_name}/val/txts/shard-%06d.jsonl"
            os.makedirs(os.path.join(out_dir, "val", "audios"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "val", "txts"), exist_ok=True)
            
            cmd_v = [
                sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
                "--input_jsonl", str(val_jsonl),
                "--tar_output_pattern", val_tar,
                "--jsonl_output_pattern", val_txt,
                "--tokenizer_path", "eustlb/higgs-audio-v2-tokenizer",
                "--shuffle", "True"
            ]
            extract_process = subprocess.Popen(cmd_v)
            extract_process.wait()
            
        play_done_chime()
        print(f"\nAll extraction tasks finished.", file=sys.stderr)
        
    threading.Thread(target=run_sequenced_extraction, daemon=True).start()
    return f"Dataset preparation for '{dataset_name}' (Split: {val_split}) started in background."


def run_inference(text, ref_audio, ref_text, model_selection, cfg_scale, steps, seed, control, duration=None, t_shift=0.1, pos_temp=5.0, class_temp=0.0, layer_penalty=5.0, chunk_dur=15.0, chunk_thr=30.0, use_ref_text=True, tags_list=None, pp=True, po=True):
    global current_model
    
    # Unified model selection logic
    if model_selection in OMNIVOICE_MODELS:
        target_ckpt = OMNIVOICE_MODELS[model_selection]
    else:
        # Local LoRA path
        target_ckpt = os.path.abspath(model_selection)
        
    res = load_omnivoice(target_ckpt)
    if "Error" in res:
        return None, res
    
    if not text or not text.strip(): return None, "Please enter text."
    
    # Workaround for Spanish punctuation drop (Issue #116)
    import re
    text = re.sub(r'([,\.\!\?\;:])', r' \1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Prepend target text with chosen tags (e.g., [singing] [laughter])
    if tags_list:
        text = " ".join(tags_list) + " " + text
    
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(steps), 
        guidance_scale=float(cfg_scale), 
        denoise=True, 
        t_shift=float(t_shift),
        position_temperature=float(pos_temp),
        class_temperature=float(class_temp),
        layer_penalty_factor=float(layer_penalty),
        audio_chunk_duration=float(chunk_dur),
        audio_chunk_threshold=float(chunk_thr),
        preprocess_prompt=bool(pp), 
        postprocess_output=bool(po)
    )
    
    kw = {
        "text": text.strip(), 
        "language": None, 
        "generation_config": gen_config,
        "duration": float(duration) if duration and float(duration) > 0 else None
    }
    
    if ref_audio:
        # If use_ref_text is False, we pass None to ignore the transcript
        final_ref_text = (ref_text or None) if use_ref_text else None
        kw["voice_clone_prompt"] = current_model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=final_ref_text)
    if control:
        kw["instruct"] = control.strip()
    
    try:
        audio_out = current_model.generate(**kw)
        play_done_chime()
        waveform = (audio_out[0] * 32767).astype(np.int16)
        return (current_model.sampling_rate, waveform), "Generation Success"
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_msg = str(e)
        if "BackendCompilerFailed" in err_msg or "triton" in err_msg.lower():
            return None, f"Triton/Compiler Error: {err_msg}. Please try switching Attention Implementation to 'sdpa' in Optimization settings."
        return None, f"Error: {err_msg}"

def start_training(model_choice, train_manifest, val_manifest, output_name, lr, steps, batch_tokens, llm_name, resume_checkpoint, grad_accum, save_steps, eval_text, eval_audio, enable_eval, lang_code, warmup_ratio=0.01, repeat_factor=1):
    global training_process
    if training_process and training_process.poll() is None:
        print("Training is already running.", file=sys.stderr)
        return "Training is already running."
    
    # Unload everything to give background process maximum VRAM
    unload_asr()
    unload_omnivoice()
    
    if not train_manifest:
        print("Train Manifest is required.", file=sys.stderr)
        return "Train Manifest is required."
    
    out_exp = project_root / "exp" / output_name if output_name else project_root / "exp" / f"omnivoice_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_exp.mkdir(parents=True, exist_ok=True)
    
    # Store custom eval audio if provided
    final_eval_audio = None
    if eval_audio and isinstance(eval_audio, str):
        eval_ext = os.path.splitext(eval_audio)[1]
        final_eval_audio = str(out_exp / f"eval_reference{eval_ext}")
        shutil.copy(eval_audio, final_eval_audio)

    train_config = {
        "llm_name_or_path": llm_name.strip(),
        "steps": int(steps),
        "learning_rate": float(lr),
        "warmup_ratio": float(warmup_ratio),
        "batch_tokens": int(batch_tokens),
        "gradient_accumulation_steps": int(grad_accum),
        "save_steps": int(save_steps),
        "init_from_checkpoint": OMNIVOICE_MODELS.get(model_choice, "k2-fsa/OmniVoice"),
        "eval_text": eval_text.strip() if eval_text else "I am training and getting better every day.",
        "eval_ref_audio": final_eval_audio,
        "enable_eval": bool(enable_eval)
    }

    # Auto-adjust num_workers based on shard count (Recipe: num_workers <= shard count)
    try:
        manifest_full = os.path.join(project_root, rel_train.replace("/", os.sep))
        shard_dir = os.path.join(os.path.dirname(manifest_full), "audios")
        if os.path.exists(shard_dir):
            shards = [f for f in os.listdir(shard_dir) if f.endswith('.tar')]
            shard_count = len(shards)
            # Clip workers at 4, but never exceed shard count
            train_config["num_workers"] = min(shard_count, 4)
        else:
            train_config["num_workers"] = 0
    except:
        train_config["num_workers"] = 0

    if os.name == 'nt' and train_config["num_workers"] > 1:
        # On Windows, more than 1 worker can sometimes be unstable with WebDataset/PyTorch
        train_config["num_workers"] = 1

    
    if resume_checkpoint and resume_checkpoint.strip():
        # Ensure resume path is relative or at least clean of drive letter if possible
        rc = resume_checkpoint.strip()
        if os.path.isabs(rc):
            try:
                rc = os.path.relpath(rc, project_root)
            except: pass
        train_config["resume_from_checkpoint"] = rc.replace("\\", "/")
    
    # Use relative paths for manifest_path to avoid WebDataset gopen error on Windows (J:\)
    rel_train = train_manifest
    if os.path.isabs(rel_train):
        try: rel_train = os.path.relpath(rel_train, project_root)
        except: pass

    data_config = {
        "train": [{ "language_id": lang_code, "manifest_path": [rel_train.replace("\\", "/")], "repeat": int(repeat_factor) }]
    }
    
    if val_manifest:
        rel_val = val_manifest
        if os.path.isabs(rel_val):
            try: rel_val = os.path.relpath(rel_val, project_root)
            except: pass
        data_config["dev"] = [{ "language_id": lang_code, "manifest_path": [rel_val.replace("\\", "/")], "repeat": 1 }]

    t_cfg_path = out_exp / "train_config.json"
    d_cfg_path = out_exp / "data_config.json"
    
    with open(t_cfg_path, "w") as f: json.dump(train_config, f, indent=2)
    with open(d_cfg_path, "w") as f: json.dump(data_config, f, indent=2)
    
    cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "-m", "omnivoice.cli.train",
        "--train_config", str(t_cfg_path),
        "--data_config", str(d_cfg_path),
        "--output_dir", str(out_exp)
    ]
    
    print(f"Starting training...\nCommand: {' '.join(cmd)}\n", file=sys.stderr)
    def run_train():
        global training_process
        is_win = sys.platform == "win32"
        training_process = subprocess.Popen(cmd, shell=is_win)
        training_process.wait()
        play_done_chime()
        print(f"\nTraining finished (Code {training_process.returncode}).", file=sys.stderr)
        
    threading.Thread(target=run_train, daemon=True).start()
    return f"Training task '{output_name}' started in background."

def stop_training():
    global training_process
    if training_process is not None and training_process.poll() is None:
        if os.name == 'nt':
            # Force kill the entire process tree on Windows (PID + children)
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(training_process.pid)], capture_output=True)
            except:
                training_process.kill()
        else:
            training_process.terminate()
            
        print("Training terminated by user.", file=sys.stderr)
        return "Training stopped."
    return "No training running."

def launch_tensorboard(output_name):
    if not output_name:
        return "Please select or type an Output Directory Name first."
    tb_dir = project_root / "exp" / output_name / "tensorboard"
    if not tb_dir.exists():
        tb_dir = project_root / "exp"
    subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", str(tb_dir), "--port", "6006"])
    import webbrowser, time
    time.sleep(2)
    webbrowser.open("http://localhost:6006")
    return "TensorBoard launched at http://localhost:6006"


# --- GUI Layout ---

# Add CSS classes matching the reference script
css = """
.title-section { border-bottom: 2px solid #e5e7eb; margin-bottom: 20px; padding-bottom: 10px; }
.tabs { margin-top: 10px; }
.form-section { padding: 15px; border-radius: 8px; background-color: rgba(128,128,128,0.05); }
.input-field { margin-bottom: 15px; }
.button-primary { background-color: #2563eb !important; color: white !important; }
.button-stop { background-color: #ef4444 !important; color: white !important; }
"""

with gr.Blocks(title="OmniVoice - Simple GUI | Inference + LoRa Training") as app:
    
    # --- Initial State for Samples ---
    sample_choices = get_sample_choices()
    default_sample_name = sample_choices[0] if sample_choices else None
    default_audio, default_text = load_sample(default_sample_name)
    
    # --- Title Section ---
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=4):
            gr.Markdown("# 🚀 OmniVoice Universal GUI")
            gr.Markdown("Stabilized and optimized for Windows - Unified Model & LoRA loader")
        with gr.Column(scale=1):
            gr.Markdown("""
            [📖 Documentation](https://github.com/k2-fsa/OmniVoice)
            """)

    with gr.Tabs(elem_classes="tabs") as tabs:
        # === 1. Prep Samples Tab ===
        with gr.Tab("🎙️ Prep Samples", id="tab_prep_samples") as tab_prep_samples:
            gr.Markdown(
                "### 🎙️ Reference Audio Library\n"
                "Manage and prepare audio samples for voice cloning. You can record, upload, transcribe, and organize your samples here.\n\n"
                "**Guide:**\n"
                "1. Use **Single Editor** to add or edit individual reference audios.\n"
                "2. Once uploaded, click **✨ Transcribe** to automatically get the text via Faster-Whisper.\n"
                "3. Review and edit the transcription manually if needed.\n"
                "4. Assign a **Sample ID** and click **💾 Save Sample** to add it to your permanent library."
            )
            
            with gr.Row():
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown("#### 📂 Your Samples")
                    sample_dropdown = gr.Dropdown(
                        choices=sample_choices,
                        value=default_sample_name,
                        label="Select Sample",
                        interactive=True
                    )
                    refresh_samples_btn = gr.Button("🔄 Refresh List", size="sm")
                    
                
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 🎙️ Transcription & Editor")

                    gr.Markdown(
                        "### 🎙️ Add or Edit Audio\n"
                        "Use the **'X'** (top right of the player) to clear the preview and drag or click to upload a new audio.\n"
                        "Once uploaded, click **✨ Transcribe** to get the text, then **💾 Save Sample** to add it to your library."
                    )
                    
                    
                    prep_audio_player = gr.Audio(label="Audio Editor (3-10s recommended - Use Trim icon to edit)", type="filepath", interactive=True, value=default_audio)
                    prep_transcription = gr.Textbox(
                        label="Reference Text / Transcription",
                        placeholder="Transcription will appear here, or enter/edit text manually...",
                        lines=4,
                        interactive=True,
                        value=default_text
                    )
                    with gr.Row():
                        transcribe_prep_btn = gr.Button("🔍 Transcribe", variant="secondary")
                        prep_whisper_model = gr.Dropdown(
                            label="🛰️ Whisper Processor",
                            choices=list(WHISPER_MODELS.keys()),
                            value="large-v3 (~10 GB VRAM)",
                            scale=1
                        )
                        save_sample_name = gr.Textbox(label="Sample ID", placeholder="e.g. news_anchor_1", scale=2, value=default_sample_name)
                        save_sample_btn = gr.Button("💾 Save Sample", variant="primary", scale=1)

                    
                    prep_op_status = gr.Textbox(label="Operation Status", interactive=False)

            def on_sample_change(name):
                audio, text = load_sample(name)
                return gr.update(value=audio), gr.update(value=text), gr.update(value=name)

            sample_dropdown.change(on_sample_change, inputs=[sample_dropdown], outputs=[prep_audio_player, prep_transcription, save_sample_name])
            refresh_samples_btn.click(lambda: gr.update(choices=get_sample_choices()), outputs=[sample_dropdown])
            
            transcribe_prep_btn.click(fn=recognize_audio, inputs=[prep_audio_player, prep_whisper_model], outputs=[prep_transcription])
            save_sample_btn.click(fn=save_prep_sample, inputs=[prep_audio_player, save_sample_name, prep_transcription], outputs=[prep_op_status]).then(
                fn=lambda: gr.update(choices=get_sample_choices()), outputs=[sample_dropdown]
            )

# === 2. Inference Tab ===
        with gr.Tab("🔊 Inference") as tab_infer:
            gr.Markdown("### 🎙️ Unified Voice Synthesis")
            
            with gr.Row():
                # --- Left Column: Reference & Voice Selection ---
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown("#### 🎙️ Model & Reference")
                    with gr.Row():
                        infer_model_select = gr.Dropdown(
                            label="OmniVoice Model", 
                            choices=list(OMNIVOICE_MODELS.keys()) + [(f"{ckpt[0]} (Trained LoRa)", ckpt[0]) for ckpt in scan_lora_checkpoints(with_info=True)], 
                            value="OmniVoice (Official)",
                            info="Select between OmniVoice models and trained LoRAs.",
                            scale=4
                        )
                        refresh_model_btn = gr.Button("🔄", scale=1, min_width=50)

                    with gr.Row():
                        infer_whisper_model = gr.Dropdown(
                            label="🛰️ Whisper Processor",
                            choices=list(WHISPER_MODELS.keys()),
                            value="large-v3 (~10 GB VRAM)",
                            info="Model for transcribing reference audio if needed.",
                            scale=5
                        )

                    with gr.Row():
                        infer_sample_select = gr.Dropdown(
                            choices=get_sample_choices(),
                            value="None",
                            label="Quick Sample Select",
                            info="Load a reference from your 'samples' library.",
                            scale=4
                        )
                        refresh_infer_sample_btn = gr.Button("🔄", scale=1, min_width=50)
                    
                    infer_ref_audio = gr.Audio(label="Reference Audio (3-10s recommended)", type="filepath", value=default_audio)
                    with gr.Row():
                        infer_use_ref_text = gr.Checkbox(
                            label="Use Reference Text (Transcription)", 
                            value=True,
                            info="Disabling transcription may improve quality, diction, and pronunciation in some cases / languages (avoiding slurring or skipped words)."
                        )
                    
                    infer_ref_text = gr.Textbox(
                        label="Reference Text / Transcription", 
                        placeholder="Automatic transcription if enabled and left empty...", 
                        lines=2,
                        value=default_text,
                        interactive=True
                    )
                    
                    gr.Markdown("#### 🎭 Voice Design (Speaker Attributes)")
                    with gr.Row():
                        infer_instruct_en = gr.Dropdown(
                            label="English Instructs",
                            choices=[
                                "male", "female", 
                                "child", "teenager", "young adult", "middle-aged", "elderly",
                                "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
                                "whisper",
                                "american accent", "australian accent", "british accent", "canadian accent", "indian accent", "chinese accent", "korean accent", "japanese accent", "portuguese accent", "russian accent"
                            ],
                            multiselect=True,
                            allow_custom_value=True,
                            info="Attributes from Gender, Age, Pitch, Style, Accent."
                        )
                        infer_instruct_zh = gr.Dropdown(
                            label="Chinese Instructs (中文指令)",
                            choices=[
                                "男", "女",
                                "儿童", "少年", "青年", "中年", "老年",
                                "极低音调", "低音调", "中音调", "高音调", "极高音调",
                                "耳语",
                                "河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"
                            ],
                            multiselect=True,
                            allow_custom_value=True,
                            info="Attributes like Dialect, Age, Gender."
                        )
                    
                    gr.Markdown("#### ✨ Emotional & Narrative Tags (Text Modifiers)")
                    with gr.Row():
                        infer_instruct_tags = gr.Dropdown(
                            label="Speech Tags",
                            choices=[
                                ("[singing] (exclusive to singing variant)", "[singing]"),
                                ("[happy] (exclusive to singing variant)", "[happy]"),
                                ("[sad] (exclusive to singing variant)", "[sad]"),
                                ("[angry] (exclusive to singing variant)", "[angry]"),
                                ("[excited] (exclusive to singing variant)", "[excited]"),
                                ("[calm] (exclusive to singing variant)", "[calm]"),
                                ("[nervous] (exclusive to singing variant)", "[nervous]"),
                                ("[whisper] (exclusive to singing variant)", "[whisper]"),
                                "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]", "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]"
                            ],
                            multiselect=True,
                            allow_custom_value=True,
                            info="[singing], basic emotion tags, and their combinations are exclusive to the Singing variant model.",
                            scale=1
                        )
                    
                    with gr.Accordion("💡 OmniVoice Guide", open=False):
                        gr.Markdown("""
                        ### 🎤 Voice Design (instruct)
                        As per `docs/voice-design.md`, describe the **speaker**. Attributes are comma-separated.
                        
                        - **Categories**: Gender, Age, Pitch, Style, Accent/Dialect.
                        - **Example (EN)**: `female, young adult, high pitch, british accent`
                        - **Example (ZH)**: `女, 青年, 高音调, 四川话` (Separated by comma or `，`).
                        
                        ### ✨ Speech Tags (Text Prepending)
                        These tags are specific to certain models. They are added to the **target text**.
                        
                        **Singing Variant Features (ModelsLab/omnivoice-singing):**
                        - **[singing] tag** — sung speech / nursery-style melodic vocals
                        - **Emotion tags** — `[happy]`, `[sad]`, `[angry]`, `[excited]`, `[calm]`, `[nervous]`, `[whisper]`
                        - **Combined tags** — e.g. `[singing] [happy] ...` or `[singing] [sad] ...`
                        
                        **General Tags (Common across models):**
                        - `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[surprise-wa]`, etc.
                        
                        ### 🗣️ Pronunciation Control
                        - **English (CMU)**: `He plays the [B EY1 S] guitar.`
                        - **Chinese (Pinyin)**: `严重SHE2本了`
                        """)
                    
                    infer_control_display = gr.Markdown("")
                    infer_control = gr.Textbox(visible=False)
                    
                    # Update final display when dropdowns change
                    def update_instruct_visual(en_list, zh_list, tags_list):
                        parts = []
                        if en_list: parts.append(", ".join(en_list))
                        if zh_list: parts.append("，".join(zh_list))
                        instruct_str = ", ".join(parts)
                        
                        tags_str = " ".join(tags_list) if tags_list else ""
                        
                        display = ""
                        if tags_str:
                            display += f"✨ **Text Tags:** `{tags_str}`\n\n"
                        if instruct_str:
                            display += f"🎭 **Voice Attributes (Instruct):** `{instruct_str}`"
                        
                        if not display:
                            display = "🌑 **Auto Voice Mode**"
                            
                        return instruct_str, display
                        
                    infer_instruct_en.change(update_instruct_visual, inputs=[infer_instruct_en, infer_instruct_zh, infer_instruct_tags], outputs=[infer_control, infer_control_display])
                    infer_instruct_zh.change(update_instruct_visual, inputs=[infer_instruct_en, infer_instruct_zh, infer_instruct_tags], outputs=[infer_control, infer_control_display])
                    infer_instruct_tags.change(update_instruct_visual, inputs=[infer_instruct_en, infer_instruct_zh, infer_instruct_tags], outputs=[infer_control, infer_control_display])

                # --- Right Column: Content & Generation Settings ---
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown("#### ✍️ Target Speech")
                    with gr.Row():
                        infer_text = gr.Textbox(
                            label="Target Text", 
                            placeholder="Enter the text you want the AI to speak...", 
                            lines=10,
                            value="[laughter] Hello! I can speak with any voice you provide as a reference.",
                            scale=4
                        )
                    
                    with gr.Accordion("⚙️ Decoding & Sampling Parameters", open=False):
                        with gr.Row():
                            infer_steps = gr.Slider(label="Inference Steps (num_step)", minimum=1, maximum=64, value=32, step=1)
                            infer_cfg = gr.Slider(label="CFG Scale (guidance_scale)", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
                        with gr.Row():
                            infer_t_shift = gr.Slider(label="Time-step shift (t_shift)", minimum=0.0, maximum=1.0, value=0.1, step=0.05)
                            infer_pos_temp = gr.Slider(label="Position Temp", minimum=0.0, maximum=10.0, value=5.0, step=0.5)
                        with gr.Row():
                            infer_class_temp = gr.Slider(label="Token Temp", minimum=0.0, maximum=2.0, value=0.0, step=0.1)
                        with gr.Row():
                            infer_layer_penalty = gr.Slider(label="Layer Penalty", minimum=0.0, maximum=10.0, value=5.0, step=0.5)
                        with gr.Row():
                            infer_pp = gr.Checkbox(label="Pre-process Text (Normalization)", value=True)
                            infer_po = gr.Checkbox(label="Post-process Audio (Fading)", value=True)
                    
                    with gr.Accordion("📏 Duration & Chunking", open=False):
                        with gr.Row():
                            infer_duration = gr.Number(label="Fixed Duration (secs)", value=0, info="0 = use speed/auto")
                            infer_seed = gr.Number(label="Seed (-1 for Random)", value=-1, precision=0)
                        with gr.Row():
                            infer_chunk_dur = gr.Slider(label="Chunk Duration", minimum=5.0, maximum=30.0, value=15.0, step=1.0)
                            infer_chunk_thr = gr.Slider(label="Chunk Threshold", minimum=10.0, maximum=60.0, value=30.0, step=1.0)

                    infer_gen_btn = gr.Button("⚡ Generate Speech", variant="primary", size="lg", elem_classes="button-primary")
                    
                    with gr.Group():
                        infer_audio_out = gr.Audio(label="Generated Audio")
                        infer_status_out = gr.Textbox(label="System Status", interactive=False)
            
            # --- Unified Event Handlers ---
            def refresh_models():
                choices = list(OMNIVOICE_MODELS.keys()) + [(f"{ckpt[0]} (Trained LoRa)", ckpt[0]) for ckpt in scan_lora_checkpoints(with_info=True)]
                return gr.update(choices=choices)

            def smart_asr_unified(audio, current_text, use_ref, whisper_model):
                if not use_ref: 
                    return current_text # Don't auto-transcribe if disabled manually
                if current_text and current_text.strip():
                    return current_text
                return recognize_audio(audio, whisper_model)

            def on_use_ref_text_change(use_ref, sample_name, current_audio, current_text, whisper_model):
                if not use_ref:
                    return ""
                # Reload if enabling and empty
                if not current_text or not current_text.strip():
                    _, text = load_sample(sample_name)
                    if not text: # If still empty, try ASR
                        return recognize_audio(current_audio, whisper_model)
                    return text
                return current_text

            def on_sample_change(sample_name, use_ref):
                audio, text = load_sample(sample_name)
                if not use_ref:
                    return audio, ""
                return audio, text

            infer_sample_select.change(on_sample_change, inputs=[infer_sample_select, infer_use_ref_text], outputs=[infer_ref_audio, infer_ref_text])
            refresh_infer_sample_btn.click(lambda: gr.update(choices=get_sample_choices()), outputs=[infer_sample_select])
            refresh_model_btn.click(refresh_models, outputs=[infer_model_select])
            
            # Transcription triggered by audio change OR checkbox enable
            infer_ref_audio.change(fn=smart_asr_unified, inputs=[infer_ref_audio, infer_ref_text, infer_use_ref_text, infer_whisper_model], outputs=[infer_ref_text])
            infer_use_ref_text.change(fn=on_use_ref_text_change, inputs=[infer_use_ref_text, infer_sample_select, infer_ref_audio, infer_ref_text, infer_whisper_model], outputs=[infer_ref_text])
            
            infer_gen_btn.click(
                run_inference,
                inputs=[
                    infer_text,
                    infer_ref_audio,
                    infer_ref_text,
                    infer_model_select,
                    infer_cfg,
                    infer_steps,
                    infer_seed,
                    infer_control,
                    infer_duration,
                    infer_t_shift,
                    infer_pos_temp,
                    infer_class_temp,
                    infer_layer_penalty,
                    infer_chunk_dur,
                    infer_chunk_thr,
                    infer_use_ref_text,
                    infer_instruct_tags,
                    infer_pp,
                    infer_po
                ],
                outputs=[infer_audio_out, infer_status_out],
            )
            
        # === 3. Dataset Preparation Tab ===
        with gr.Tab("📂 Dataset Preparation", id="tab_dataset") as tab_dataset:
            gr.Markdown("### 🛠️ Dataset Creation & Auto-Transcription")

            gr.HTML("""
            <div style="background-color: rgba(255, 0, 0, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(255, 0, 0, 0.2); backdrop-filter: blur(10px); margin-bottom: 20px;">
                <h3 style="color: #ef4444; margin-top: 0;">⚠️ Caution</h3>
                <p style="margin-bottom: 0;">LoRA training for OmniVoice is experimental and results are not guaranteed. Since the base model is already highly fine-tuned, LoRA training carries a high risk of rapid overfitting. It is likely only effective for massive datasets, new languages, or highly specific voice styles. This section is intended for advanced users capable of precisely configuring hyperparameters for their specific data.</p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown(
                        "### 📂 Dataset Creation for Training\n"
                        "This workflow prepares your raw audio files for the LoRA fine-tuning process.\n\n"
                        "**This will:**\n"
                        "1. **Copy** audios to the `data/` folder.\n"
                        "2. **Normalize** volume to -0.95 dB (implicitly handled or skipped per OmniVoice).\n"
                        "3. **Convert** to WebDataset Shards.\n"
                        "4. **Auto-Transcribe** using Faster-Whisper Large-v3.\n"
                        "5. **Generate** `data.lst` manifest."
                    )
                    
                    gr.Markdown("#### 📁 Folder Selection")
                    with gr.Row():
                        src_folder = gr.Textbox(
                            label="Source Audio Folder",
                            placeholder="C:\\Users\\Voice\\AudioFolder...",
                            scale=4
                        )
                        explorer_btn = gr.Button("📂 Browse", scale=1)
                    
                    def open_folder_explorer():
                        try:
                            import tkinter as tk
                            from tkinter import filedialog
                            root = tk.Tk()
                            root.attributes('-topmost', 1)
                            root.withdraw()
                            path = filedialog.askdirectory(title="Select Source Audio Folder")
                            root.destroy()
                            return path if path else gr.update()
                        except:
                            return gr.update()
                            
                    explorer_btn.click(fn=open_folder_explorer, inputs=[], outputs=[src_folder])

                    with gr.Row():
                        dataset_name_input = gr.Textbox(
                            label="Target Dataset Name",
                            value="my_new_dataset",
                            info="This will create a folder in 'data/' with your audios and data.lst.",
                            scale=3
                        )
                        dataset_lang_iso = gr.Textbox(
                            label="Language ISO",
                            value="en",
                            scale=1,
                            info="ISO code for auto-transcription (e.g. en, es, zh). See: docs/languages.md"
                        )
                        dataset_whisper_model = gr.Dropdown(
                            label="🛰️ Whisper Processor",
                            choices=list(WHISPER_MODELS.keys()),
                            value="large-v3 (~10 GB VRAM)",
                            scale=3
                        )
                    
                    val_split_slider = gr.Slider(
                        label="Validation Split Ratio",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.05,
                        info="Ratio of dataset to use for validation (e.g. 0.1 = 10%)."
                    )
                    
                    batch_size_slider = gr.Slider(
                        label="Whisper Batch Size",
                        minimum=1,
                        maximum=64,
                        value=16,
                        step=1,
                        info="Batch size for Faster-Whisper batched inference (if supported)."
                    )
                    
                    prep_btn = gr.Button("Process & Transcribe", variant="primary", elem_classes="button-primary")
                    
                with gr.Column(scale=3, elem_classes="form-section"):
                    gr.Markdown("#### 📊 Processing Status")
                    prep_status_out = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )



            prep_btn.click(
                fn=prepare_voxcpm_dataset,
                inputs=[src_folder, dataset_name_input, val_split_slider, batch_size_slider, dataset_lang_iso, dataset_whisper_model],
                outputs=[prep_status_out]
            )

        # === 4. Training Tab ===
        with gr.Tab("🚀 Training") as tab_train:
            gr.Markdown("### 🎯 Model Training Configuration")

            gr.HTML("""
            <div style="background-color: rgba(255, 0, 0, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(255, 0, 0, 0.2); backdrop-filter: blur(10px); margin-bottom: 20px;">
                <h3 style="color: #ef4444; margin-top: 0;">⚠️ Caution</h3>
                <p style="margin-bottom: 0;">LoRA training for OmniVoice is experimental and results are not guaranteed. Since the base model is already highly fine-tuned, LoRA training carries a high risk of rapid overfitting. It is likely only effective for massive datasets, new languages, or highly specific voice styles. This section is intended for advanced users capable of precisely configuring hyperparameters for their specific data.</p>
            </div>
            """)


            with gr.Row():
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 📁 Model & Dataset Selection")
                    
                    train_model_select = gr.Dropdown(
                        label="OmniVoice Foundation Model",
                        choices=list(OMNIVOICE_MODELS.keys()),
                        value="OmniVoice (Official)",
                        info="System will download the model automatically from HuggingFace if not present."
                    )

                    with gr.Row():
                        train_manifest = gr.Dropdown(
                            label="Train Manifest (data.lst)",
                            choices=scan_datasets(),
                            value=scan_datasets()[0] if scan_datasets() else None,
                            allow_custom_value=True,
                            scale=6,
                            info="Select a manifest from 'data/' folder."
                        )
                        lang_code = gr.Textbox(
                            label="Language ISO",
                            value="en",
                            scale=2,
                            info="ISO code (e.g. en, es, zh, ja, ko, etc). See: docs/languages.md"
                        )
                        refresh_train_btn = gr.Button("🔄", scale=1, min_width=50)

                    with gr.Row():
                        val_manifest = gr.Dropdown(
                            label="Validation Manifest (Optional)",
                            choices=scan_datasets(),
                            value=None,
                            allow_custom_value=True,
                            info="Optional validation manifest. Leave empty if not used.",
                            scale=8
                        )
                        refresh_val_btn = gr.Button("🔄", scale=1, min_width=50)

                    with gr.Row():
                        vram_preset = gr.Radio(
                            label="Target GPU VRAM (Auto-Config)",
                            choices=["8 GB", "12 GB", "16 GB", "24 GB", "32 GB", "48 GB", "96 GB", "Small Dataset (Tuned - < 10 minutes)"],
                            value="24 GB",
                            elem_classes="input-field",
                            info="Select your GPU VRAM. 'Small Dataset' uses optimal anti-overfit parameters for small data. (You can adjust the preset settings to your liking below)"
                        )

                    with gr.Row():
                        ds_info = gr.Markdown("📊 *Dataset Stats: Select a manifest to view stats.*")

                    with gr.Row():
                        output_name = gr.Dropdown(
                            label="Output Directory Name",
                            choices=get_existing_lora_projects(),
                            value="",
                            allow_custom_value=True,
                            scale=8,
                            info="Training results will be saved in exp/[name]. Selecting an existing folder will resume training."
                        )
                        refresh_out_btn = gr.Button("🔄", scale=1, min_width=50)
                    
                    def refresh_manifests():
                        manifests = scan_datasets()
                        return gr.update(choices=manifests), gr.update(choices=manifests)

                    def refresh_projects():
                        return gr.update(choices=get_existing_lora_projects())

                    refresh_train_btn.click(refresh_manifests, outputs=[train_manifest, val_manifest])
                    refresh_val_btn.click(refresh_manifests, outputs=[train_manifest, val_manifest])
                    refresh_out_btn.click(refresh_projects, outputs=[output_name])

                    gr.Markdown("#### ⚙️ Core Hyperparameters")

                    with gr.Row():
                        lr = gr.Dropdown(
                            label="Learning Rate",
                            choices=["1e-3", "5e-4", "1e-4", "5e-5", "1e-5"],
                            value="1e-4",
                            allow_custom_value=True,
                            elem_classes="input-field",
                            info="Peak learning rate."
                        )
                        steps_num = gr.Number(
                            label="Total Steps",
                            value=250,
                            precision=0,
                            elem_classes="input-field",
                            info="Total training steps."
                        )
                        batch_tokens_num = gr.Dropdown(
                            label="Batch Tokens",
                            choices=["1024", "2048", "4096", "8192", "16384", "32768", "65536"],
                            value="4096",
                            allow_custom_value=True,
                            elem_classes="input-field",
                            info="Tokens per batch. Lower if you get OOM."
                        )
                        grad_accum = gr.Number(
                            label="Gradient Accumulation",
                            value=4,
                            precision=0,
                            elem_classes="input-field",
                            info="Steps before weight update. Higher = better stability/quality."
                        )
                        save_steps = gr.Number(
                            label="Save Every X Steps",
                            value=25,
                            precision=0,
                            elem_classes="input-field",
                            info="Checkpoint frequency. Also sends audio sample to Tensorboard."
                        )
                    with gr.Row():
                        warmup_ratio = gr.Number(
                            label="Warmup Ratio",
                            value=0.01,
                            elem_classes="input-field",
                            info="Proportion of training to warm up learning rate."
                        )
                        repeat_factor = gr.Number(
                            label="Dataset Repeat",
                            value=1,
                            precision=0,
                            elem_classes="input-field",
                            info="Number of times to loop the dataset."
                        )

                    with gr.Accordion("🎨 Eval Zone (Tensorboard Audio)", open=False, elem_classes="accordion"):
                        enable_eval = gr.Checkbox(
                            label="Enable Eval Zone",
                            value=False,
                            info="Uncheck to disable VRAM-heavy audio evolution samples during training."
                        )
                        eval_text = gr.Textbox(
                            label="Test Inference Text",
                            value="This is my voice evolution during training. I hope I sound like the reference soon!",
                            lines=2,
                            interactive=True,
                            info="Model will speak this text in Tensorboard samples."
                        )
                        eval_audio = gr.Audio(
                            label="Inference Reference Audio",
                            type="filepath"
                        )

                    with gr.Accordion("🔧 Advanced Settings", open=False, elem_classes="accordion"):
                        with gr.Row():
                            llm_name = gr.Textbox(
                                label="LLM Name or Path",
                                value="Qwen/Qwen3-0.6B",
                                info="Local LLM path or HuggingFace ID."
                            )
                            resume_checkpoint = gr.Textbox(
                                label="Resume from Checkpoint Path",
                                value="",
                                placeholder="exp/omnivoice/checkpoint-100000",
                                info="If provided, resumes training from this local checkpoint directory."
                            )

                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Training", variant="primary", elem_classes="button-primary")
                        stop_btn = gr.Button("⏹️ Stop Training", variant="stop", elem_classes="button-stop")
                        tb_btn = gr.Button("📊 TensorBoard", variant="secondary")

                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 📟 Training Process")
                    logs_out = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )

            start_btn.click(
                start_training,
                inputs=[
                    train_model_select,
                    train_manifest,
                    val_manifest,
                    output_name,
                    lr,
                    steps_num,
                    batch_tokens_num,
                    llm_name,
                    resume_checkpoint,
                    grad_accum,
                    save_steps,
                    eval_text,
                    eval_audio,
                    enable_eval,
                    lang_code,
                    warmup_ratio,
                    repeat_factor
                ],
                outputs=[logs_out],
            )
            stop_btn.click(stop_training, outputs=[logs_out])
            tb_btn.click(launch_tensorboard, inputs=[output_name], outputs=[logs_out])

            def on_auto_calc(manifest, vram):
                count, total_dur, avg_dur, total_tokens = calculate_dataset_stats(manifest)
                
                # 1. Base Tuned Hyperparameters (Optimized for small-dataset stability)
                lr = 1e-5
                repeat = 3
                warmup = 0.05
                save_steps = 100
                
                # 2. VRAM Scaling (Targeting 4096+ tokens if possible)
                # Note: tokens reduced by ~12% for each level to provide safety margin (OOM protection)
                if "8 GB" in vram:
                    tokens = 896
                    accum = 10
                elif "12 GB" in vram:
                    tokens = 1344
                    accum = 8
                elif "16 GB" in vram:
                    tokens = 1792
                    accum = 6
                elif "24 GB" in vram or "Small Dataset" in vram:
                    tokens = 3584
                    accum = 2
                elif "32 GB" in vram:
                    tokens = 7168
                    accum = 1
                elif "48 GB" in vram:
                    tokens = 28672
                    accum = 1
                elif "96 GB" in vram:
                    tokens = 57344
                    accum = 1
                else:
                    tokens = 4096
                    accum = 2

                # 3. Blended Dataset & VRAM Scaling
                # Effective Batch Size (EBS)
                ebs = tokens * accum
                ebs_factor = ebs / 4096.0 # 4096 is the ebs for the "Small Dataset" recipe
                
                hours = total_dur / 3600 if total_dur > 0 else (count * 5.0) / 3600
                
                if hours < 0.166: # < 10 minutes (Ultra-Stable Recipe Base)
                    lr_base = 1e-5
                    repeat = 3
                    save_steps = 100
                    warmup = 0.05
                else:
                    # Exponential growth based on data volume
                    # We grow from 1e-5 towards ~1e-4 based on hours
                    lr_base = 1e-5 * (2**(hours / 5.0)) # Double LR every 5 hours of data
                    lr_base = min(lr_base, 1e-4) # Base cap before VRAM scaling
                    
                    if hours < 1:
                        repeat = 2
                        save_steps = 150
                    elif hours < 10:
                        repeat = 1
                        save_steps = 250
                    else:
                        repeat = 1
                        save_steps = 500
                        warmup = 0.01 # Standard warmup for larger data

                # Final LR scaled by effective batch size (Linear Scaling Rule)
                lr = lr_base * ebs_factor
                lr = max(1e-6, min(lr, 5e-4)) # Safety training bounds

                # 4. Step Calculation (Independent of VRAM choice)
                # Calculates steps based on a virtual standard batch of 8192 tokens.
                # VRAM only scales the internal batch_tokens/accum, not the total updates.
                avg_tokens_per_sample = total_tokens / count if count > 0 else 1000
                if avg_tokens_per_sample < 200: avg_tokens_per_sample = 200
                
                virtual_batch = 8192 
                samples_per_virtual_step = virtual_batch / avg_tokens_per_sample
                if samples_per_virtual_step < 1: samples_per_virtual_step = 1
                
                suggested_steps = int((count * repeat) / samples_per_virtual_step)
                
                if hours < 0.166:
                    total_steps = max(min(suggested_steps, 800), 100)
                elif hours < 1:
                    total_steps = max(min(suggested_steps, 2000), 500)
                else:
                    total_steps = max(suggested_steps, 1500)
                
                total_steps = min(total_steps, 300000)

                # Format info string
                minutes = total_dur / 60 if total_dur > 0 else (count * 5.0) / 60
                info = f"📊 **Stats:** {count} clips | ~{minutes:.1f} min total | ~{total_tokens:,} tokens | Avg: {avg_dur:.1f}s\n"
                info += f"🪄 **Auto-Scaling:** {tokens*accum} batch layout ({vram}) | {repeat}x repeat | Steps: {total_steps}"
                if count < 50:
                    info += "\n⚠️ **Caution:** Tiny dataset found! Using ultra-stable parameters."

                lr_str = f"{lr:.0e}".replace("e-05", "e-5").replace("1e-04", "1e-4").replace("5e-05", "5e-5")
                return str(tokens), int(accum), lr_str, int(total_steps), int(save_steps), float(warmup), int(repeat), info

            vram_preset.change(
                on_auto_calc,
                inputs=[train_manifest, vram_preset],
                outputs=[batch_tokens_num, grad_accum, lr, steps_num, save_steps, warmup_ratio, repeat_factor, ds_info]
            )
            train_manifest.change(
                on_auto_calc,
                inputs=[train_manifest, vram_preset],
                outputs=[batch_tokens_num, grad_accum, lr, steps_num, save_steps, warmup_ratio, repeat_factor, ds_info]
            )

            # --- Final Cross-Tab Event Handlers ---
            dataset_lang_iso.change(lambda x: x, inputs=[dataset_lang_iso], outputs=[lang_code])


if __name__ == "__main__":
    os.makedirs("exp", exist_ok=True)
    app.queue().launch(
        server_name="127.0.0.1", 
        server_port=7860,
        inbrowser=True,
        css=css
    )
