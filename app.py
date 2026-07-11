import os
import re
import json
import platform
import subprocess
import toml
import glob
import math
from pathlib import Path
import sys
import gradio as gr
import psutil
from PIL import Image
import torch
import numpy as np
import onnxruntime as rt
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import shutil
from typing import Generator, List, Tuple
import warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

CSS = """
.gradio-container {
    position: relative !important;
}

#main-header {
    margin-bottom: -10px !important;
}

.attribution {
    position: absolute;
    right: 25px;
    top: 5px;
    z-index: 99999;
    pointer-events: none;
}

.author-text {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 16px;
    font-weight: 700;
    text-shadow: 0 0 15px rgba(187, 154, 247, 0.6);
    letter-spacing: 0.5px;
}

#log-container textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 14px !important;
    line-height: 1.3 !important;
    white-space: pre-wrap !important;
    overflow-x: hidden !important;
    height: 385px !important;
    border: 1px solid #2f334d !important;
}

*::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
*::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1) !important;
}
*::-webkit-scrollbar-thumb {
    background: #7aa2f7 !important;
    border-radius: 10px !important;
}
*::-webkit-scrollbar-thumb:hover {
    background: #bb9af7 !important;
}

footer {
    display: none !important;
}

"""

JS_SCROLL = """
function() {
    setTimeout(() => {
        const el = document.querySelector('#log-container textarea');
        if (el) {
            el.scrollTop = el.scrollHeight;
        }
    }, 50);
}
"""


LOG_BLACKLIST = [
    "triton not found",
    "flop counting will not work",
    "Lib\\site-packages\\torch\\utils\\flop_counter.py"
]


LOG_BOX__MAX_LINES = 16
GALLERY_HEIGHT = 440
MAX_LOG_LINES = 500


ROOT = Path(__file__).resolve().parent
PORTABLE_PYTHON = ROOT / "python_embeded" / "python.exe"

TRAIN_BASE = ROOT / "training"
OUTPUT_BASE = TRAIN_BASE / "output"
SETTINGS_FILE = TRAIN_BASE / "settings.json"
PRESETS_FILE = TRAIN_BASE / "presets.json"

TRAIN_DIR = TRAIN_BASE / "sd-scripts" 
TRAIN_SCRIPT = TRAIN_DIR / "anima_train_network.py"

training_process = None

# Module-level training state, decoupled from any single Gradio request/generator
# so a page refresh doesn't sever the connection to a still-running subprocess.
training_state_lock = threading.Lock()
training_active = False
training_log_lines = []
training_eta_text = ""
training_sample_dir = None
training_last_image_count = 0
training_log_file = None

for d in [TRAIN_BASE, OUTPUT_BASE]:
    d.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = {
    "trigger_word": "",
    "dataset_path": "",
    "dit_path": str(ROOT / "models" / "anima" / "dit" / "anima-preview.safetensors"),
    "qwen_path": str(ROOT / "models" / "anima" / "text_encoder" / "qwen_3_06b_base.safetensors"),
    "vae_path": str(ROOT / "models" / "anima" / "vae" / "qwen_image_vae.safetensors"),
    "network_rank": 32,
    "learning_rate": "1.0", # Prodigy default
    "optimizer": "Prodigy", # Prodigy default
    "training_steps": 2400,
    "save_steps": 300,
    "sample_steps": 300,
    "pos_prompt": "",
    "neg_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name",
    "width": 1024,
    "height": 1024,
    "sample_steps_gen": 30,
    "sample_cfg": 4.0,
    "sample_seed": 42,
    "train_seed": 42,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "side_min": 512,
    "side_max": 768,
    "tagger_gen_thresh": 0.35,
    "tagger_char_thresh": 0.85,
    "tagger_overwrite": False
}
def load_presets():
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("presets", [])
        except Exception:
            pass
    return []

def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    return settings

def save_settings(settings_dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4)

def auto_save_state(*args):
    keys = list(DEFAULT_SETTINGS.keys())
    current_state = dict(zip(keys, args))
    save_settings(current_state)


HIDDEN_SETTINGS = {
    "lr_scheduler": "cosine",
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "gradient_checkpointing": True,
    "network_module": "networks.lora_anima",
    "network_train_unet_only": True,
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "weighting_scheme": "none",
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "cache_text_encoder_outputs_to_disk": True,
    "cache_text_encoder_outputs": True,
    "sdpa": True,
    "weighting_scheme": "logit_normal",
    "max_data_loader_n_workers": 4,
    "persistent_data_loader_workers": True,
    "max_grad_norm": 1.0,
    "vae_batch_size": 1,
    "blocks_to_swap": 0,
    "sigmoid_scale": 1.3,
}


VALID_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

def count_dataset_images(dataset_path: str) -> int:
    # exposures-per-image formula: steps * batch * grad_accum / num_images
    path = Path(dataset_path)
    if not path.exists():
        return 0
    return len([f for f in path.glob('*') if f.is_file() and f.suffix.lower() in VALID_IMG_EXTS])

def analyze_dataset_resolution(dataset_path: str) -> Tuple[int, int]:
    path = Path(dataset_path)
    if not path.exists(): return 512, 768

    valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [f for f in path.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    if not image_files: return 512, 768

    max_area = 0
    max_side = 0

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                max_area = max(max_area, w * h)
                max_side = max(max_side, w, h)
        except Exception: continue

    # Calculate the side of the square that covers the area of the largest bucket
    # If the largest is 512x768, this yields 640.
    base_res = int(math.ceil(math.sqrt(max_area) / 64.0) * 64)
    max_bucket = int(math.ceil(max_side / 64.0) * 64)

    return base_res, max_bucket


def create_sample_prompts(project_name, trigger_word, pos_prompt, neg_prompt, width, height, steps_gen, cfg, seed, out_dir):
    prompt_path = out_dir / f"{project_name}_prompts.txt"
    trigger = trigger_word.strip()
    user_prompt = pos_prompt.strip().replace("\n", " ")
    
    if trigger and not user_prompt.startswith(trigger):
        actual_pos = f"{trigger}, {user_prompt}" if user_prompt else trigger
    else:
        actual_pos = user_prompt if user_prompt else trigger

    actual_neg = neg_prompt.strip().replace("\n", " ")
    prompt_str = f"{actual_pos} --n {actual_neg} --w {int(width)} --h {int(height)} --l {float(cfg)} --s {int(steps_gen)} --d {int(seed)}"
    with open(prompt_path, "w", encoding="utf-8") as f: f.write(prompt_str)
    return str(prompt_path)

def create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, out_dir, max_steps, batch_size, grad_acc):
    config_path = out_dir / f"{project_name}_dataset.toml"
    prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else None

    path = Path(dataset_path)
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    num_images = len([f for f in path.glob('*') if f.is_file() and f.suffix.lower() in valid_exts])

    if num_images == 0: num_images = 1

    effective_batch = int(batch_size) * int(grad_acc)
    calculated_repeats = (int(max_steps) * effective_batch) / num_images
    final_repeats = max(1, math.ceil(calculated_repeats))

    dataset_config = {
        "general": {
            "enable_bucket": True, 
            "min_bucket_reso": 256, 
            "max_bucket_reso": max_bucket, 
            "bucket_reso_steps": 64, 
            "bucket_no_upscale": True
        },
        "datasets": [{
            "resolution": base_res, 
            "subsets": [{
                "image_dir": Path(dataset_path).resolve().as_posix(), 
                "caption_extension": ".txt", 
                "num_repeats": final_repeats,
                "caption_prefix": prefix, 
                "keep_tokens": 1, 
                "caption_dropout_rate": 0.05
            }]
        }]
    }
    with open(config_path, "w", encoding="utf-8") as f: 
        toml.dump(dataset_config, f)
    return str(config_path)

def create_training_toml(project_name, config_save_dir, actual_output_dir, rank, lr, optimizer, max_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc, skip_cache_check=False):
    config_path = config_save_dir / f"{project_name}_training.toml"

    network_rank = int(rank)
    network_alpha = network_rank
    
    opt_args = ["weight_decay=0.01"]
    if optimizer == "Prodigy":
        scheduler = "constant"
        lr_warmup = None  # schedule-free; warmup handled internally by safeguard_warmup
        opt_args = ["decouple=True", "weight_decay=0.01", "d_coef=1.0", "use_bias_correction=True", "safeguard_warmup=True", "betas=0.9,0.99"]
    else:
        scheduler = "cosine"
        lr_warmup = 0.1  # 10% warmup — backend accepts float ratio

    training_config = {
        "pretrained_model_name_or_path": Path(models["dit_path"]).resolve().as_posix(),
        "qwen3": Path(models["qwen_path"]).resolve().as_posix(),
        "vae": Path(models["vae_path"]).resolve().as_posix(),
        "network_module": HIDDEN_SETTINGS["network_module"],
        "network_dim": int(rank),
        "network_alpha": network_alpha,
        "network_train_unet_only": HIDDEN_SETTINGS["network_train_unet_only"],
        "gradient_checkpointing": HIDDEN_SETTINGS["gradient_checkpointing"],
        "max_grad_norm": 1.0,
        "learning_rate": float(lr),
        "optimizer_type": optimizer,
        "optimizer_args": opt_args,
        "lr_scheduler": scheduler,
        **({"lr_warmup_steps": lr_warmup} if lr_warmup is not None else {}),
        "max_train_steps": int(max_steps),
        "train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_acc),
        "mixed_precision": HIDDEN_SETTINGS["mixed_precision"],
        "output_dir": actual_output_dir.resolve().as_posix(),
        "output_name": project_name,
        "save_every_n_steps": int(save_steps),
        "sample_every_n_steps": int(sample_steps),
        "sample_prompts": Path(prompt_path).resolve().as_posix(),
        "sample_sampler": "euler",
        "timestep_sampling": HIDDEN_SETTINGS["timestep_sampling"],
        "discrete_flow_shift": HIDDEN_SETTINGS["discrete_flow_shift"],
        "sigmoid_scale": HIDDEN_SETTINGS["sigmoid_scale"],
        "weighting_scheme": HIDDEN_SETTINGS["weighting_scheme"],
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "save_state": True,
        "save_last_n_steps_state": 1,
        "save_state_on_train_end": True,
        "skip_cache_check": bool(skip_cache_check),
        "attn_mode": "sdpa",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "max_data_loader_n_workers": 4,
        "vae_chunk_size": 32,
        "vae_disable_cache": True,
        "seed": int(train_seed),
    }
    with open(config_path, "w", encoding="utf-8") as f: toml.dump(training_config, f)
    return str(config_path)


def get_latest_images(sample_dir):
    if not sample_dir.exists(): return []
    images = glob.glob(str(sample_dir / "*.png")) + glob.glob(str(sample_dir / "*.jpg")) + glob.glob(str(sample_dir / "*.webp"))
    images.sort(key=os.path.getmtime, reverse=True)
    
    return [(img, Path(img).name) for img in images]


# ==========================================
# CROPPER CLASS (U2Net Head-First + Buckets)
# ==========================================
class SmartCropper:
    def __init__(self) -> None:
        self.session: rt.InferenceSession | None = None
        self.model_path: Path = ROOT / "models" / "u2net" / "u2net.onnx"

    def load_model(self) -> str:
        if self.session is not None: 
            return "Already loaded"
        providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
        self.session = rt.InferenceSession(str(self.model_path), providers=providers)
        return "GPU" if "CUDA" in self.session.get_providers()[0] else "CPU"

    def get_valid_buckets(self, side_min: int, side_max: int) -> List[Tuple[int, int]]:
        s_min, s_max = int(side_min), int(side_max)
        buckets = {(s_min, s_min)} 
        for s in range(s_min + 64, s_max + 64, 64):
            buckets.add((s_min, s))
            buckets.add((s, s_min))
        return sorted(list(buckets), key=lambda x: (x[0] * x[1]))

    def get_best_bucket(self, w: int, h: int, buckets: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], str]:
        orig_ar = w / h
        log_orig = math.log(orig_ar)
        best_b = buckets[0]
        min_diff = float('inf')
        
        for b in buckets:
            b_ar = b[0] / b[1]
            diff = abs(math.log(b_ar) - log_orig)
            if diff < min_diff:
                min_diff = diff
                best_b = b
        
        reason = f"AR: {orig_ar:.2f} -> {best_b[0]/best_b[1]:.2f}"
        return best_b, reason

    def process_image(self, original_img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h_orig, w_orig = original_img.shape[:2]
        
        if abs((w_orig / h_orig) - (tw / th)) < 0.01:
            return cv2.resize(original_img, (tw, th), interpolation=cv2.INTER_AREA)

        low_res_scale = 1024 / max(h_orig, w_orig)
        img_sm = cv2.resize(original_img, (int(w_orig*low_res_scale), int(h_orig*low_res_scale)), interpolation=cv2.INTER_AREA)
        
        input_size = 320
        img_inp = cv2.resize(img_sm, (input_size, input_size), interpolation=cv2.INTER_AREA)
        img_inp = img_inp.astype(np.float32) / 255.0
        img_inp -= [0.485, 0.456, 0.406]
        img_inp /= [0.229, 0.224, 0.225]
        input_tensor = np.expand_dims(np.transpose(img_inp, (2, 0, 1)), 0)
        
        mask = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0][0][0]
        mask = cv2.resize(mask, (img_sm.shape[1], img_sm.shape[0]))
        
        y_idx, x_idx = np.where(mask > 0.15)
        if len(y_idx) > 0:
            top_y = int(np.min(y_idx) / low_res_scale)
            center_x = int(np.mean(x_idx) / low_res_scale)
        else:
            top_y, center_x = h_orig // 4, w_orig // 2

        scale = max(tw / w_orig, th / h_orig)
        cw, ch = int(tw / scale), int(th / scale)
        y1 = max(0, min(top_y - int(ch * 0.05), h_orig - ch))
        x1 = max(0, min(center_x - cw // 2, w_orig - cw))

        return cv2.resize(original_img[y1:y1+ch, x1:x1+cw], (tw, th), interpolation=cv2.INTER_AREA)

    def unload_model(self) -> None:
        self.session = None

smart_cropper = SmartCropper()

def generate_bucket_summary(bucket_counts: dict) -> List[str]:
    """Generates a nicely formatted text report on image distribution."""
    if not bucket_counts:
        return []
        
    lines = ["", "📊 Bucket Distribution Summary:"]
    total = 0
    # Sort keys (WxH tuples) by area for pretty printing
    for bkt in sorted(bucket_counts.keys(), key=lambda x: x[0] * x[1]):
        count = bucket_counts[bkt]
        total += count
        lines.append(f"  ├─ {bkt[0]}x{bkt[1]}: {count} images")
    lines.append(f"  └─ Total Dataset Size: {total} images")
    lines.append("")
    return lines

def run_smart_crop_ui(dataset_dir: str, side_min: float, side_max: float, current_logs: str) -> Generator[str, None, None]:
    log_lines = current_logs.split('\n') if current_logs else []
    path = Path(dataset_dir)
    
    if not dataset_dir or not path.exists():
        log_lines.append("❌ Error: Dataset path invalid!")
        yield "\n".join(log_lines)
        return

    backup_dir = path / "original_images"
    backup_dir.mkdir(parents=True, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    available_buckets = smart_cropper.get_valid_buckets(int(side_min), int(side_max))
    log_lines.append(f"📋 Buckets configured: {len(available_buckets)}")
    yield "\n".join(log_lines)

    # 1. Move new original images
    for f in path.glob('*'):
        if f.is_file() and f.suffix.lower() in valid_exts and f.parent != backup_dir:
            try:
                if f.suffix.lower() == '.png':
                    with Image.open(f) as img:
                        if img.size in available_buckets:
                            continue
            except Exception: pass
            shutil.move(str(f), str(backup_dir / f.name))

    # 2. Task preparation (Idempotency + Statistics collection)
    originals = [f for f in backup_dir.glob('*') if f.suffix.lower() in valid_exts]
    tasks = []
    bucket_counts = {}
    skipped = 0

    for orig in originals:
        try:
            with Image.open(orig) as img:
                w, h = img.size
            (tw, th), reason = smart_cropper.get_best_bucket(w, h, available_buckets)
            out_p = path / f"{orig.stem}.png"
            
            if out_p.exists():
                with Image.open(out_p) as check:
                    if check.size == (tw, th):
                        skipped += 1
                        bucket_counts[(tw, th)] = bucket_counts.get((tw, th), 0) + 1
                        continue
            tasks.append((orig, out_p, tw, th))
        except Exception: continue

    if not tasks:
        log_lines.append(f"✅ All {skipped} images are already bucketed correctly.")
        log_lines.extend(generate_bucket_summary(bucket_counts))
        yield "\n".join(log_lines)
        return

    # 3. Multi-threaded inference
    try:
        log_lines.append(f"🚀 Model: {smart_cropper.load_model()}")
        log_lines.append(f"⚙️ Processing {len(tasks)} images (Skipped: {skipped})...")
        yield "\n".join(log_lines)

        processed = 0
        lock = threading.Lock()

        def worker(task_data: Tuple[Path, Path, int, int]) -> str | None:
            in_p, out_p, tw, th = task_data
            nonlocal processed
            try:
                img = cv2.imread(str(in_p))
                if img is not None:
                    res = smart_cropper.process_image(img, tw, th)
                    cv2.imwrite(str(out_p), res, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                
                with lock:
                    processed += 1
                    # Add successful crop to statistics
                    bucket_counts[(tw, th)] = bucket_counts.get((tw, th), 0) + 1
                return None
            except Exception as e:
                return f"⚠️ {in_p.name}: {str(e)}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, t) for t in tasks]
            
            for future in as_completed(futures):
                err = future.result()
                if err: log_lines.append(err)
                
                if processed % 5 == 0 or processed == len(tasks):
                    msg = f"🔄 Progress: {processed}/{len(tasks)}"
                    if "Progress:" in log_lines[-1]: log_lines[-1] = msg
                    else: log_lines.append(msg)
                    yield "\n".join(log_lines)

        log_lines.append(f"✅ Done! Processed: {processed}")
        
    except Exception as e:
        log_lines.append(f"❌ Error: {str(e)}")
    finally:
        smart_cropper.unload_model()
        log_lines.extend(generate_bucket_summary(bucket_counts))
        yield "\n".join(log_lines)


# ==========================================
# TAGGER CLASS (WD14)
# ==========================================
class WDTagger:
    def __init__(self):
        self.model = None
        self.tag_names = []
        self.general_indexes = []
        self.character_indexes = []
        self.target_size = 448
        self.model_dir = ROOT / "models" / "wd-eva02-large-tagger-v3"
        self.model_path = self.model_dir / "model.onnx"
        self.csv_path = self.model_dir / "selected_tags.csv"
        self.kaomojis = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]

    def load_model(self):
        if self.model is not None:
            return "Already loaded"
            
        if not self.model_path.exists() or not self.csv_path.exists():
            raise FileNotFoundError(f"Model or CSV not found in {self.model_dir}.")

        df = pd.read_csv(self.csv_path)
        name_series = df["name"].map(lambda x: str(x).replace("_", " ") if pd.notna(x) and str(x) not in self.kaomojis else str(x))
        self.tag_names = name_series.tolist()
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])

        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        
        self.model = rt.InferenceSession(str(self.model_path), providers=providers)
        current_provider = self.model.get_providers()[0]
        return "GPU (CUDA)" if "CUDA" in current_provider else "CPU (Slow Mode)"

    def preprocess(self, image):
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        image = canvas.convert("RGB")
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        if max_dim != self.target_size:
            padded_image = padded_image.resize((self.target_size, self.target_size), Image.BICUBIC)
        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1] # BGR
        return np.expand_dims(image_array, axis=0)

    def predict(self, image, gen_thresh, char_thresh):
        image_array = self.preprocess(image)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: image_array})
        preds = outputs[0][0]
        
        general_tags = [self.tag_names[i] for i in self.general_indexes if i < len(preds) and preds[i] > gen_thresh]
        char_tags = [self.tag_names[i] for i in self.character_indexes if i < len(preds) and preds[i] > char_thresh]
        
        char_tags = [char.replace("(", r"\(").replace(")", r"\)") for char in char_tags]
        general_tags = sorted(general_tags, key=lambda x: preds[self.tag_names.index(x)], reverse=True)
        
        final_list = []
        if char_tags: final_list.append(", ".join(char_tags))
        if general_tags: final_list.append(", ".join(general_tags).replace("(", r"\(").replace(")", r"\)"))
        return ", ".join(final_list)

    def unload_model(self):
        import gc
        import torch
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()    

wd_tagger = WDTagger()

# ==========================================
# GUI EXECUTOR FUNCTIONS
# ==========================================

def run_auto_tagging(dataset_dir, gen_thresh, char_thresh, overwrite, current_logs):
    log_lines = current_logs.split('\n') if current_logs else []
    
    if not dataset_dir or not os.path.exists(dataset_dir):
        log_lines.append("❌ Tagger Error: Dataset path invalid!")
        yield "\n".join(log_lines)
        return

    valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    all_files = [f for f in Path(dataset_dir).glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    
    image_files = []
    skipped = 0
    for f in all_files:
        if f.with_suffix('.txt').exists() and not overwrite:
            skipped += 1
        else:
            image_files.append(f)

    if not image_files and skipped == 0:
        log_lines.append("❌ Tagger Error: No images found!")
        yield "\n".join(log_lines)
        return
        
    if not image_files and skipped > 0:
        log_lines.append(f"ℹ️ All images already have captions. Skipped: {skipped}")
        yield "\n".join(log_lines)
        return

    log_lines.append(f"🏷️ Initializing WD-Tagger (Multi-threaded)...")
    yield "\n".join(log_lines)
    
    try:
        mode = wd_tagger.load_model()
        log_lines.append(f"🚀 Model loaded using: {mode}")
        log_lines.append(f"⚙️ Processing {len(image_files)} images in 4 threads... (Skipped: {skipped})")
        yield "\n".join(log_lines)

        processed_count = 0
        total_to_process = len(image_files)
        lock = threading.Lock()

        def process_single_image(img_path):
            nonlocal processed_count
            try:
                with Image.open(img_path) as img:
                    tags = wd_tagger.predict(img, gen_thresh, char_thresh)
                with open(img_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write(tags)
                with lock:
                    processed_count += 1
                return None
            except Exception as e:
                return f"⚠️ Error {img_path.name}: {str(e)}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_image, path) for path in image_files]
            
            for future in as_completed(futures):
                error_msg = future.result()
                if error_msg: log_lines.append(error_msg)
                
                if processed_count % 5 == 0 or processed_count == total_to_process:
                    msg = f"🔄 Tag Progress: {processed_count}/{total_to_process} images."
                    if "Tag Progress:" in log_lines[-1]: log_lines[-1] = msg
                    else: log_lines.append(msg)
                    yield "\n".join(log_lines)

        log_lines.append(f"✅ Tagging complete! Processed: {processed_count} | Skipped: {skipped}")
        
    except Exception as e:
        log_lines.append(f"❌ Critical Tagger Error: {str(e)}")
        
    finally:
        wd_tagger.unload_model()
        log_lines.append("🧹 VRAM cleared.")
        yield "\n".join(log_lines)


def open_dataset_folder_ui(dataset_dir, current_logs):
    log_lines = current_logs.split('\n') if current_logs else []
    if not dataset_dir or not os.path.exists(dataset_dir):
        log_lines.append("❌ Error: Dataset path invalid or empty!")
        return "\n".join(log_lines)
    
    target_dir = Path(dataset_dir)
    if platform.system() == "Windows": os.startfile(target_dir)
    else: subprocess.Popen(["xdg-open", str(target_dir)])
    
    log_lines.append(f"📁 Opened Dataset folder.")
    return "\n".join(log_lines)


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

_STEP_PATTERN = re.compile(r"(\d+)/(\d+)")
# T4 ETA: matches tqdm rate lines — e.g. "500/1000 [05:12<05:12, 1.60it/s]"
_ETA_RE = re.compile(r'(\d+)/(\d+)\s*\[[\d:]+<([\d:?]+),\s*([\d.]+)(it/s|s/it)\]')


def _training_worker(log_file_path):
    """Runs in a background thread, independent of any Gradio request, so a page
    refresh (which cancels the in-flight click generator) can't stop log/ETA/preview
    updates or leave the subprocess's stdout pipe undrained."""
    global training_process, training_active, training_eta_text

    log_fh = open(log_file_path, "a", encoding="utf-8", errors="ignore")
    try:
        for line in iter(training_process.stdout.readline, ""):
            line_str = line.replace('\r', '').strip()
            if not line_str: continue
            if any(skip_word in line_str for skip_word in LOG_BLACKLIST): continue

            log_fh.write(line_str + "\n")
            log_fh.flush()

            with training_state_lock:
                if "subprocess.CalledProcessError" in line_str and "returned non-zero exit status 15" in line_str:
                    training_log_lines.append("🛑 Interrupted by user.")
                    break

                if "steps:" in line_str and "/" in line_str:
                    match = _STEP_PATTERN.search(line_str)
                    if match:
                        current_step_info = match.group(0)
                        if training_log_lines and "steps:" in training_log_lines[-1] and current_step_info in training_log_lines[-1]:
                            training_log_lines[-1] = line_str
                        else:
                            training_log_lines.append(line_str)
                    else:
                        training_log_lines.append(line_str)
                else:
                    training_log_lines.append(line_str)

                if len(training_log_lines) > MAX_LOG_LINES: del training_log_lines[:-MAX_LOG_LINES]

                em = _ETA_RE.search(line_str)
                if em:
                    cur_s, tot_s, _, rate_val, rate_unit = em.groups()
                    cur_i, tot_i, rate_f = int(cur_s), int(tot_s), float(rate_val)
                    if cur_i > 0 and rate_f > 0:
                        remaining = tot_i - cur_i
                        secs = remaining / rate_f if rate_unit == "it/s" else remaining * rate_f
                        mins, sec = int(secs // 60), int(secs % 60)
                        training_eta_text = f"≈ {mins}m {sec:02d}s remaining  ({cur_i}/{tot_i} steps)"

        if training_process is not None:
            training_process.wait()

        with training_state_lock:
            training_log_lines.append("✅ Process finished or stopped.")

    except Exception as e:
        with training_state_lock:
            training_log_lines.append(f"❌ Error: {str(e)}")
    finally:
        log_fh.close()
        with training_state_lock:
            training_active = False
            training_eta_text = ""
        training_process = None


def find_resumable_state(project_out_dir, project_name):
    """sd-scripts writes '{name}-step{N:08d}-state' per save and '{name}-state' at
    train end (train_util.STEP_STATE_NAME / LAST_STATE_NAME). Pick the most recently
    modified one, since a higher step number in the final on-train-end state dir
    (no step suffix) should still win over an older mid-run step checkpoint."""
    if not project_out_dir.exists():
        return None
    candidates = list(project_out_dir.glob(f"{project_name}-step*-state")) + list(project_out_dir.glob(f"{project_name}-state"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def start_training(trigger_word, dataset_path, dit_p, qwen_p, vae_p, rank, lr, optimizer, t_steps, save_steps, sample_steps, pos, neg, w, h, s_steps_gen, s_cfg, s_seed, train_seed, batch_size, grad_acc, side_min=512, side_max=768, resume=True, skip_cache_check=False):
    global training_process, training_active, training_log_lines, training_eta_text, training_sample_dir, training_last_image_count

    model_errors = []
    if not dit_p or not os.path.isfile(dit_p): model_errors.append(f"❌ DiT file not found: {dit_p}")
    if not qwen_p or not os.path.isfile(qwen_p): model_errors.append(f"❌ Qwen3 file not found: {qwen_p}")
    if not vae_p or not os.path.isfile(vae_p): model_errors.append(f"❌ VAE file not found: {vae_p}")

    dataset_errors = []
    if not dataset_path or not os.path.exists(dataset_path):
        dataset_errors.append(f"❌ Dataset path not found: {dataset_path}")
    else:
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        dataset_path_obj = Path(dataset_path)
        image_files = [f for f in dataset_path_obj.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]

        if not image_files:
            dataset_errors.append(f"❌ No valid images found in the dataset path.")
        else:
            # Check for missing text files (captions)
            missing_captions = [img.name for img in image_files if not img.with_suffix('.txt').exists()]
            if missing_captions:
                dataset_errors.append(f"❌ Error: Found {len(missing_captions)} images without .txt captions. Please use the 'Auto-Caption Dataset' tool first.")

            # Check for oversized images
            oversized_images = []
            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        w_img, h_img = img.size
                        if w_img >= 2048 or h_img >= 2048:
                            oversized_images.append(img.name)
                except Exception:
                    pass

            if oversized_images:
                dataset_errors.append(f"❌ Error: Found {len(oversized_images)} images that are too large (>= 2048px). Please use the 'Smart Aspect Ratio Bucketing' tool to resize them before training.")

    if model_errors or dataset_errors:
        full_error = ""
        if model_errors:
            full_error += "\n".join(model_errors)
            full_error += "\n\n⚠️ ERROR: Please check and set the correct model paths in the section:\n'🔧 Paths to Models <- Set Once'\n\n"
        if dataset_errors:
            full_error += "\n".join(dataset_errors)

        return full_error.strip(), gr.update(), ""

    if not torch.cuda.is_available():
        cuda_error = (
            "❌ ERROR: NVIDIA GPU not detected or CUDA drivers are not installed!\n\n"
            "Technical details:\n"
            "- PyTorch cannot initialize CUDA.\n"
            "- Training on CPU is extremely slow and is not supported by this script.\n\n"
            "Please update your NVIDIA drivers and restart the app."
        )
        return cuda_error, gr.update(), ""

    if training_process is not None and training_process.poll() is None:
        return "⚠️ Training is already running!", gr.update(), ""

    project_name = re.sub(r'[^a-zA-Z0-9]', '_', trigger_word.strip()).strip('_') or "untitled"

    project_out_dir = OUTPUT_BASE / project_name
    sample_dir = project_out_dir / "sample"
    project_configs_dir = project_out_dir / "configs"

    for d in [project_out_dir, sample_dir, project_configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with training_state_lock:
        training_log_lines = [f"🚀 Preparing: {project_name}..."]
        training_eta_text = ""
        training_sample_dir = sample_dir
        training_last_image_count = 0

    # T1c: bucket warning at log start (non-blocking) — use the user's actual Min/Max Side
    bw = check_bucket_batch(dataset_path, batch_size, side_min, side_max)
    if bw:
        with training_state_lock:
            for bw_line in bw.split("\n"):
                training_log_lines.append(bw_line)

    with training_state_lock:
        training_log_lines.append("🔍 Analyzing dataset images...")
    base_res, max_bucket = analyze_dataset_resolution(dataset_path)
    with training_state_lock:
        training_log_lines.append(f"📐 Auto-Resolution Set: Base {base_res}px, Max Bucket {max_bucket}px")

    models = {"dit_path": dit_p, "qwen_path": qwen_p, "vae_path": vae_p}
    prompt_path = create_sample_prompts(project_name, trigger_word, pos, neg, w, h, s_steps_gen, s_cfg, s_seed, project_configs_dir)
    dataset_toml = create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, project_configs_dir, t_steps, batch_size, grad_acc)
    training_toml = create_training_toml(project_name, project_configs_dir, project_out_dir, rank, lr, optimizer, t_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc, skip_cache_check)

    cmd = [
        str(PORTABLE_PYTHON.resolve()), "-m", "accelerate.commands.launch", "--num_processes=1", "--mixed_precision=bf16", "--dynamo_backend=no",
        TRAIN_SCRIPT.resolve().as_posix(),
        "--config_file", Path(training_toml).resolve().as_posix(),
        "--dataset_config", Path(dataset_toml).resolve().as_posix()
    ]

    if resume:
        state_dir = find_resumable_state(project_out_dir, project_name)
        if state_dir is not None:
            cmd += ["--resume", state_dir.resolve().as_posix()]
            with training_state_lock:
                training_log_lines.append(f"🔁 Continuing from checkpoint: {state_dir.name}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRAIN_DIR.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore"
    env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    env["KMP_WARNINGS"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["ACCELERATE_USE_CPU"] = "False"

    log_file_path = project_out_dir / "train.log"
    try:
        log_file_path.write_text("", encoding="utf-8")
    except Exception:
        pass

    try:
        training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, cwd=str(TRAIN_DIR.resolve()), env=env, encoding="utf-8", errors="ignore")
        training_active = True
        threading.Thread(target=_training_worker, args=(log_file_path,), daemon=True).start()
    except Exception as e:
        with training_state_lock:
            training_log_lines.append(f"❌ Error: {str(e)}")
        training_active = False
        training_process = None

    with training_state_lock:
        return "\n".join(training_log_lines), gr.update(), training_eta_text


def poll_training_status(sort_mode="Latest"):
    """Bound to a gr.Timer and to ui.load, so the log/preview/ETA reflect the
    module-level training state regardless of which browser request is asking —
    this is what lets a still-running subprocess survive a page refresh."""
    global training_last_image_count
    with training_state_lock:
        log_text = "\n".join(training_log_lines)
        eta = training_eta_text
        sample_dir = training_sample_dir
        active = training_active
        last_count = training_last_image_count

    if not active and not log_text:
        return gr.update(), gr.update(), gr.update()

    gallery_update = gr.update()
    # "By Step" mode is refreshed manually (needs a checkpoint-matching pass); don't
    # fight the user's chosen view with the recency-sorted "Latest" auto-refresh.
    if sample_dir is not None and sort_mode == "Latest":
        imgs = get_latest_images(sample_dir)
        if len(imgs) != last_count:
            training_last_image_count = len(imgs)
            gallery_update = imgs

    return log_text, gallery_update, eta


def stop_training():
    global training_process
    if training_process is not None:
        try:
            parent = psutil.Process(training_process.pid)
            for child in parent.children(recursive=True):
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            gone, alive = psutil.wait_procs(parent.children(recursive=True), timeout=3)
            for survival in alive:
                try: survival.kill()
                except psutil.NoSuchProcess: pass
            parent.terminate()
            parent.wait(timeout=3)
            return "🛑 Stopping training... (Clearing VRAM)"
        except psutil.NoSuchProcess:
            return "ℹ️ Process already finished."
        except Exception as e:
            return f"⚠️ Error during stop: {str(e)}"
    return "ℹ️ Not running."

def open_output_folder(trigger_word):
    proj = re.sub(r'[^a-zA-Z0-9]', '_', trigger_word.strip()).strip('_')
    target_dir = OUTPUT_BASE / proj if proj else OUTPUT_BASE
    if not target_dir.exists(): target_dir = OUTPUT_BASE
    if platform.system() == "Windows": os.startfile(target_dir)
    else: subprocess.Popen(["xdg-open", str(target_dir)])
    return "📁 Folder opened."

# Heuristic starting points anchored to batch-4 figures from the Anima training post
_ADAMW_LR_BY_BATCH = {1: "0.00005", 2: "0.00006", 4: "0.00008", 8: "0.00012"}

def _adamw_lr_for_batch(batch_size):
    batch = max(1, int(batch_size))
    nearest = min(_ADAMW_LR_BY_BATCH.keys(), key=lambda k: abs(k - batch))
    return _ADAMW_LR_BY_BATCH[nearest]

def handle_optimizer_change(opt, batch_size):
    # auto-LR (T1a): Prodigy is schedule-free (fixed 1.0); AdamW LR comes from the batch table.
    # Only sets the default on optimizer/batch change — the field stays editable afterward.
    if opt == "Prodigy":
        return "1.0"
    return _adamw_lr_for_batch(batch_size)

def suggest_steps(dataset_path, batch_size, grad_acc, target_exp):
    n = count_dataset_images(dataset_path)
    if n == 0:
        return gr.update(), gr.update(), gr.update(), "No images found at dataset path."

    eff_batch = max(1, int(batch_size) * int(grad_acc))
    target = float(target_exp) if target_exp else 30.0

    def to_steps(exp):
        return max(1, round((exp * n / eff_batch) / 25) * 25)

    steps = to_steps(target)
    lo, hi = to_steps(target - 5), to_steps(target + 5)
    cadence = max(50, round((steps / 6) / 25) * 25)

    info = (f"{n} images | eff. batch {eff_batch} | target {target:.0f} exp/img\n"
            f"Suggested: {steps} steps (band {lo}–{hi} @ {target-5:.0f}–{target+5:.0f} exp/img)\n"
            f"Save/preview every {cadence} steps → ~{steps // cadence} checkpoints to A/B")

    return gr.update(value=steps), gr.update(value=cadence), gr.update(value=cadence), info

def apply_preset(name):
    presets = {p["name"]: p for p in load_presets()}
    p = presets.get(name)
    if not p:
        return tuple(gr.update() for _ in range(8))
    return (
        gr.update(value=p["optimizer"]),
        gr.update(value=p["learning_rate"]),
        gr.update(value=p["network_rank"]),
        gr.update(value=p["training_steps"]),
        gr.update(value=p["train_batch_size"]),
        gr.update(value=p["gradient_accumulation_steps"]),
        gr.update(value=p["save_steps"]),
        gr.update(value=p["sample_steps"]),
    )


# ==========================================
# T1b — Exposures / image gauge
# ==========================================
def compute_exp_gauge(dataset_path, steps, batch_size, grad_acc):
    if steps is None or batch_size is None or grad_acc is None:
        return ""  # a Number field cleared mid-edit passes None — don't raise a TypeError toast
    n = count_dataset_images(dataset_path)
    if n == 0:
        return ""
    exp = int(steps) * int(batch_size) * int(grad_acc) / n
    if exp <= 24:   band = "❄️ Cool"
    elif exp <= 35: band = "✅ Healthy"
    elif exp <= 50: band = "🔥 Warm"
    else:           band = "💀 Fry-risk"
    return f"{exp:.1f} exp/image — {band}  ({n} images · {int(steps)} steps · eff. batch {int(batch_size)*int(grad_acc)})"


# ==========================================
# T1c — Bucket-vs-batch silent failure check
# ==========================================
def check_bucket_batch(dataset_path, batch_size, side_min, side_max):
    n = count_dataset_images(dataset_path)
    if n == 0 or int(batch_size) <= 1:
        return ""
    path = Path(dataset_path)
    if not path.exists():
        return ""
    batch = int(batch_size)
    available_buckets = smart_cropper.get_valid_buckets(int(side_min), int(side_max))
    bucket_counts = {}
    for img_path in path.glob('*'):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_IMG_EXTS:
            continue
        try:
            with Image.open(img_path) as img:
                w, h = img.size
            bucket, _ = smart_cropper.get_best_bucket(w, h, available_buckets)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        except Exception:
            continue
    thin = [(b, c) for b, c in bucket_counts.items() if c < batch]
    if not thin:
        return ""
    return "\n".join(
        f"⚠️ Bucket {bw}×{bh}: {cnt} image{'s' if cnt != 1 else ''} < batch {batch} — lower batch to {cnt} or add images at this ratio"
        for (bw, bh), cnt in sorted(thin)
    )


# ==========================================
# T2 — Analyze & Configure (chains T1a/T1b/T1c + suggest_steps + resolution)
# ==========================================
def analyze_and_configure(dataset_path, optimizer, batch_size, grad_acc, target_exp):
    n = count_dataset_images(dataset_path)
    if n == 0:
        no = gr.update()
        return no, no, no, no, no, no, "No images found at dataset path.", ""

    base_res, max_bucket_res = analyze_dataset_resolution(dataset_path)

    eff_batch = max(1, int(batch_size) * int(grad_acc))
    target = float(target_exp) if target_exp else 30.0
    def to_steps(exp):
        return max(1, round((exp * n / eff_batch) / 25) * 25)
    steps = to_steps(target)
    cadence = max(50, round((steps / 6) / 25) * 25)
    lo, hi = to_steps(target - 5), to_steps(target + 5)

    lr = "1.0" if optimizer == "Prodigy" else _adamw_lr_for_batch(batch_size)

    # check against the resolution we're about to write into the side fields, not the stale ones
    bucket_warn = check_bucket_batch(dataset_path, batch_size, base_res, max_bucket_res)

    info = (f"{n} images | eff. batch {eff_batch} | target {target:.0f} exp/img\n"
            f"Steps: {steps} (band {lo}–{hi} @ 25–35 exp/img) | cadence: every {cadence} → ~{steps // cadence} checkpoints\n"
            f"LR: {lr} | Resolution: base {base_res}px · max bucket {max_bucket_res}px")
    if bucket_warn:
        info += f"\n{bucket_warn}"

    exp = steps * int(batch_size) * int(grad_acc) / n
    if exp <= 24:   band = "❄️ Cool"
    elif exp <= 35: band = "✅ Healthy"
    elif exp <= 50: band = "🔥 Warm"
    else:           band = "💀 Fry-risk"
    gauge = f"{exp:.1f} exp/image — {band}  ({n} images · {steps} steps · eff. batch {eff_batch})"

    return (gr.update(value=steps), gr.update(value=cadence), gr.update(value=cadence),
            gr.update(value=lr), gr.update(value=base_res), gr.update(value=max_bucket_res),
            info, gauge)


# ==========================================
# T3 — Post-train checkpoint A/B gallery
# ==========================================
def get_ab_gallery(trigger_word):
    proj = re.sub(r'[^a-zA-Z0-9]', '_', trigger_word.strip()).strip('_') or "untitled"
    sample_dir = OUTPUT_BASE / proj / "sample"
    if not sample_dir.exists():
        return []
    images = []
    for pattern in ('*.png', '*.jpg', '*.webp'):
        images.extend(sample_dir.glob(pattern))
    def parse_step(p):
        # sample files: {project}_{step:06d}_{i:02d}_{YYYYMMDDHHMMSS}[_{seed}].png
        # anchor on the fixed index_timestamp tail so a digit-bearing project name can't shadow the step
        m = re.search(r'_(\d{6,})_\d{2}_\d{8,}', p.stem)
        if m:
            return int(m.group(1))
        m = re.search(r'(\d{4,8})', p.stem)
        return int(m.group(1)) if m else 0
    images.sort(key=lambda p: (parse_step(p), p.name))
    out_dir = OUTPUT_BASE / proj
    result = []
    for img in images:
        step = parse_step(img)
        ckpts = sorted(out_dir.glob(f"*{step:06d}*.safetensors")) or sorted(out_dir.glob(f"*{step}*.safetensors"))
        ckpt_name = ckpts[0].name if ckpts else "—"
        result.append((str(img), f"Step {step} — {ckpt_name}"))
    return result


def refresh_gallery_for_mode(sort_mode, trigger_word):
    """Single gallery, two renderers over the same on-disk sample folder:
    'Latest' = recency-sorted, unlabeled (for eyeballing progress mid-run);
    'By Step' = step-sorted, labeled with the matching checkpoint (for comparing
    a run's history after the fact). Replaces the old separate A/B gallery widget."""
    if sort_mode == "By Step":
        return get_ab_gallery(trigger_word)
    if training_sample_dir is not None:
        return get_latest_images(training_sample_dir)
    return []


# ==========================================
# Misc — Caption editor helpers
# ==========================================
def _list_caption_images(dataset_path):
    path = Path(dataset_path)
    if not path.exists():
        return []
    return sorted(f for f in path.glob('*') if f.is_file() and f.suffix.lower() in VALID_IMG_EXTS)

def load_caption_for_edit(dataset_path, idx):
    imgs = _list_caption_images(dataset_path)
    if not imgs:
        return None, "", "No images found"
    idx = max(0, min(int(idx) - 1, len(imgs) - 1))
    img_path = imgs[idx]
    txt_path = img_path.with_suffix('.txt')
    caption = txt_path.read_text(encoding='utf-8').strip() if txt_path.exists() else ""
    return str(img_path), caption, f"{idx + 1} / {len(imgs)} — {img_path.name}"

def save_caption_file(dataset_path, idx, caption_text):
    imgs = _list_caption_images(dataset_path)
    if not imgs:
        return "No images found"
    idx = max(0, min(int(idx) - 1, len(imgs) - 1))
    imgs[idx].with_suffix('.txt').write_text(caption_text.strip(), encoding='utf-8')
    return f"✅ Saved: {imgs[idx].name}.txt"

def get_output_path(trigger_word):
    proj = re.sub(r'[^a-zA-Z0-9]', '_', trigger_word.strip()).strip('_') or "untitled"
    return str(OUTPUT_BASE / proj)


# ==========================================
# UI BUILDER
# ==========================================
cs = load_settings()

with gr.Blocks(title="Anima TrainFlow: Easy LoRA Trainer for Anima 2B") as ui:
    gr.Markdown(
        "# Anima TrainFlow\n"
        '<div class="attribution"><span class="author-text">Created by ThetaCursed</span></div>',
        elem_id="main-header"
    )
    
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    trigger_word = gr.Textbox(label="Trigger Word / Project Name", value=cs.get("trigger_word", ""), placeholder="e.g., unique_style")
                    dataset_path = gr.Textbox(label="Dataset Path (Images + .txt)", value=cs.get("dataset_path", ""), placeholder="C:/Images/MyDataset")
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze & Configure", variant="secondary")
                with gr.Accordion("🔧 Paths to Models <- Set Once", open=False):
                    dit_input = gr.Textbox(label="DiT", value=cs.get("dit_path", ""))
                    qwen_input = gr.Textbox(label="Qwen3", value=cs.get("qwen_path", ""))
                    vae_input = gr.Textbox(label="VAE", value=cs.get("vae_path", ""))
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary")
                    stop_btn = gr.Button("🛑 Stop", variant="stop")
                    folder_btn = gr.Button("📁 Checkpoint Folder", variant="secondary")
                    copy_path_btn = gr.Button("📋 Copy Output Path", variant="secondary")
                resume_checkbox = gr.Checkbox(label="🔁 Continue from last checkpoint if one exists for this project", value=True)
                skip_cache_check_checkbox = gr.Checkbox(
                    label="⚡ Skip cache validity check on start (faster, but only safe if the dataset images/captions haven't changed since the last cache)",
                    value=False,
                )
                output_path_display = gr.Textbox(label="Output Path", interactive=False, lines=1, elem_id="output-path-box")
            with gr.Column(scale=1):
                with gr.Row():
                    preset_dd = gr.Dropdown(label="Preset", choices=[p["name"] for p in load_presets()], value=None)
                    apply_preset_btn = gr.Button("Apply Preset", variant="secondary")
                with gr.Row():
                    rank_input = gr.Number(label="Network Rank", value=cs.get("network_rank", 16), precision=0)
                    lr_input = gr.Textbox(label="Learning Rate", value=cs.get("learning_rate", "1.0"))
                    optimizer_input = gr.Dropdown(label="Optimizer", choices=["Prodigy", "AdamW8bit", "AdamW"], value=cs.get("optimizer", "Prodigy"))
                    train_seed_val = gr.Number(value=cs.get("train_seed", 42), visible=False)
                    batch_size_input = gr.Number(label="Batch Size", value=cs.get("train_batch_size", 1), precision=0)
                with gr.Row():
                    steps_input = gr.Number(label="Training Steps", value=cs.get("training_steps", 2400), precision=0)
                    save_steps_input = gr.Number(label="Save Every n Steps", value=cs.get("save_steps", 300), precision=0)
                    sample_steps_input = gr.Number(label="Preview Every n Steps", value=cs.get("sample_steps", 300), precision=0)
                    grad_acc_input = gr.Number(label="Gradient Accumulation", value=cs.get("gradient_accumulation_steps", 1), precision=0)
                with gr.Row():
                    target_exp_input = gr.Number(label="Target exp/image", value=30, precision=0, min_width=120)
                    suggest_btn = gr.Button("Suggest Steps", variant="secondary")
                suggest_info = gr.Textbox(label="Suggestion", interactive=False, lines=3)
                exp_gauge = gr.Textbox(label="Exposures / image", interactive=False, lines=1)

    with gr.Row():
        with gr.Column(scale=1):
            eta_output = gr.Textbox(label="ETA", interactive=False, lines=1)
            output_log = gr.Textbox(label="Logs", lines=LOG_BOX__MAX_LINES, max_lines=LOG_BOX__MAX_LINES, interactive=False, autoscroll=True, elem_id="log-container")
            
            with gr.Group():
                gr.Markdown("### Smart Aspect Ratio Bucketing")
                with gr.Row():
                    side_min_input = gr.Number(label="Min Side (Base)", value=cs.get("side_min", 512), precision=0)
                    side_max_input = gr.Number(label="Max Side (Limit)", value=cs.get("side_max", 768), precision=0)
                with gr.Row():
                    crop_btn = gr.Button("✂️ Start Bucketing", variant="secondary")
                    open_ds_btn = gr.Button("📂 Open Dataset", variant="secondary")
                    
            # --- AUTO CAPTION ---
            with gr.Group():
                gr.Markdown("### Auto-Caption Dataset")
                with gr.Row():
                    tagger_gen_thresh = gr.Slider(0.0, 1.0, value=cs.get("tagger_gen_thresh", 0.35), step=0.01, label="General Tags Threshold")
                    tagger_char_thresh = gr.Slider(0.0, 1.0, value=cs.get("tagger_char_thresh", 0.85), step=0.01, label="Character Tags Threshold")
                with gr.Row():
                    tagger_btn = gr.Button("Create .txt captions", variant="secondary")
                    tagger_overwrite = gr.Checkbox(label="Overwrite existing .txt", value=cs.get("tagger_overwrite", False))

            with gr.Accordion("✏️ Caption Editor", open=False):
                with gr.Row():
                    caption_idx = gr.Number(label="Image #", value=1, precision=0, min_width=80)
                    caption_prev_btn = gr.Button("◀", min_width=50)
                    caption_next_btn = gr.Button("▶", min_width=50)
                    caption_load_btn = gr.Button("Load", variant="secondary")
                caption_image = gr.Image(label="Image", height=200, interactive=False)
                caption_label = gr.Textbox(label="", interactive=False, lines=1)
                caption_text = gr.Textbox(label="Caption", lines=4, interactive=True)
                caption_save_btn = gr.Button("💾 Save Caption", variant="primary")
                caption_status = gr.Textbox(label="", interactive=False, lines=1)


        with gr.Column(scale=1):
            with gr.Row():
                gallery_sort = gr.Radio(choices=["Latest", "By Step"], value="Latest", label="Sort", scale=2, min_width=160)
                refresh_ab_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
            preview_gallery = gr.Gallery(label="Sample Previews", columns=2, rows=2, height=GALLERY_HEIGHT, object_fit="contain")
            with gr.Group():
                pos_prompt = gr.Textbox(label="Prompt (Trigger word added automatically)", lines=2, value=cs.get("pos_prompt", ""))
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=1, value=cs.get("neg_prompt", ""))
                with gr.Row():
                    width_input = gr.Number(label="Width", value=cs.get("width", 1024), precision=0, min_width=80)
                    height_input = gr.Number(label="Height", value=cs.get("height", 1024), precision=0, min_width=80)
                    sample_steps_gen_input = gr.Number(label="Steps", value=cs.get("sample_steps_gen", 30), precision=0, min_width=80)
                    sample_cfg_input = gr.Number(label="CFG", value=cs.get("sample_cfg", 4.0), min_width=80)
                    sample_seed_input = gr.Number(label="Seed", value=cs.get("sample_seed", 42), precision=0, min_width=80)

    training_inputs = [
        trigger_word, dataset_path, dit_input, qwen_input, vae_input,
        rank_input, lr_input, optimizer_input, 
        steps_input, save_steps_input, sample_steps_input,
        pos_prompt, neg_prompt, width_input, height_input,
        sample_steps_gen_input, sample_cfg_input, sample_seed_input, train_seed_val, batch_size_input, grad_acc_input
    ]

    all_settings_list = training_inputs + [
        side_min_input, side_max_input, tagger_gen_thresh, tagger_char_thresh, tagger_overwrite
    ]

    def load_state_on_refresh():
        current_settings = load_settings()
        return [current_settings.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS.keys()]

    ui.load(fn=load_state_on_refresh, inputs=None, outputs=all_settings_list)

    # Polls module-level training state (survives page refresh — the subprocess
    # itself is never tied to a browser session) so log/preview/ETA rehydrate
    # instantly on load and keep updating even if the original click request died.
    training_timer = gr.Timer(1.0, active=True)
    training_timer.tick(fn=poll_training_status, inputs=[gallery_sort], outputs=[output_log, preview_gallery, eta_output])
    ui.load(fn=poll_training_status, inputs=[gallery_sort], outputs=[output_log, preview_gallery, eta_output])

    gallery_sort.change(fn=refresh_gallery_for_mode, inputs=[gallery_sort, trigger_word], outputs=[preview_gallery])

    output_log.change(None, None, None, js=JS_SCROLL)

    apply_preset_btn.click(
        fn=apply_preset,
        inputs=[preset_dd],
        outputs=[optimizer_input, lr_input, rank_input, steps_input,
                 batch_size_input, grad_acc_input, save_steps_input, sample_steps_input],
    )
    optimizer_input.change(fn=handle_optimizer_change, inputs=[optimizer_input, batch_size_input], outputs=[lr_input])
    batch_size_input.change(fn=handle_optimizer_change, inputs=[optimizer_input, batch_size_input], outputs=[lr_input])

    suggest_btn.click(
        fn=suggest_steps,
        inputs=[dataset_path, batch_size_input, grad_acc_input, target_exp_input],
        outputs=[steps_input, save_steps_input, sample_steps_input, suggest_info],
    )

    # T1b: live exposures/image gauge
    _gauge_inputs = [dataset_path, steps_input, batch_size_input, grad_acc_input]
    for _comp in _gauge_inputs:
        _comp.change(fn=compute_exp_gauge, inputs=_gauge_inputs, outputs=[exp_gauge])

    # T2: Analyze & Configure
    analyze_btn.click(
        fn=analyze_and_configure,
        inputs=[dataset_path, optimizer_input, batch_size_input, grad_acc_input, target_exp_input],
        outputs=[steps_input, save_steps_input, sample_steps_input, lr_input, side_min_input, side_max_input, suggest_info, exp_gauge],
    )

    for comp in all_settings_list:
        comp.change(fn=auto_save_state, inputs=all_settings_list)

    crop_btn.click(
        fn=run_smart_crop_ui,
        inputs=[dataset_path, side_min_input, side_max_input, output_log],
        outputs=[output_log]
    )
    open_ds_btn.click(
        fn=open_dataset_folder_ui,
        inputs=[dataset_path, output_log],
        outputs=[output_log]
    )
    tagger_btn.click(
        fn=run_auto_tagging,
        inputs=[dataset_path, tagger_gen_thresh, tagger_char_thresh, tagger_overwrite, output_log],
        outputs=[output_log]
    )

    # T3: sample preview gallery — manual refresh respects the current sort mode
    refresh_ab_btn.click(fn=refresh_gallery_for_mode, inputs=[gallery_sort, trigger_word], outputs=[preview_gallery])

    # T4: ETA wired as third output of start_training
    start_btn.click(fn=start_training, inputs=training_inputs + [side_min_input, side_max_input, resume_checkbox, skip_cache_check_checkbox], outputs=[output_log, preview_gallery, eta_output])
    stop_btn.click(fn=stop_training, outputs=output_log)
    folder_btn.click(fn=open_output_folder, inputs=[trigger_word], outputs=output_log)

    # Misc: output path display
    copy_path_btn.click(fn=get_output_path, inputs=[trigger_word], outputs=[output_path_display])
    copy_path_btn.click(None, None, None, js="() => { const el = document.querySelector('#output-path-box textarea'); if(el){ navigator.clipboard.writeText(el.value); } }")

    # Misc: caption editor
    def _caption_nav(dataset_path, idx, delta):
        imgs = _list_caption_images(dataset_path)
        new_idx = max(1, min(int(idx) + delta, len(imgs))) if imgs else 1
        return new_idx
    caption_load_btn.click(fn=load_caption_for_edit, inputs=[dataset_path, caption_idx], outputs=[caption_image, caption_text, caption_label])
    caption_prev_btn.click(fn=lambda d, i: _caption_nav(d, i, -1), inputs=[dataset_path, caption_idx], outputs=[caption_idx]).then(fn=load_caption_for_edit, inputs=[dataset_path, caption_idx], outputs=[caption_image, caption_text, caption_label])
    caption_next_btn.click(fn=lambda d, i: _caption_nav(d, i, +1), inputs=[dataset_path, caption_idx], outputs=[caption_idx]).then(fn=load_caption_for_edit, inputs=[dataset_path, caption_idx], outputs=[caption_image, caption_text, caption_label])
    caption_save_btn.click(fn=save_caption_file, inputs=[dataset_path, caption_idx, caption_text], outputs=[caption_status])

if __name__ == "__main__":
     ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, theme=gr.themes.Soft(), css=CSS)