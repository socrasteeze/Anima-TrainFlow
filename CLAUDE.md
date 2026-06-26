# Anima TrainFlow — Claude Context

## What This Project Is

A single-page Gradio GUI for training LoRA adapters on the **Anima 2B** diffusion model. Targets Windows users with 6GB+ VRAM. The full workflow lives in one screen: smart crop → auto-tag → configure → train → review checkpoints.

**Portable edition** ships with an embedded Python runtime (`python_embeded/`). Users run `start_trainer.bat` and never touch a terminal.

---

## Project Layout

```
app.py                          # Entire application (~1340 lines, single file)
start_trainer.bat               # Launcher — calls python_embeded/python.exe app.py
Install_Requirements.bat        # One-time dependency installer
training/
  settings.json                 # Persisted UI state (auto-saved on change)
  presets.json                  # Named training presets (optimizer/LR/rank/batch)
  sd-scripts/                   # Modified sd-scripts fork for Anima 2B
    anima_train_network.py      # Core LoRA training backend
  output/                       # Checkpoints and samples land here
models/
  anima/                        # DiT, Qwen3 text encoder, VAE
  wd-eva02-large-tagger-v3/     # WD14 tagger for auto-captioning
  u2net/                        # U2Net ONNX model for smart cropping
python_embeded/                 # Embedded portable Python environment
```

---

## How to Run

```bat
start_trainer.bat
```

Opens Gradio on `http://localhost:7860`. For manual installs, run `Install_Requirements.bat` first.

---

## App Architecture (`app.py`)

The entire application is one file. Key sections by line range:

| Lines | Section |
|-------|---------|
| 1–122 | Imports, CSS/JS, constants, path setup |
| 124–203 | Settings load/save; hidden training params (BF16, scheduler, etc.) |
| 206–353 | Dataset utilities and TOML config generators |
| 355–453 | Gallery refresh, bucket summary, log filtering |
| 366–558 | Smart crop — `SmartCropper` (U2Net) class + multi-threaded crop pipeline |
| 564–727 | Auto-tag — `WDTagger` (WD14) class + multi-threaded captioning |
| 730–963 | Training start/stop, validation, subprocess launch; suggestion/gauge helpers |
| 965–1145 | Analysis (exposures gauge, suggest steps, analyze & configure), caption editor helpers |
| 1153–1341 | Gradio UI builder |

Training is launched as a subprocess via `accelerate launch anima_train_network.py`. Logs are streamed back and filtered through a Gradio Textbox.

---

## Key Design Decisions

- **Single file** — `app.py` contains everything. Do not split into modules without a strong reason.
- **No new tabs** — The zero-tab layout is intentional. All controls stay on one screen.
- **Embedded Python** — `python_embeded/` is the runtime for the portable edition. Do not add dependencies that require native compilation unless they ship prebuilt wheels.
- **Settings auto-save** — Every relevant UI component calls `save_settings()` on change. New UI fields should follow this pattern.
- **Non-persisted ephemeral fields** — Some fields (e.g., the "target exposures" input for step suggestion) are intentionally not saved to `settings.json`. Keep this behavior for fields that are recalculated each session.
- **Presets live in `training/presets.json`** — Adding or editing presets is done there, not in code.

---

## Training Parameters

Hidden defaults (not exposed in UI) are set in the `HIDDEN_SETTINGS` dict near the top of `app.py`:

- `mixed_precision`: `bf16`
- `gradient_checkpointing`: `true`
- `lr_scheduler`: `cosine`
- `cache_latents`: `true`
- `cache_text_encoder_outputs`: `true`

Exposed parameters: trigger word, dataset path, rank, LR, steps, batch size, gradient accumulation, save/sample every N steps, optimizer.

Network alpha is **not** exposed — it is auto-derived as `network_alpha = network_rank` in `create_training_toml()`.

---

## Models

| Model | Purpose | Location |
|-------|---------|----------|
| Anima 2B DiT | Main diffusion model | `models/anima/` |
| Qwen3 text encoder | Token encoding | `models/anima/` |
| VAE | Image encoding/decoding | `models/anima/` |
| WD EVA02 Large Tagger v3 | Auto-captioning | `models/wd-eva02-large-tagger-v3/` |
| U2Net | Subject detection for smart crop | `models/u2net/u2net.onnx` |

---

## Recent Features

- **Preset system** — Dropdown loads named configs from `training/presets.json`, overwriting optimizer, LR, rank, batch, steps, and cadence fields.
- **Step suggestion helper** — Calculates steps from image count × target exposures per image; suggests save/preview cadence for ~6 checkpoints.
- **Analyze & Configure** — One click reads the dataset, suggests steps/cadence/LR, detects resolution, and warns on bucket-vs-batch mismatches.
- **Exposures gauge** — Live overfit-risk readout (`steps × batch × grad_accum / num_images`) with health bands.
- **A/B checkpoint gallery** — Scans output samples, groups by step, and pairs each with its `.safetensors` checkpoint for review.

---

## Git Notes

- This is a fork of `ThetaCursed/Anima-TrainFlow`.
- Remote: `socrasteeze/Anima-TrainFlow`.
- Do not add `Co-Authored-By` trailers to commits.

### What to commit

Only commit enhancements and deliberate changes — source files that you authored or modified:

- `app.py`
- `training/presets.json`, `training/settings.json`
- `training/sd-scripts/` source files
- Top-level docs and config (`README.md`, `CLAUDE.md`, `TODO.md`, `*.bat`, `*.md`)

**Do not commit anything under `python_embeded/`** — that entire directory is the portable Python runtime populated by `pip install` and `Install_Requirements.bat`. Its churn is meaningless noise in git history. If it isn't already in `.gitignore`, add it.
