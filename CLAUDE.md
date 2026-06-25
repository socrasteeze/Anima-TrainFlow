# Anima TrainFlow — Claude Context

## What This Project Is

A single-page Gradio GUI for training LoRA adapters on the **Anima 2B** diffusion model. Targets Windows users with 6GB+ VRAM. The full workflow lives in one screen: smart crop → auto-tag → configure → train → review checkpoints.

**Portable edition** ships with an embedded Python runtime (`python_embeded/`). Users run `start_trainer.bat` and never touch a terminal.

---

## Project Layout

```
app.py                          # Entire application (~1100 lines, single file)
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
| 1–150 | Imports, CSS/JS, constants, path setup |
| 152–203 | Settings load/save; hidden training params (BF16, scheduler, etc.) |
| 208–350 | Dataset utilities and TOML config generators |
| 352–450 | Gallery refresh, bucket summary, log filtering |
| 453–645 | Smart crop — multi-threaded U2Net subject-aware cropping |
| 648–725 | Auto-tag — multi-threaded WD14 captioning |
| 745–970 | Training start/stop, validation, subprocess launch |
| 971–1112 | Gradio UI builder |

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
- `lr_scheduler`: `rex`
- `cache_latents`: `true`
- `cache_text_encoder_outputs`: `true`

Exposed parameters: trigger word, dataset path, rank, network alpha, LR, steps, batch size, gradient accumulation, save/sample every N steps, optimizer.

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

Specification docs for these features are in `atf_PRESETS_BRIEF.md` and `atf_SUGGEST_STEPS_BRIEF.md`.

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
