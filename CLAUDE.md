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
| 1–123 | Imports, CSS/JS, constants, path setup |
| 124–203 | Settings load/save, presets loader, `HIDDEN_SETTINGS` |
| 206–352 | Dataset utilities and TOML config generators |
| 355–558 | Preview gallery + smart crop — multi-threaded U2Net subject-aware cropping |
| 561–742 | Auto-tag — multi-threaded WD14 captioning |
| 744–950 | Training start/stop, validation, subprocess launch, log stream + ETA |
| 952–1145 | Config helpers: auto-LR table, suggest-steps, presets, exp gauge, bucket check, Analyze & Configure, A/B gallery, caption editor |
| 1148–1342 | Gradio UI builder and event wiring |

Two index-coupled lists in the UI builder are load-bearing: `training_inputs` (feeds `start_training` positionally) is a prefix of `all_settings_list`, which must stay index-aligned with `DEFAULT_SETTINGS.keys()` (the `auto_save_state` zip-mapping). Append new persisted keys at the end of both; never reorder. Extra event inputs go in at the `.click` site as `training_inputs + [extra, ...]`, not into the lists.

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
- `cache_latents`: `true`
- `cache_text_encoder_outputs`: `true`

The LR scheduler is NOT taken from `HIDDEN_SETTINGS` (its `lr_scheduler` key is unused) — it's chosen per optimizer inside `create_training_toml`: `constant` for Prodigy (schedule-free, `safeguard_warmup=True`), `cosine` with `lr_warmup_steps = 0.1` (10% float ratio) for AdamW/AdamW8bit. `network_alpha` is hardcoded equal to rank by design (see TODO.md "DROPPED").

Exposed parameters: trigger word, dataset path, rank, LR, steps, batch size, gradient accumulation, save/sample every N steps, optimizer. LR and steps are auto-populated (auto-LR from optimizer × batch; suggest-steps / Analyze & Configure) but remain editable.

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

All implemented and verified (2026-07-02 review); see TODO.md for per-feature line references and open fix-ups:

- **Preset system** — Dropdown loads named configs from `training/presets.json`, overwriting optimizer, LR, rank, batch, steps, and cadence fields.
- **Step suggestion helper** — Steps from image count × target exposures per image; suggests save/preview cadence for ~6 checkpoints.
- **Auto-LR** — LR populates from optimizer × batch size (`_ADAMW_LR_BY_BATCH`); Prodigy pins `1.0`.
- **Baked-in warmup** — 10% LR warmup on the cosine branch, invisible to the user.
- **Exposures/image gauge** — live readout of the governing overfit metric with cool/healthy/warm/fry bands.
- **Bucket-vs-batch warning** — flags buckets thinner than the batch size (non-blocking).
- **Analyze & Configure** — one click chains resolution analysis, suggest-steps, auto-LR, bucket check, and the gauge.
- **Checkpoint A/B gallery** — post-train gallery of sample images grouped by step, labeled with the matching `.safetensors`.
- **Training ETA** — parsed from tqdm rate lines in the streamed log.
- **Caption editor** — per-image `.txt` viewing/editing in an accordion.

Specification docs (historical, both fully implemented): `atf_PRESETS_BRIEF.md` and `atf_SUGGEST_STEPS_BRIEF.md`.

---

## Git Notes

- This is a fork of `ThetaCursed/Anima-TrainFlow`.
- Remote: `socrasteeze/Anima-TrainFlow`.
- Do not add `Co-Authored-By` trailers to commits.

### What to commit

Only commit enhancements and deliberate changes — source files that you authored or modified:

- `app.py`
- `training/presets.json`
- `training/sd-scripts/` source files
- Top-level docs and config (`README.md`, `CLAUDE.md`, `TODO.md`, `*.bat`, `*.md`)

**Do not commit `training/settings.json`** — it is auto-saved per-user state containing machine-specific absolute paths. It is in `.gitignore`.

**Do not commit anything under `python_embeded/`** — that entire directory is the portable Python runtime populated by `pip install` and `Install_Requirements.bat`. It is in `.gitignore`, but ~1000 files remain tracked from an old commit (hence the noisy `git status`); untracking them (`git rm -r --cached python_embeded/`) is an open task in TODO.md.
