# Anima TrainFlow — TODO / Build Spec

Design law for this app: **remove a decision, don't add an option.** Every enhancement automates expert knowledge and surfaces it as an editable default. No raw knobs the user must already understand. The suggest-steps helper is the template — follow it.

> **Status:** The full build backlog (W → T4 + Misc) has shipped. Only the Icebox remains, and both items are deliberately deferred. See "Completed" for where each feature lives in `app.py`.

---

## Completed

### Core app
- [x] Zero-tab single-page Gradio UI
- [x] Smart crop (U2Net subject-aware, multi-threaded) — `SmartCropper`, `run_smart_crop_ui`
- [x] Auto-captioning (WD14 EVA02 v3, multi-threaded) — `WDTagger`, `run_auto_tagging`
- [x] Dataset validation (missing captions, oversized images, model paths)
- [x] Persistent settings (`training/settings.json`, auto-save on change) — `auto_save_state`
- [x] Live training previews (auto-updating gallery every N steps)
- [x] Preset system — `training/presets.json`, applies optimizer/LR/rank/batch/steps/cadence — `apply_preset`
- [x] Step suggestion helper — steps from image count × target exp/image, + cadence for ~6 checkpoints — `suggest_steps`
- [x] Tailscale access — `server_name="0.0.0.0"` in `ui.launch()`
- [x] Checkpoint → ComfyUI junction (`mklink /J` from `training/output` into loras dir)

### Build spec (W → T4)
- [x] **W. LR warmup** — `create_training_toml` emits `lr_warmup_steps = 0.1` (float ratio) for warmup-capable schedulers; skipped for Prodigy (schedule-free, `safeguard_warmup`). No UI, no preset key.
- [x] **T1a. Auto-LR from optimizer × batch** — `_ADAMW_LR_BY_BATCH` lookup + `_adamw_lr_for_batch` (nearest batch); wired through `handle_optimizer_change` on optimizer/batch change. Prodigy → `1.0`. Field stays editable + persisted; only the default is computed.
- [x] **T1b. Exposures/image gauge** — `compute_exp_gauge` live read-only readout (`steps*batch*grad_acc / num_images`) with cool/healthy/warm/fry bands. Pure display.
- [x] **T1c. Bucket-vs-batch warning** — `check_bucket_batch` flags any bucket with fewer images than the batch size. Warning only, never blocks training.
- [x] **T2. "Analyze & Configure" button** — `analyze_and_configure` chains count/resolution → suggest_steps → auto-LR → bucket check → gauge in one click.
- [x] **T3. Post-train checkpoint A/B gallery** — `get_ab_gallery` scans `training/output/<project>/sample/`, groups by step, and pairs each tile with its matching `.safetensors`.
- [x] **T4. Training ETA** — `start_training` parses the streamed s/it (`_eta_re`) and shows "≈ Xm remaining".

### Misc
- [x] `python_embeded/` confirmed in `.gitignore`
- [x] "Copy output path" button next to the checkpoint folder opener — `copy_path_btn`
- [x] Per-image caption editing — `load_caption_for_edit` / `save_caption_file`

---

## Verified backend capability (no sd-scripts edits needed)
Inventoried bundled `library/optimizer.py` (Anima fork shares the upstream kohya tree):
- Warmup: `lr_warmup_steps` honored; **accepts a float ratio** — pass `0.1` for 10%.
- Schedulers present: constant, constant_with_warmup, cosine, cosine_with_restarts, linear, polynomial, inverse_sqrt.
- `network_alpha` is an independent network arg.
All shipped items above are app-side only.

---

## DROPPED
- ~~Network alpha override (16/8)~~ — rejected. A second knob for a second-order gain already reachable via LR/exposure. Keep `network_alpha = network_rank` hardcoded. Rank dropdown (16/32) stays the only capacity lever.

---

## Icebox — deferred, real cost / low payoff

### CAME optimizer
- Confirmed ABSENT from bundled sd-scripts (no `came` in tree).
- Needs a new dependency (`pytorch-optimizer` etc.) pip-installed into `python_embeded` — the embedded-env cascade risk that has bitten this setup.
- Marginal payoff: Prodigy + AdamW8bit cover style work; source post rated CAME a sidegrade (overfits on noise as often as it helps).
- If revisited: route via kohya's generic `--optimizer_type module.ClassName` importlib path. CAME LR band 2e-5–5e-5 (~half AdamW), default `0.00003`. Never inherit AdamW LR — instant fry.

### Rex scheduler
- Not built into the fork. Author's own verdict: cosine is equivalent. Cosine + warmup (W) covers the intent. Skip.

---

## Standing constraints (apply to every future edit)
- Never reorder `DEFAULT_SETTINGS` / `all_settings_list`; append new persisted keys to the end of BOTH at the same position.
- No repeats field — steps drive repeats in `create_dataset_toml`.
- Non-persisted helper controls (gauges, target field) stay OUT of `all_settings_list`.
- Don't touch the embedded Python env for pure app-side work.
