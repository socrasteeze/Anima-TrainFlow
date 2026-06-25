# Anima TrainFlow — TODO

## Completed

- [x] Zero-tab single-page Gradio UI
- [x] Smart crop (U2Net subject-aware, multi-threaded)
- [x] Auto-captioning (WD14 EVA02 v3, multi-threaded)
- [x] Dataset validation (missing captions, oversized images, model paths)
- [x] Persistent settings (`training/settings.json`, auto-save on change)
- [x] Live training previews (auto-updating gallery every N steps)
- [x] Preset system — dropdown loads named configs from `training/presets.json`, applies optimizer/LR/rank/batch/steps/cadence (Phase 1)
- [x] Step suggestion helper — calculates steps from image count × target exposures/image, suggests save/preview cadence for ~6 checkpoints (Phase 1)

---

## Pending — Phase 2 Backend Uplift

These require verifying support in the bundled `training/sd-scripts` fork before exposing in the UI. Grep the fork first — do not add a dead UI control for an unsupported flag.

### 2a. Network alpha override

- [ ] Decouple `network_alpha` from `network_rank` (currently hardcoded as equal in `create_training_toml`)
- [ ] Add `network_alpha` field to UI (Number input, default 16)
- [ ] Thread it through `start_training` → `create_training_toml`
- [ ] Add `network_alpha` to `DEFAULT_SETTINGS` **and** `all_settings_list` at the same relative position (append to end of both to avoid corrupting the zip-mapping)
- [ ] Add `network_alpha` key to `presets.json` schema

### 2b. LR warmup (~10%)

- [ ] Grep `training/sd-scripts` for `lr_warmup_steps` to confirm the flag exists and its expected type (int vs ratio)
- [ ] In `create_training_toml`, compute `warmup = int(max_steps * 0.10)` and inject `lr_warmup_steps` when scheduler supports it (cosine / Rex — NOT constant or Prodigy)

### 2c. Rex scheduler

- [ ] Grep `training/sd-scripts` for `rex` as a registered `lr_scheduler` choice
- [ ] If supported: add scheduler selection to preset schema and `create_training_toml` (replace the hardcoded `constant`/`cosine` branch)
- [ ] If not supported in fork: fall back to cosine — do not promise Rex in the UI

### 2d. CAME optimizer

- [ ] Grep `training/sd-scripts` for `came` / `CAME` — it is NOT in the current optimizer list
- [ ] If absent: it requires `pytorch-optimizer` wired into the optimizer factory — confirm before committing to this
- [ ] If supported: add `"CAME"` to `optimizer_input` choices and `handle_optimizer_change`
- [ ] CAME LR band: 2e-5–5e-5 (≈ half of AdamW). Default preset LR: `0.00003`
- [ ] Add a CAME preset to `presets.json` — do NOT inherit AdamW LR values

---

## Other Ideas

- [ ] Check if `python_embeded/` is in `.gitignore`; add it if not
- [ ] Expose the `rex` scheduler toggle in UI (conditional on 2c confirming support)
- [ ] Add a "copy output path" button next to the checkpoint folder opener
- [ ] Per-image caption editing tab (open `.txt` alongside image for quick correction)
