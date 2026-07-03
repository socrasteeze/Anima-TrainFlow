# Anima TrainFlow — TODO / Build Spec

Design law for this app: **remove a decision, don't add an option.** Every enhancement automates expert knowledge and surfaces it as an editable default. No raw knobs the user must already understand. The suggest-steps helper is the template — follow it.

> **Status (2026-07-02):** All planned tiers (W, T1a–T1c, T2, T3 MVP, T4) plus the misc items shipped in commits `78505014` and `252b4d0f`. A code review verified every feature against its spec — implementations are correct and the settings zip-mapping is intact (26 keys, order preserved). The six review fix-ups (F1–F6) are now applied and verified, and git hygiene is done. **Only optional work remains: T3 Phase 2 (re-inference A/B), or stop — the tool is feature-complete for its design law.**

---

## Completed
- [x] Zero-tab single-page Gradio UI
- [x] Smart crop (U2Net subject-aware, multi-threaded)
- [x] Auto-captioning (WD14 EVA02 v3, multi-threaded)
- [x] Dataset validation (missing captions, oversized images, model paths)
- [x] Persistent settings (`training/settings.json`, auto-save on change)
- [x] Live training previews (auto-updating gallery every N steps)
- [x] Preset system — `training/presets.json`, applies optimizer/LR/rank/batch/steps/cadence (`apply_preset`, app.py ~986)
- [x] Step suggestion helper — steps from image count × target exp/image + cadence for ~6 checkpoints (`suggest_steps`, app.py ~965)
- [x] Tailscale access — `server_name="0.0.0.0"` in `ui.launch()`
- [x] Checkpoint → ComfyUI junction (`mklink /J` from `training/output` into loras dir)
- [x] **W** — LR warmup baked in: `lr_warmup_steps = 0.1` (float ratio) on the cosine branch only; Prodigy skips it (schedule-free, `safeguard_warmup=True` instead). `create_training_toml`, app.py ~302–308.
- [x] **T1a** — Auto-LR from optimizer × batch: `_ADAMW_LR_BY_BATCH` table (app.py ~953), `handle_optimizer_change` also fires on `batch_size_input.change`. Field stays editable.
- [x] **T1b** — Exposures/image gauge: `compute_exp_gauge` (app.py ~1006), live `.change` on dataset path / steps / batch / grad-acc. Display only.
- [x] **T1c** — Bucket-vs-batch warning: `check_bucket_batch` (app.py ~1021), surfaced in Analyze & Configure and at training start (non-blocking). *Has fix-ups — see below.*
- [x] **T2** — "Analyze & Configure" button: `analyze_and_configure` (app.py ~1053) chains resolution + steps + cadence + LR + bucket check + gauge in one click.
- [x] **T3 (MVP)** — Post-train A/B gallery: `get_ab_gallery` (app.py ~1094) scans `training/output/<proj>/sample/`, sorts by step, labels each tile with step + matching `.safetensors` name.
- [x] **T4** — Training ETA: tqdm-rate regex in the log stream (app.py ~824–899), "≈ Xm remaining" textbox.
- [x] "Copy output path" button (app.py ~1178, ~1328)
- [x] Per-image caption editor (accordion; load/prev/next/save, app.py ~1119–1141, ~1331–1339)
- [x] `python_embeded/` and `models/` are in `.gitignore`

---

## FIX-UPS — found in 2026-07-02 code review — ✅ ALL RESOLVED (verified same day)

Fixed and verified by importing the real functions and asserting behavior (12/12 checks pass; the UI graph also rebuilds cleanly with the rewired handlers). Settings zip-mapping left intact — `training_inputs` (21) and `all_settings_list` (26) were not touched.

- [x] **F1.** Bucket check at training start no longer hardcodes `512, 768` — `start_training` gained `side_min/side_max` params (defaults 512/768) and the click passes `training_inputs + [side_min_input, side_max_input]`. `training_inputs` itself untouched, so the zip-mapping is safe.
- [x] **F2.** `analyze_and_configure` now runs `check_bucket_batch` against the freshly computed `base_res`/`max_bucket_res` (what the side fields become), not the stale incoming values.
- [x] **F3.** `analyze_and_configure` honors `target_exp_input` (`float(target_exp) if target_exp else 30.0`), consistent with Suggest Steps.
- [x] **F4.** `compute_exp_gauge` returns `""` when steps/batch/grad-acc is `None` (a cleared Number field), so mid-edit no longer raises a TypeError toast.
- [x] **F5.** `parse_step` anchors on the fixed `_{step:06d}_{i:02d}_{14-digit-timestamp}` tail, so digit-bearing project names (`style2024`, `test_123456`) and seed suffixes parse correctly; legacy fallback retained.
- [x] **F6.** Dead `saved_adam_lr` `gr.State` removed; `handle_optimizer_change` simplified to `(opt, batch_size) -> lr` (auto-LR is the intended behavior, save/restore was vestigial); `analyze_and_configure` no longer takes the unused param.

### Known limitation (documented, not a bug to fix now)
T1c approximates sd-scripts *training* bucketing with the SmartCropper's *crop* buckets. Exact when the dataset was smart-cropped in-app (images land precisely on crop buckets); can misgroup for datasets cropped elsewhere. A faithful check would group by exact image dims (post-crop dims are already 64-multiples) under `bucket_no_upscale=True` rules from `create_dataset_toml`.

---

## GIT HYGIENE — repo state, not code

- [x] **Untracked `python_embeded/`** — 996 files removed from the index via `git rm -r --cached` (commit `31d979cc`); blobs stay in history, working tree is quiet.
- [x] **`models/*/put_model_here` placeholders** — restored (`git checkout -- models/`) so fresh clones keep the directory skeleton.
- [x] **Committed the docs** — `atf_*.md` briefs (IMPLEMENTED headers), TODO.md, CLAUDE.md refresh (commit `11866fdd`).
- [x] `training/settings.json` is machine-specific state (absolute local paths) — added to `.gitignore`; removed from CLAUDE.md's commit list. Do not commit it.

---

## TIER 3 PHASE 2 — optional, only if in-training samples prove insufficient

- [ ] Re-run a fresh prompt set against the last K saved `.safetensors` (reuse the sampling path pointed at saved files). Goal: pick the best checkpoint in-app, load only that one into ComfyUI. Build only if the MVP gallery's in-training samples aren't enough for A/B judging.

---

## Icebox — deferred, real cost / low payoff

### CAME optimizer
- Confirmed ABSENT from bundled sd-scripts (no `came` in tree).
- Needs a new dependency (`pytorch-optimizer` etc.) pip-installed into `python_embeded` — the embedded-env cascade risk that has bitten this setup.
- Marginal payoff: Prodigy + AdamW8bit cover style work; source post rated CAME a sidegrade (overfits on noise as often as it helps).
- If revisited: route via kohya's generic `--optimizer_type module.ClassName` importlib path. CAME LR band 2e-5–5e-5 (~half AdamW), default `0.00003`. Never inherit AdamW LR — instant fry.

### Rex scheduler
- Not built into the fork. Author's own verdict: cosine is equivalent. Cosine + warmup (W) covers the intent. Skip.

### DROPPED
- ~~Network alpha override (16/8)~~ — rejected. A second knob for a second-order gain already reachable via LR/exposure. Keep `network_alpha = network_rank` hardcoded. Rank stays the only capacity lever.

---

## Verified backend capability (no sd-scripts edits needed)
Inventoried bundled `library/optimizer.py` (Anima fork shares the upstream kohya tree):
- Warmup: `lr_warmup_steps` honored; **accepts a float ratio** — `0.1` = 10% (now in use).
- Schedulers present: constant, constant_with_warmup, cosine, cosine_with_restarts, linear, polynomial, inverse_sqrt.
- `network_alpha` is an independent network arg (unused by design — see DROPPED).
- Sample images are saved as `{output_name}_{steps:06d}_{i:02d}_{timestamp}{seed}.png` (`train_util.py` ~6765) — T3 depends on this naming.

---

## Standing constraints (apply to every item)
- Never reorder `DEFAULT_SETTINGS` / `all_settings_list`; append new persisted keys to the end of BOTH at the same position. `training_inputs` is a prefix of `all_settings_list` — growing it shifts the zip-mapping; pass extra event inputs as `training_inputs + [extra, ...]` at the `.click` site instead.
- No repeats field — steps drive repeats in `create_dataset_toml`.
- Non-persisted helper controls (gauges, target field) stay OUT of `all_settings_list`.
- Don't touch the embedded Python env — everything above is pure app-side.
- `training/settings.json` is local state: never commit it, never hand-edit expectations into it.

## Recommended order
~~F1–F6~~ ✅ done · ~~git hygiene~~ ✅ done → then either T3 Phase 2 or stop; the tool is feature-complete for its design law.
