# Anima TrainFlow — TODO / Build Spec

Design law for this app: **remove a decision, don't add an option.** Every enhancement automates expert knowledge and surfaces it as an editable default. No raw knobs the user must already understand. The suggest-steps helper is the template — follow it.

---

## Completed
- [x] Zero-tab single-page Gradio UI
- [x] Smart crop (U2Net subject-aware, multi-threaded)
- [x] Auto-captioning (WD14 EVA02 v3, multi-threaded)
- [x] Dataset validation (missing captions, oversized images, model paths)
- [x] Persistent settings (`training/settings.json`, auto-save on change)
- [x] Live training previews (auto-updating gallery every N steps)
- [x] Preset system — `training/presets.json`, applies optimizer/LR/rank/batch/steps/cadence (Phase 1)
- [x] Step suggestion helper — steps from image count × target exp/image, + cadence for ~6 checkpoints (Phase 1)
- [x] Tailscale access — `server_name="0.0.0.0"` in `ui.launch()`
- [x] Checkpoint → ComfyUI junction (`mklink /J` from `training/output` into loras dir)

---

## Verified backend capability (no sd-scripts edits needed)
Inventoried bundled `library/optimizer.py` (Anima fork shares the upstream kohya tree):
- Warmup: `lr_warmup_steps` honored; **accepts a float ratio** — pass `0.1` for 10%.
- Schedulers present: constant, constant_with_warmup, cosine, cosine_with_restarts, linear, polynomial, inverse_sqrt.
- `network_alpha` is an independent network arg.
All items below are app-side only.

---

## DROPPED
- ~~Network alpha override (16/8)~~ — rejected. A second knob for a second-order gain already reachable via LR/exposure. Keep `network_alpha = network_rank` hardcoded. Rank dropdown (16/32) stays the only capacity lever.

---

## SHIP NOW — invisible improvement, no UI
### W. Bake in LR warmup
- [ ] In `create_training_toml`, emit `lr_warmup_steps = 0.1` (float ratio; backend computes step count)
- [ ] Gate: only for warmup-capable schedulers (cosine etc.). Skip for `constant` and Prodigy (schedule-free → dummy scheduler). Scheduler is currently hardcoded cosine for non-Prodigy, so this is a one-line add inside that branch.
- [ ] No UI. No preset key. User never sees it.

---

## TIER 1 — siblings of suggest-steps (cheap, high payoff)

### T1a. Auto-LR from optimizer × batch
Kills the second expert-only field. LR should populate, not be typed.
- [ ] Extend `handle_optimizer_change` (~line 908) to also read `batch_size_input`; add a `.change` trigger on `batch_size_input` calling the same logic.
- [ ] Lookup table (comment as heuristic starting points, batch-4 anchored from source post):
  - Prodigy → `1.0` (any batch)
  - AdamW8bit / AdamW: batch 1 → `0.00005`, batch 2 → `0.00006`, batch 4 → `0.00008`, batch 8 → `0.00012`
  - Unlisted batch → nearest listed
- [ ] Matches existing pattern: populates the field, user can override after. Do not lock it. `lr_input` stays editable + persisted; only the *default* is now computed.

### T1b. Exposures/image gauge (live readout)
Surfaces the governing overfit metric the user can't see today.
- [ ] Read-only `gr.Markdown`/`Textbox` below the steps row.
- [ ] `.change` on steps / batch / grad_acc / dataset_path → recompute `exp = steps*batch*grad_acc / num_images` (reuse `count_dataset_images`).
- [ ] Show value + band: ≤24 cool · 25–35 healthy · 36–50 warm · >50 fry-risk. (Style bands; same numbers suggest-steps uses.)
- [ ] Pure display. No field writes.

### T1c. Bucket-vs-batch warning (into existing dataset validation)
Catches the silent failure where batch can't fill a bucket → repeats inflate invisibly.
- [ ] New helper: bucket every image using sd-scripts rules — `bucket_reso_steps=64`, `min_bucket_reso=256`, `max_bucket_reso=max_bucket`, `bucket_no_upscale=True`. Count per bucket. Must match `create_dataset_toml` bucketing or the warning lies.
- [ ] If `min(bucket_counts) < batch_size`: warn in validation output — name the bucket, its count, the fix ("drop to batch N or raise repeats"). Note grad-accum is exempt (crosses buckets).
- [ ] Warning only — never block training.

---

## TIER 2 — capstone (wire the atoms into one action)

### T2. "Analyze & Configure" button
Folder in → sane config out, one click. No new logic, just chains existing auto-features.
- [ ] Button near `dataset_path`. On click, in order:
  1. `count_dataset_images` + `analyze_dataset_resolution` → set resolution fields
  2. `suggest_steps` → set steps + save/sample cadence
  3. T1a logic → set LR from current optimizer × batch
  4. T1c → run bucket check, surface any warning
  5. T1b → refresh the gauge
- [ ] Single populate of all derived fields; user reviews, then trains.
- [ ] This is what makes it feel like a flow trainer instead of a form. Highest seamlessness-per-effort once T1 exists.

---

## TIER 3 — flagship (closes the loop; real build)

### T3. Post-train checkpoint A/B gallery
Overfit is non-monotonic; the peak must be found by comparison, not calculation.
- [ ] **Cheap MVP first:** the trainer already writes sample images per cadence to `training/output/<project>/sample/` (named by step). After a run, scan that dir, group by step, lay out in a labeled gallery (step N = checkpoint at N). No re-inference — collection + layout only.
- [ ] Label each tile with step + matching checkpoint filename so user maps winner → `.safetensors`.
- [ ] Optional later: re-run a fresh prompt set against the last K saved `.safetensors` (reuse the sampling path pointed at saved files). Only if in-training samples prove insufficient.
- [ ] Goal: pick the best checkpoint in-app, load only that one into ComfyUI.

---

## TIER 4 — polish (trivial, reassuring)

### T4. Training ETA
- [ ] Parse s/it from the streamed sd-scripts progress in `output_log` after ~5 steps.
- [ ] Show "≈ X min remaining" from `(total_steps - current) * s_per_it`. Display only.

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

## Misc
- [ ] Confirm `python_embeded/` is in `.gitignore`; add if not
- [ ] "Copy output path" button next to checkpoint folder opener
- [ ] Per-image caption editing (open `.txt` alongside image for quick correction)

---

## Recommended ship order
W (free, do alongside any edit) → T1a + T1b (finish the "never type a number you don't understand" story) → T1c (silent-failure insurance) → T2 (tie the bow) → T3 (flagship outcome feature) → T4 (polish).

## Standing constraints (apply to every item)
- Never reorder `DEFAULT_SETTINGS` / `all_settings_list`; append new persisted keys to the end of BOTH at the same position.
- No repeats field — steps drive repeats in `create_dataset_toml`.
- Non-persisted helper controls (gauges, target field) stay OUT of `all_settings_list`.
- Don't touch the embedded Python env for any Tier 1–4 item — all are pure app-side.
