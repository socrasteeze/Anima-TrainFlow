**STATUS: IMPLEMENTED** (commit `96c280bf`; verified 2026-07-02). Phase 1 shipped as specced. Phase 2: alpha override DROPPED by design, warmup shipped as baked-in default, Rex/CAME iceboxed — see TODO.md. Kept as historical spec; do not re-execute.

# FRAG ORDER — Preset System for Anima-TrainFlow fork

Target: `app.py` (single-file Gradio app, ~1036 lines). Backend: modified `sd-scripts` for Anima 2B.
Objective: add selectable training presets driving the existing parameter controls, plus the backend hooks needed to fully express them.
Constraint: do not break `settings.json` auto-save. Work in two phases. Phase 1 ships standalone. Phase 2 is optional uplift.

---

## INTEL — current state (verify before touching)

Param controls live in the right column of the `gr.Blocks` UI (lines ~941–952):
- `rank_input` (Number, "Network Rank")
- `lr_input` (Textbox, "Learning Rate")
- `optimizer_input` (Dropdown, choices `["Prodigy","AdamW8bit","AdamW"]`)
- `batch_size_input`, `steps_input`, `save_steps_input`, `sample_steps_input`, `grad_acc_input`

Persistence (lines ~127–166):
- `DEFAULT_SETTINGS` dict defines key set AND order.
- `auto_save_state(*args)` does `dict(zip(DEFAULT_SETTINGS.keys(), args))`. Order coupling is load-bearing.
- `all_settings_list` (line ~998) must stay index-aligned with `DEFAULT_SETTINGS.keys()`.

Hardcoded behavior the post wants changed:
- `network_alpha = network_rank` — `create_training_toml`, line ~278. Alpha cannot diverge from rank.
- Scheduler — `create_training_toml`, lines ~281–285: `constant` for Prodigy, else `cosine`. No warmup. No Rex.
- Repeats — `create_dataset_toml`, lines ~246–248: derived as `ceil(max_steps * batch * grad_acc / num_images)`. Steps drive repeats. There is NO repeats input and there must not be one.
- `handle_optimizer_change` (line ~908) swaps LR between Prodigy (`1.0`) and a saved AdamW LR. Presets must cooperate with this.

---

## RULE 1 — DO NOT add a repeats control
The post sets repeats manually; this trainer derives them from steps. Inverse model. Express the post's intent by setting STEPS, not repeats. State this in a code comment so it isn't "fixed" later.

## RULE 2 — Keep presets OUT of the settings zip-mapping
Store presets in a separate `training/presets.json`. Do NOT add preset definitions to `DEFAULT_SETTINGS`. Applying a preset writes to the existing component values (which then auto-save through the normal path). This avoids touching the fragile zip ordering.

---

## PHASE 1 — Presets over existing fields (ship this first)

1. Create `training/presets.json`. Schema:
```json
{
  "presets": [
    {
      "name": "Style — AdamW8bit (safe)",
      "optimizer": "AdamW8bit",
      "learning_rate": "0.00008",
      "network_rank": 16,
      "training_steps": 1000,
      "train_batch_size": 4,
      "gradient_accumulation_steps": 1,
      "save_steps": 250,
      "sample_steps": 250
    },
    {
      "name": "Style — AdamW8bit (low VRAM)",
      "optimizer": "AdamW8bit",
      "learning_rate": "0.00008",
      "network_rank": 16,
      "training_steps": 1000,
      "train_batch_size": 2,
      "gradient_accumulation_steps": 2,
      "save_steps": 250,
      "sample_steps": 250
    },
    {
      "name": "Prodigy (default / auto-LR)",
      "optimizer": "Prodigy",
      "learning_rate": "1.0",
      "network_rank": 16,
      "training_steps": 1000,
      "train_batch_size": 4,
      "gradient_accumulation_steps": 1,
      "save_steps": 250,
      "sample_steps": 250
    }
  ]
}
```
LR values are batch-4 figures from the source post (AdamW 7e-5–1e-4, midpoint 8e-5). Do not invent ranges; use these.

2. Loader. Add near `load_settings()`:
```python
PRESETS_FILE = TRAIN_BASE / "presets.json"
def load_presets():
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("presets", [])
        except Exception:
            pass
    return []
```

3. UI. In the right column, above the `rank_input` row (~line 942), add:
```python
with gr.Row():
    preset_dd = gr.Dropdown(label="Preset", choices=[p["name"] for p in load_presets()], value=None)
    apply_preset_btn = gr.Button("Apply Preset", variant="secondary")
```

4. Apply handler. Define alongside `handle_optimizer_change`:
```python
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
```
Output order MUST match the wiring below.

5. Wire it. Near the other `.click`/`.change` handlers (~line 1009). Apply BEFORE the optimizer-change reconciliation so LR isn't clobbered:
```python
apply_preset_btn.click(
    fn=apply_preset,
    inputs=[preset_dd],
    outputs=[optimizer_input, lr_input, rank_input, steps_input,
             batch_size_input, save_steps_input, sample_steps_input  # NOTE: keep order identical to apply_preset return
            ],
)
```
WARNING: the `apply_preset` return has 8 values; the outputs list must list all 8 components in the SAME order (optimizer, lr, rank, steps, batch, grad_acc, save_steps, sample_steps). Fix the list above to include `grad_acc_input`. Do not desync.

6. Verify: launch, select each preset, confirm fields populate, confirm `settings.json` saves, confirm a training run starts. Phase 1 done.

Phase 1 covers: optimizer, LR, rank, steps, batch, grad_acc, save/sample cadence. It does NOT cover alpha≠rank, warmup, Rex scheduler, or CAME. Those are Phase 2.

---

## PHASE 2 — Backend uplift (optional; needed for full post fidelity)

Only attempt if you want 16/8 alpha, warmup, Rex, or CAME. Each is a backend edit. Confirm the bundled sd-scripts fork supports the arg before exposing it — grep `training/sd-scripts` for the flag first.

### 2a. Alpha override (16/8)
- `create_training_toml`: replace `network_alpha = network_rank` with a passed-in `alpha` param; thread it through `start_training` and the call site (~line 810).
- Add `network_alpha_input = gr.Number(label="Network Alpha", value=cs.get("network_alpha", 16))`.
- New persisted key → must be added to BOTH `DEFAULT_SETTINGS` and `all_settings_list` at the SAME relative position, or the zip-mapping corrupts. Append at the end of both to minimize risk.
- Add `network_alpha` to preset schema once exposed.

### 2b. Warmup (~10%)
- sd-scripts uses `lr_warmup_steps` (int) or a ratio depending on fork. Grep to confirm.
- In `create_training_toml`, compute `warmup = int(max_steps * 0.10)` and inject `"lr_warmup_steps": warmup` when scheduler supports it (cosine/Rex; NOT constant/Prodigy).

### 2c. Rex scheduler
- Confirm the fork registers `rex` as an `lr_scheduler` choice. If not, it requires a custom scheduler module — out of scope, fall back to cosine.
- If supported: add scheduler to preset schema and select in `create_training_toml` instead of the hardcoded branch.

### 2d. CAME optimizer
- CAME is NOT in the bundled optimizer list and may not be importable. Grep sd-scripts for `came` / `CAME`. If absent, it needs `pytorch-optimizer` or equivalent wired into the optimizer factory — non-trivial, confirm before promising.
- If supported: add `"CAME"` to `optimizer_input` choices and `handle_optimizer_change`. CAME LR band is 2e-5–5e-5 (≈half of AdamW). Add a CAME preset at lr `0.00003`. Do NOT let it inherit AdamW LR — that fries the LoRA.

---

## DON'T
- Don't reorder `DEFAULT_SETTINGS` or `all_settings_list`.
- Don't add a repeats input.
- Don't expose CAME/Rex/warmup in the UI before confirming backend support — a dead dropdown choice that errors mid-run is worse than its absence.
- Don't hardcode preset values in `app.py`; they live in `presets.json` so they're editable without code changes.

## DEFINITION OF DONE
Phase 1: preset dropdown + Apply populates all eight fields, persists, trains. Presets editable via JSON. No regression to existing manual flow.
