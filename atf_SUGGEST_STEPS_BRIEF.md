**STATUS: IMPLEMENTED** (commit `96c280bf`; verified 2026-07-02). Shipped as specced, including the non-persisted target field and sanity-check math. Kept as historical spec; do not re-execute.

# FRAG ORDER ADDENDUM — Step Suggestion Helper

Scope: Phase 1, additive. Reads image count from the dataset path and suggests step count + checkpoint cadence. Suggests STEPS ONLY — not batch, LR, or rank (those don't derive from image count; suggesting them would be invention).
Constraint: do not touch the `settings.json` zip-mapping. The new control is NOT persisted — keep it out of `DEFAULT_SETTINGS` and `all_settings_list`.

---

## RATIONALE (bake into a comment)
Governing metric for Anima LoRA overfit is exposures-per-image:
`exp_per_image = steps * batch * grad_accum / num_images`
Style band ≈ 25–35 exp/image. Default target 30. Steps derive from target + image count + effective batch.
Cadence targets ~6 checkpoints so the user can A/B for the overfit sweet spot (training is non-monotonic; the peak must be found empirically, not calculated).

---

## STEP 1 — shared image counter
A count exists inline in `create_dataset_toml` (~line 238). Extract it to a helper near `analyze_dataset_resolution` (~line 194) so both call sites share it:
```python
VALID_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

def count_dataset_images(dataset_path: str) -> int:
    path = Path(dataset_path)
    if not path.exists():
        return 0
    return len([f for f in path.glob('*') if f.is_file() and f.suffix.lower() in VALID_IMG_EXTS])
```
Optional: refactor `create_dataset_toml` to call this. Not required for the feature; do it only if clean.

## STEP 2 — suggestion function
Define alongside `handle_optimizer_change` (~line 908):
```python
def suggest_steps(dataset_path, batch_size, grad_acc, target_exp):
    n = count_dataset_images(dataset_path)
    if n == 0:
        return gr.update(), gr.update(), gr.update(), "No images found at dataset path."

    eff_batch = max(1, int(batch_size) * int(grad_acc))
    target = float(target_exp) if target_exp else 30.0

    def to_steps(exp):
        return max(1, round((exp * n / eff_batch) / 25) * 25)  # round to nearest 25

    steps = to_steps(target)
    lo, hi = to_steps(target - 5), to_steps(target + 5)
    cadence = max(50, round((steps / 6) / 25) * 25)  # ~6 checkpoints for A/B

    info = (f"{n} images | eff. batch {eff_batch} | target {target:.0f} exp/img\n"
            f"Suggested: {steps} steps (band {lo}-{hi} @ {target-5:.0f}-{target+5:.0f} exp/img)\n"
            f"Save/preview every {cadence} steps -> ~{steps // cadence} checkpoints to A/B")

    return gr.update(value=steps), gr.update(value=cadence), gr.update(value=cadence), info
```
Output order: steps, save_steps, sample_steps, info. Keep aligned with wiring below.

## STEP 3 — UI
In the right column, near the steps row (~line 949). Add the target field and button, plus a read-only info box:
```python
with gr.Row():
    target_exp_input = gr.Number(label="Target exp/image", value=30, precision=0, min_width=120)
    suggest_btn = gr.Button("Suggest Steps", variant="secondary")
suggest_info = gr.Textbox(label="Suggestion", interactive=False, lines=3)
```
`target_exp_input` is intentionally NOT added to `all_settings_list` — non-persisted, resets to 30 on reload. Do not add it to `DEFAULT_SETTINGS`.

## STEP 4 — wire it
Near the other handlers (~line 1009):
```python
suggest_btn.click(
    fn=suggest_steps,
    inputs=[dataset_path, batch_size_input, grad_acc_input, target_exp_input],
    outputs=[steps_input, save_steps_input, sample_steps_input, suggest_info],
)
```
Outputs list order MUST match `suggest_steps` return: steps, save_steps, sample_steps, info.

---

## INTERACTION NOTES
- Run order: set batch + grad_accum (or apply a preset) FIRST, then Suggest Steps — the calc reads current batch. If the user changes batch after suggesting, they re-click.
- Suggestion writes to steps/save/sample; auto-save fires through the normal `.change` path, so values persist as usual. Only the target field is ephemeral.
- The band (±5 exp/img) is shown but not applied — user sees the safe range, the populated value is mid-band.

## DON'T
- Don't suggest batch/LR/rank from image count.
- Don't persist `target_exp_input`.
- Don't add a repeats field (steps drive repeats in `create_dataset_toml`).
- Don't reorder `DEFAULT_SETTINGS` / `all_settings_list`.

## DONE
Click Suggest Steps with a valid dataset path → steps + cadence populate, info box explains the math, training runs unchanged. Empty/invalid path → info box says so, no fields touched.

---

## SANITY CHECK (84 images, batch 4, GA 1, target 30)
steps = round((30 * 84 / 4)/25)*25 = round(630/25)*25 = 625
band  = 25 exp -> 525, 35 exp -> 735
cadence = round((625/6)/25)*25 = round(104/25)*25 = 100  -> ~6 checkpoints
Matches the data-backed band. If output differs, the formula is wired wrong.
