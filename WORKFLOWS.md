# BackPop hierarchical workflows

## Recommended production workflow: 3D selection-corrected inference

Use `run_hierarchical_3d.sh` for production population inference. This is the default scientific workflow because the selection function depends on both mass and redshift, and the 3D single-event likelihood/injection campaign keeps the event posterior base measure consistent with the LVK/Farr selection correction.

End-to-end outline:

1. Run the 3D single-event catalog with `lucky_strikes_zform` so each event writes `points.npy`, `log_w.npy`, `log_z.npy`, and mode metadata with `likelihood_mode=3D`.
2. Build the matching COSMIC merger injection catalog with `run_injections.py --config_name lucky_strikes_zform --likelihood_mode 3D --output_path ./injections/gwtc3_cosmic_mergers.npz`.
3. Provide the LVK found-injection file through `LVK_FOUND_PATH`.
4. Launch `./run_hierarchical_3d.sh`.

The script fails before sampling if event metadata, injection metadata, required files/fields, or selection-correction inputs are missing or inconsistent. Set `ALLOW_INCONSISTENT_SELECTION_MODEL=True` only for an intentional legacy diagnostic run.

## Fast diagnostic 2D workflow

Use `run_hierarchical_2d.sh` for quick comparisons to the Lucky Strikes-style 2D likelihood or for debugging sampler behavior. It is not the recommended production workflow unless you also provide a self-consistent 2D injection campaign generated with `--likelihood_mode 2D` and matching 2D event metadata.

The 2D script keeps the command-line interface unchanged and performs the same preflight consistency checks as the 3D script.

## No-selection smoke test

For a fast smoke test of the hierarchical sampler only, call `hierarchical_backpop_jax.py` directly and omit both `--injections_path` and `--lvk_found_path`. This intentionally disables selection effects and should not be used for scientific population constraints.

Example:

```bash
python hierarchical_backpop_jax.py \
  --results_root ./results \
  --config_name lucky_strikes_zform \
  --output_dir ./results/hierarchical/lucky_strikes_zform/nuts/no_selection_smoke \
  --n_samples 200 \
  --num_warmup 50 \
  --num_samples 50 \
  --num_chains 1
```

## Selection-estimator comparison

Use the workflow scripts for the guarded LVK/Farr estimator runs, then compare against alternative estimators or sampler implementations in separate output directories. Keep the event likelihood mode and injection campaign mode matched for every comparison. If comparing 2D and 3D, treat the 2D run as a diagnostic sensitivity check rather than a production replacement.
