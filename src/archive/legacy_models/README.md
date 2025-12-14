# Legacy Models Archive

This directory contains deprecated/legacy model implementations that are no longer used in the main project but kept for historical reference.

## Files

### `diffusion_model.py`
**Status**: Replaced by `mid_model.py`

**Reason**:
- Initial diffusion implementation
- Superseded by MID (Motion Indeterminacy Diffusion)
- Kept for reference only

---

### `model_comparison.py`
**Status**: Deprecated

**Reason**:
- Early model comparison utilities
- No longer used in current pipeline
- Kept for historical reference

---

## Note

These files are **not imported** anywhere in the current codebase and should **not be used** for new development.

For current implementations, see:
- `src/models/mid_model.py` - MID Diffusion model
- `src/models/mid_integrated.py` - Fully integrated model
- `src/baselines/` - Baseline models for comparison
