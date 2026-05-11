# Changes Log

## 2026-05-10

Implemented requested training-improvement workflow:

- Implemented a uniform default initial n_present distribution in RandomTeleop, so training sees each robot-count regime (1..10) equally by default instead of the prior hand-biased mix. This should improve robustness/generalization across counts without adding rollout or model compute.

- Added an optional --anneal-lr flag to train_hallway.py and implemented lightweight linear learning-rate annealing across iterations. This typically improves late-stage PPO stability at negligible runtime cost.

- Expanded default periodic eval coverage from sparse count checkpoints to all regimes 1..10 via --eval-n-present-counts default. This improves checkpoint selection quality with modest evaluation overhead only (not rollout/training-step overhead).

1. **Comm-range sweep + SGD-epoch sweep runner**
   - Added `code/scripts/sweep_train.py`.
   - Script runs two sweep phases:
     - communication radius sweep (`--comm-ranges`, defaults `2.0,3.0,4.0`) with fixed middle SGD epoch value.
     - SGD epochs sweep (`--sgd-iters`, defaults `4,6,8`) with fixed middle comm range value.
   - Each run calls `code/train_hallway.py` with explicit tags and `--anneal-lr`.
   - Optional `--curriculum-teleop` support propagates to all sweep runs.

2. **Disturbance curriculum in trainer**
   - Updated `code/train_hallway.py` with:
     - `--curriculum-teleop`
     - `--curriculum-start-scale` (default `0.25`)
   - When enabled, teleop probabilities `p_grab`, `p_release`, `p_spawn`, `p_delete` are linearly ramped from
     `curriculum_start_scale * base_prob` at the first iteration to `1.0 * base_prob` by the last iteration.
   - This keeps early training easier and gradually restores full disturbance difficulty.

3. **No architecture-size increase**
   - Changes focus on training schedule/hyperparameter search behavior, not larger model or rollout horizon.