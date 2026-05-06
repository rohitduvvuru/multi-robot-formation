# Changes

## Performance-focused updates

1. **Fixed PPO minibatch training bug in `code/train.py`**
   - Previously, the SGD loop iterated one timestep index at a time (`mb_inds = b_inds[start]`), which effectively used single-sample updates and ignored `sgd_minibatch_size`.
   - Updated loop to slice true minibatches (`mb_inds = b_inds[start:end]`) using `range(0, batch_size, minibatch_size)`.
   - Added a safety assertion that `sgd_minibatch_size <= max_time_steps`.

2. **Removed hard-coded agent count in `code/model.py`**
   - Replaced `self.n_agents = 5` with `self.n_agents = obs_space["pos"].shape[0]`.
   - Replaced hard-coded per-agent output width with `int(num_outputs / self.n_agents)`.
   - This makes model sizing follow environment configuration consistently and prevents shape mismatch bugs when changing team size.

## Why these changes

- The minibatch fix improves optimization stability and sample efficiency for PPO.
- Dynamic agent sizing improves robustness to environment config changes and reduces silent architectural mismatch.


3. **Added automatic teleoperation disturbance schedule in `code/train.py`**
   - Implemented `apply_auto_teleop(...)` with configurable grab/release probabilities and drift actions.
   - Added curriculum for `p_grab` via linear schedule from `p_grab_start` to `p_grab_end`.

4. **Added reward framing for maintenance vs adaptation in `code/train.py`**
   - Added adaptation windows triggered by teleop state toggles.
   - Applied configurable reward scaling (`maintenance_scale`, `adapt_scale`) per environment while in/out of adaptation windows.

5. **Added communication-radius annealing in `code/train.py` and `code/model.py`**
   - Added per-iteration comm-radius schedule (`start`→`end` over `anneal_frac` of training).
   - Added `Agent.set_comm_range()` to update GNN neighborhood radius during training.


6. **Ported training/eval runtime to CUDA in `code/train.py` and `code/eval.py`**
   - Switched runtime device selection to `cuda` when available, with CPU fallback for compatibility.
   - Updated env config at runtime to match selected torch device so environment tensors and policy tensors live on the same device.
   - Updated reward-scaling tensor creation to use the current returns device explicitly.


7. **Upgraded evaluation workflow in `code/eval.py`**
   - Added CLI flags: `--weights`, `--episodes`, and `--render`.
   - Added multi-episode evaluation loop with per-episode outputs and aggregate summary metrics (success rate, reward mean/std, length mean/std).
   - Kept CUDA-aware device selection for evaluation runtime.
