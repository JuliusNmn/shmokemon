# 2D Buddy Sandbox

Minimal Rapier2D + Macroquad playground for prototyping RL-friendly simulations. The `SimulationWorld` struct seeds a simple scene (floor plus a falling rectangle) that can be reused offline or visualized in real time.

## Prerequisites

- Rust toolchain `stable` (1.91 or newer recommended)  
- macOS/Linux with GPU support for Macroquad (works headless/offline too)

## Offline stepping (no graphics)

```
cargo run --release -- <steps>
```

- `steps` defaults to `240` when omitted.
- Prints sampled position/velocity data to stdout for logging into RL buffers.

## Online visualization (Macroquad window)

```bash
cargo run --release --bin visualize [-- <step_limit>]
```

- Omit `step_limit` for an endless run.
- The renderer advances physics every few frames (`FRAMES_PER_STEP`) so you can tweak visuals without affecting the simulation state.

### Saving physics states from the visualizer

While the visualizer is running you can press:

- `D` to dump the current physics state.

Each dump creates a directory under `state_store/` (or the `--state-dir` you passed) with:

- `state.bin` – bincode-serialized `SimulationWorld`
- `state.png` – PNG screenshot of the current visual world

Example layout:

```text
state_store/
  state_0123abcddeadbeef/
    state.bin
    state.png
```

## GA training with saved states

The GA trainer (`train_ga`) will automatically use snapshots from `state_store/` if they exist.

- **Basic training run**:

  ```bash
  cargo run --release --bin train_ga -- \
    --population 64 \
    --top-k 8 \
    --generations 50 \
    --output best_brain_stand_upright.bin
  ```

- During evaluation, each episode samples one snapshot from `state_store/` (preferring `state.bin` files inside snapshot directories). If no states exist, it falls back to `SimulationWorld::new()`.

When a new best brain is found, `train_ga` writes:

- `best_brain_stand_upright.bin` – network weights
- `best_brain_stand_upright.bin.snapshots.txt` – metadata describing which snapshot was used for that best evaluation:

```text
best_score      <score>
snapshot_index  <index_into_loaded_snapshot_list>
snapshot_path   <path/to/state_store/.../state.bin>
```

## Notes

- All scene constants live in `src/lib.rs`; tweak sizes, gravity, or damping there to explore new setups.
- The visualizer and offline runner both instantiate `SimulationWorld::new()` when no state is provided, ensuring trajectories match bit-for-bit between modes.

