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

```
cargo run --release --bin visualize [-- <step_limit>]
```

- Omit `step_limit` for an endless run.
- The renderer advances physics every few frames (`FRAMES_PER_STEP`) so you can tweak visuals without affecting the simulation state.

## Notes

- All scene constants live in `src/lib.rs`; tweak sizes, gravity, or damping there to explore new setups.
- The visualizer and offline runner both instantiate `SimulationWorld::new()`, ensuring trajectories match bit-for-bit between modes.

