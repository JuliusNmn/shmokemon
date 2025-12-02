# Reinforcement Learning Interface

This document describes the RL interface for the 2D Buddy simulation.

## Overview

The `BuddyIO` interface provides a bridge between the hierarchical physics simulation and flat arrays suitable for reinforcement learning algorithms.

## Architecture

### Hierarchical Structures

The interface uses hierarchical structs that mirror the buddy's physical structure:

- **`BuddySense`**: Complete observation of the buddy's state
  - 10 limbs (torso, head, 2 upper arms, 2 lower arms, 2 upper legs, 2 lower legs)
  - 9 joints (neck, 2 shoulders, 2 elbows, 2 hips, 2 knees)

- **`BuddyAction`**: Control commands for the buddy
  - 9 joint angular velocities (one per joint)

### Flat Arrays

For RL algorithms, the interface provides flattening/unflattening:

- **Sense array**: 68 floats
  - 10 limbs × 5 values each = 50 floats
    - sin(absolute_angle)
    - cos(absolute_angle)
    - angular_velocity
    - absolute_velocity.x
    - absolute_velocity.y
  - 9 joints × 2 values each = 18 floats
    - angle (relative between connected bodies)
    - angular_velocity (relative)

- **Action array**: 9 floats
  - One angular velocity per joint

## Usage

### Basic Example

```rust
use two_dbuddy::{SimulationWorld, BuddyIO};

let mut world = SimulationWorld::new();

// Get observation
let sense_flat = world.buddy_sense_flat();
assert_eq!(sense_flat.len(), 68);

// Apply action
let action = vec![0.0; 9]; // All joints stationary
world.apply_buddy_action_flat(&action);

// Step simulation
world.step();
```

### Hierarchical Access

```rust
// Get hierarchical sense data
let sense = world.buddy_sense();

// Access specific limb data
println!("Torso angle: sin={}, cos={}", 
    sense.torso.absolute_angle.sin,
    sense.torso.absolute_angle.cos);
println!("Torso velocity: x={}, y={}", 
    sense.torso.absolute_velocity.x,
    sense.torso.absolute_velocity.y);

// Access joint data
println!("Front shoulder angle: {}", sense.front_shoulder_joint.angle);
```

### Custom RL Loop

```rust
use two_dbuddy::{SimulationWorld, BuddyIO, BuddyAction};

let mut world = SimulationWorld::new();

for episode in 0..1000 {
    // Reset world
    world = SimulationWorld::new();
    
    for step in 0..500 {
        // Get observation
        let observation = world.buddy_sense_flat();
        
        // Your RL agent decides action
        let action = your_agent.act(&observation);
        
        // Apply action
        world.apply_buddy_action_flat(&action);
        
        // Step simulation
        world.step();
        
        // Calculate reward
        let reward = calculate_reward(&world);
        
        // Train agent
        your_agent.learn(observation, action, reward);
    }
}
```

## Limb and Joint Naming

### Limbs (10 total)
- `torso`
- `head`
- `front_arm_upper`
- `front_arm_lower`
- `back_arm_upper`
- `back_arm_lower`
- `front_leg_upper`
- `front_leg_lower`
- `back_leg_upper`
- `back_leg_lower`

### Joints (9 total)
- `neck_joint` (torso ↔ head)
- `front_shoulder_joint` (torso ↔ front_arm_upper)
- `front_elbow_joint` (front_arm_upper ↔ front_arm_lower)
- `back_shoulder_joint` (torso ↔ back_arm_upper)
- `back_elbow_joint` (back_arm_upper ↔ back_arm_lower)
- `front_hip_joint` (torso ↔ front_leg_upper)
- `front_knee_joint` (front_leg_upper ↔ front_leg_lower)
- `back_hip_joint` (torso ↔ back_leg_upper)
- `back_knee_joint` (back_leg_upper ↔ back_leg_lower)

## API Reference

### `BuddyIO`

Static methods for converting between hierarchical and flat representations:

#### `sense(buddy, rigid_body_set, impulse_joint_set) -> BuddySense`
Get hierarchical sense data from the physics simulation.

#### `flatten_sense(sense: &BuddySense) -> Vec<f32>`
Convert hierarchical sense to flat array (68 floats).

#### `unflatten_action(flat_action: &[f32]) -> BuddyAction`
Convert flat action array (9 floats) to hierarchical action.

#### `apply_action(action, buddy, impulse_joint_set)`
Apply hierarchical action to the buddy.

#### `sense_size() -> usize`
Returns 68 (size of flattened sense array).

#### `action_size() -> usize`
Returns 9 (size of action array).

### `SimulationWorld` Convenience Methods

#### `buddy_sense() -> BuddySense`
Get hierarchical sense data.

#### `buddy_sense_flat() -> Vec<f32>`
Get flat sense array.

#### `apply_buddy_action(&BuddyAction)`
Apply hierarchical action.

#### `apply_buddy_action_flat(&[f32])`
Apply flat action array.

## Notes

- **Angles**: Absolute angles are measured from the upward direction (0 radians = straight up)
- **Joint angles**: Relative angles between connected bodies
- **Angular velocities**: Positive = counter-clockwise rotation
- **Coordinate system**: x = horizontal, y = vertical (positive = up)
- **Joint limits**: Each joint has limits of ±π/2 radians
- **Motor model**: Force-based with stiffness=20.0, damping=2.0, max_force=50.0

## Running the Demo

```bash
cargo run --example rl_demo
```

This will demonstrate the RL interface with sample output showing the hierarchical structure and flat arrays.
