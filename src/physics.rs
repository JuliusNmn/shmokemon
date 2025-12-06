use rapier2d::prelude::*;

// Simulation constants
pub const FIXED_TIME_STEP: Real = 1.0 / 60.0;
pub const PIXELS_PER_METER: f32 = 80.0;

// Collision groups
pub const GROUP_WORLD: u32 = 0b0001;
pub const GROUP_BUDDY: u32 = 0b0010;

// World constants
pub const FLOOR_HALF_EXTENTS: [Real; 2] = [2.5, 0.1];
pub const FLOOR_HEIGHT: Real = -1.5;

// Buddy spawn
pub const BUDDY_SPAWN_HEIGHT: Real = 0.2;

#[derive(Debug, Clone)]
pub struct RigidBodySnapshot {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub rotation: f32,
}
