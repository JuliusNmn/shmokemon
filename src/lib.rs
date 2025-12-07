mod physics;
mod buddy;
mod world;
mod grabbable_world;
mod rl_interface;
mod brain;
mod ga;

// Re-export public items
pub use physics::{PIXELS_PER_METER, RigidBodySnapshot, FLOOR_HALF_EXTENTS, FLOOR_HEIGHT};
pub use buddy::{
    Buddy,
    HEAD_RADIUS,
    TORSO_WIDTH,
    TORSO_HEIGHT,
    ARM_THICKNESS,
    UPPER_ARM_LENGTH,
    LOWER_ARM_LENGTH,
    LEG_THICKNESS,
    UPPER_LEG_LENGTH,
    LOWER_LEG_LENGTH,
};
pub use world::SimulationWorld;
pub use grabbable_world::{GrabbableWorld, BodyVisualShape};
pub use rl_interface::{
    BuddyIO, BuddySense, BuddyAction, LimbSense, JointSense, 
    AngleSense, VelocitySense
};
pub use brain::{Brain, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SPARSITY_INPUT_HIDDEN, SPARSITY_HIDDEN_OUTPUT};
pub use ga::train_stand_upright;
