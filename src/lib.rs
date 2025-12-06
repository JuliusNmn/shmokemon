mod physics;
mod buddy;
mod world;
mod grabbable_world;
mod rl_interface;
mod brain;
mod ga;

// Re-export public items
pub use physics::{PIXELS_PER_METER, RigidBodySnapshot, FLOOR_HALF_EXTENTS, FLOOR_HEIGHT};
pub use buddy::{Buddy, BuddyPart, BuddyPartShape};
pub use world::SimulationWorld;
pub use grabbable_world::GrabbableWorld;
pub use rl_interface::{
    BuddyIO, BuddySense, BuddyAction, LimbSense, JointSense, 
    AngleSense, VelocitySense
};
pub use brain::{Brain, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SPARSITY_INPUT_HIDDEN, SPARSITY_HIDDEN_OUTPUT};
pub use ga::train_stand_upright;
