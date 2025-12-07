use rapier2d::prelude::*;
use crate::world::SimulationWorld;
use crate::physics::PIXELS_PER_METER;
use crate::buddy::Buddy;
use crate::physics::RigidBodySnapshot;
use crate::rl_interface::{BuddySense, BuddyAction};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub enum BodyVisualShape {
    Circle { radius: Real },
    Box { half_extents: [Real; 2] },
}

/// State for grabbing/picking up the buddy
#[derive(Debug, Clone)]
pub struct GrabState {
    pub mouse_body_handle: RigidBodyHandle,
    pub joint_handle: ImpulseJointHandle,
    pub grabbed_body_handle: RigidBodyHandle,
    pub anchor_point: Point<Real>, // Local point on the grabbed body
}

/// A wrapper around SimulationWorld that adds mouse grabbing functionality
pub struct GrabbableWorld {
    world: SimulationWorld,
    grab_state: Option<GrabState>,
}

impl GrabbableWorld {
    /// Create a new GrabbableWorld
    pub fn new() -> Self {
        Self {
            world: SimulationWorld::new(),
            grab_state: None,
        }
    }
    
    /// Convert screen coordinates to world coordinates
    pub fn screen_to_world(&self, screen_x: f32, screen_y: f32, screen_width: f32, screen_height: f32) -> Point<Real> {
        let world_x = (screen_x - screen_width * 0.5) / PIXELS_PER_METER as f32;
        let world_y = (screen_height * 0.5 - screen_y) / PIXELS_PER_METER as f32;
        point![world_x as Real, world_y as Real]
    }
    
    /// Query which buddy body (if any) was clicked at the given world point
    fn query_buddy_part_at_point(&self, world_point: Point<Real>) -> Option<(RigidBodyHandle, Point<Real>)> {
        let buddy = self.world.buddy();
        let handles = [
            buddy.torso,
            buddy.head,
            buddy.front_arm_upper,
            buddy.front_arm_lower,
            buddy.back_arm_upper,
            buddy.back_arm_lower,
            buddy.front_leg_upper,
            buddy.front_leg_lower,
            buddy.back_leg_upper,
            buddy.back_leg_lower,
        ];

        // Check each buddy body part's collider to see if the point intersects
        for body_handle in handles {
            // Find colliders attached to this body
            for (_collider_handle, collider) in self.world.collider_set.iter() {
                if collider.parent() == Some(body_handle) {
                    // Check if point is inside the collider shape using intersection test
                    // Project the point onto the shape to see if it's inside
                    let proj = collider.shape().project_point(collider.position(), &world_point, false);
                    if (proj.point - world_point).norm() < 0.05 {
                        return Some((body_handle, world_point));
                    }
                }
            }
        }
        None
    }
    
    /// Create a grab at the specified world point (if clicking on a buddy part)
    pub fn try_grab_at_point(&mut self, world_point: Point<Real>) -> bool {
        // Release any existing grab first
        self.release_grab();
        
        // Query for buddy part at this point
        if let Some((body_handle, hit_point)) = self.query_buddy_part_at_point(world_point) {
            // Get the body to find the local anchor point
            if let Some(body) = self.world.rigid_body_set.get(body_handle) {
                // Convert world point to local point on the body
                let local_anchor = body.position().inverse() * hit_point;
                
                // Create a kinematic body at the mouse position (will be updated each frame)
                let mouse_body = RigidBodyBuilder::kinematic_position_based()
                    .translation(world_point.coords)
                    .build();
                let mouse_body_handle = self.world.rigid_body_set.insert(mouse_body);
                
                // Create a revolute joint connecting the mouse body to the clicked body
                // This allows rotation around the anchor point while keeping them connected
                let joint = RevoluteJointBuilder::new()
                    .local_anchor1(Point::origin()) // Anchor at origin of mouse body
                    .local_anchor2(local_anchor) // Anchor at clicked point on buddy part
                    .build();
                
                let joint_handle = self.world.impulse_joint_set.insert(
                    mouse_body_handle,
                    body_handle,
                    joint,
                    true,
                );
                
                self.grab_state = Some(GrabState {
                    mouse_body_handle,
                    joint_handle,
                    grabbed_body_handle: body_handle,
                    anchor_point: local_anchor,
                });
                
                return true;
            }
        }
        false
    }
    
    /// Update the mouse body position to follow the mouse
    pub fn update_mouse_position(&mut self, world_point: Point<Real>) {
        if let Some(ref grab_state) = self.grab_state {
            if let Some(mouse_body) = self.world.rigid_body_set.get_mut(grab_state.mouse_body_handle) {
                // For position-based kinematic bodies, set the next position directly
                mouse_body.set_next_kinematic_translation(world_point.coords);
            }
        }
    }
    
    /// Release the current grab
    pub fn release_grab(&mut self) {
        if let Some(grab_state) = self.grab_state.take() {
            // Remove the joint
            self.world.impulse_joint_set.remove(
                grab_state.joint_handle,
                true,
            );
            
            // Remove the mouse body
            self.world.rigid_body_set.remove(
                grab_state.mouse_body_handle,
                &mut self.world.island_manager,
                &mut self.world.collider_set,
                &mut self.world.impulse_joint_set,
                &mut self.world.multibody_joint_set,
                true,
            );
        }
    }
    
    /// Check if currently grabbing something
    pub fn is_grabbing(&self) -> bool {
        self.grab_state.is_some()
    }
    
    // Delegate all SimulationWorld methods
    pub fn step(&mut self) {
        self.world.step();
    }
    
    pub fn buddy(&self) -> &Buddy {
        self.world.buddy()
    }
    
    pub fn time(&self) -> Real {
        self.world.time()
    }
    
    pub fn body_snapshot(&self, handle: RigidBodyHandle) -> Option<RigidBodySnapshot> {
        self.world.body_snapshot(handle)
    }
    
    pub fn buddy_torso_state(&self) -> Option<RigidBodySnapshot> {
        self.world.buddy_torso_state()
    }
    
    pub fn buddy_head_state(&self) -> Option<RigidBodySnapshot> {
        self.world.buddy_head_state()
    }
    
    pub fn buddy_sense(&self) -> BuddySense {
        self.world.buddy_sense()
    }
    
    pub fn buddy_sense_flat(&self) -> Vec<f32> {
        self.world.buddy_sense_flat()
    }
    
    pub fn apply_buddy_action(&mut self, action: &BuddyAction) {
        self.world.apply_buddy_action(action);
    }
    
    pub fn apply_buddy_action_flat(&mut self, flat_action: &[f32]) {
        self.world.apply_buddy_action_flat(flat_action);
    }
    
    pub fn torque_history(&self) -> &VecDeque<Real> {
        self.world.torque_history()
    }

    pub fn set_gravity(&mut self, gravity: Vector<Real>) {
        self.world.set_gravity(gravity);
    }

    pub fn set_buddy_gravity_scale(&mut self, scale: Real) {
        self.world.set_buddy_gravity_scale(scale);
    }

    pub fn set_zero_gravity(&mut self) {
        self.world.set_gravity(vector![0.0, 0.0]);
    }

    pub fn set_default_gravity(&mut self) {
        self.world.set_gravity(vector![0.0, -9.81]);
    }

    /// Query an approximate visual shape (circle/box) for the given body handle,
    /// based on the collider shape attached to that body.
    pub fn body_visual_shape(&self, handle: RigidBodyHandle) -> Option<BodyVisualShape> {
        for (_collider_handle, collider) in self.world.collider_set.iter() {
            if collider.parent() == Some(handle) {
                let shape = collider.shape();
                if let Some(ball) = shape.as_ball() {
                    return Some(BodyVisualShape::Circle { radius: ball.radius });
                }
                if let Some(cuboid) = shape.as_cuboid() {
                    let he = cuboid.half_extents;
                    return Some(BodyVisualShape::Box { half_extents: [he.x, he.y] });
                }
            }
        }
        None
    }
}

