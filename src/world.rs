use rapier2d::{geometry::DefaultBroadPhase, prelude::*};
use std::collections::VecDeque;
use crate::buddy::Buddy;
use crate::physics::{
    BUDDY_SPAWN_HEIGHT, FIXED_TIME_STEP, FLOOR_HALF_EXTENTS, FLOOR_HEIGHT, 
    GROUP_BUDDY, GROUP_WORLD, RigidBodySnapshot,
};
use crate::rl_interface::{BuddyIO, BuddySense, BuddyAction};

pub struct SimulationWorld {
    pipeline: PhysicsPipeline,
    gravity: Vector<Real>,
    integration_parameters: IntegrationParameters,
    pub(crate) island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    pub(crate) rigid_body_set: RigidBodySet,
    pub(crate) collider_set: ColliderSet,
    pub(crate) impulse_joint_set: ImpulseJointSet,
    pub(crate) multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    pub(crate) buddy: Buddy,
    time: Real,
    torque_history: VecDeque<Real>,
}

impl SimulationWorld {
    pub fn new() -> Self {
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = FIXED_TIME_STEP;

        let gravity = vector![0.0, -9.81];

        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();
        let mut impulse_joint_set = ImpulseJointSet::new();

        // Static floor.
        let floor_body = RigidBodyBuilder::fixed()
            .translation(vector![0.0, FLOOR_HEIGHT])
            .build();
        let floor_handle = rigid_body_set.insert(floor_body);
        let floor_collider = ColliderBuilder::cuboid(FLOOR_HALF_EXTENTS[0], FLOOR_HALF_EXTENTS[1])
            .restitution(0.0)
            .friction(0.9)
            .collision_groups(InteractionGroups::new(
                Group::from_bits_truncate(GROUP_WORLD),
                Group::from_bits_truncate(GROUP_BUDDY),
            ))
            .build();
        collider_set.insert_with_parent(floor_collider, floor_handle, &mut rigid_body_set);

        // Buddy rig.
        let buddy = Buddy::spawn(
            vector![0.0, BUDDY_SPAWN_HEIGHT],
            &mut rigid_body_set,
            &mut collider_set,
            &mut impulse_joint_set,
        );

        Self {
            pipeline: PhysicsPipeline::new(),
            gravity,
            integration_parameters,
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set,
            collider_set,
            impulse_joint_set,
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            buddy,
            time: 0.0,
            torque_history: VecDeque::with_capacity(1000),
        }
    }

    pub fn step(&mut self) {
        let physics_hooks = ();
        let event_handler = ();
        self.pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &physics_hooks,
            &event_handler,
        );
        self.time += self.integration_parameters.dt;
        
        // Calculate total applied torque from all buddy joints
        let mut total_torque = 0.0;
        for joint_handle in self.buddy.joints() {
            if let Some(joint) = self.impulse_joint_set.get(joint_handle) {
                if let Some(rev) = joint.data.as_revolute() {
                    if let Some(motor) = rev.data.motor(JointAxis::AngX) {
                        let applied_impulse = motor.impulse;
                        let applied_torque = applied_impulse / self.integration_parameters.dt;
                        total_torque += applied_torque.abs();
                    }
                }
            }
        }
        
        // Store torque in history buffer (keep last 1000 values)
        self.torque_history.push_back(total_torque);
        if self.torque_history.len() > 1000 {
            self.torque_history.pop_front();
        }
        
    }

    pub fn buddy(&self) -> &Buddy {
        &self.buddy
    }
    
    pub fn time(&self) -> Real {
        self.time
    }

    pub fn body_snapshot(&self, handle: RigidBodyHandle) -> Option<RigidBodySnapshot> {
        self.rigid_body_set
            .get(handle)
            .map(|body| RigidBodySnapshot {
                position: [body.translation().x as f32, body.translation().y as f32],
                velocity: [body.linvel().x as f32, body.linvel().y as f32],
                rotation: body.rotation().angle() as f32,
            })
    }

    pub fn buddy_torso_state(&self) -> Option<RigidBodySnapshot> {
        self.body_snapshot(self.buddy.torso)
    }

    pub fn buddy_head_state(&self) -> Option<RigidBodySnapshot> {
        self.body_snapshot(self.buddy.head)
    }
    
    /// Get the buddy's current sense data (hierarchical)
    pub fn buddy_sense(&self) -> BuddySense {
        BuddyIO::sense(&self.buddy, &self.rigid_body_set, &self.impulse_joint_set)
    }
    
    /// Get the buddy's current sense data as a flat array for RL
    pub fn buddy_sense_flat(&self) -> Vec<f32> {
        let sense = self.buddy_sense();
        BuddyIO::flatten_sense(&sense)
    }
    
    /// Apply an action to the buddy from a hierarchical action struct
    pub fn apply_buddy_action(&mut self, action: &BuddyAction) {
        BuddyIO::apply_action(action, &self.buddy, &mut self.impulse_joint_set);
    }
    
    /// Apply an action to the buddy from a flat array
    pub fn apply_buddy_action_flat(&mut self, flat_action: &[f32]) {
        let action = BuddyIO::unflatten_action(flat_action);
        self.apply_buddy_action(&action);
    }
    
    /// Get the torque history buffer (last 1000 torque values)
    pub fn torque_history(&self) -> &VecDeque<Real> {
        &self.torque_history
    }
}
