use rapier2d::prelude::*;
use crate::physics::{GROUP_BUDDY, GROUP_WORLD};

// Buddy dimensions
pub const HEAD_RADIUS: Real = 0.22;
pub const NECK_LENGTH: Real = 0.05;
pub const TORSO_WIDTH: Real = 0.35;
pub const TORSO_HEIGHT: Real = 0.9;
pub const ARM_THICKNESS: Real = 0.08;
pub const SHOULDER_CLEARANCE: Real = 0.05;
pub const UPPER_ARM_LENGTH: Real = 0.45;
pub const LOWER_ARM_LENGTH: Real = 0.4;
pub const LEG_THICKNESS: Real = 0.1;
pub const HIP_CLEARANCE: Real = 0.03;
pub const UPPER_LEG_LENGTH: Real = 0.55;
pub const LOWER_LEG_LENGTH: Real = 0.5;

// Buddy physics properties
pub const BUDDY_LINEAR_DAMPING: Real = 0.05;
pub const BUDDY_ANGULAR_DAMPING: Real = 0.05;
pub const BUDDY_DENSITY: Real = 1.0;
pub const JOINT_STIFFNESS: Real = 20.0;
pub const JOINT_DAMPING: Real = 2.0;
pub const JOINT_MAX_FORCE: Real = 50.0;

#[derive(Debug, Clone)]
pub enum BuddyPartShape {
    Circle { radius: Real },
    Box { half_extents: [Real; 2] },
}

#[derive(Debug, Clone)]
pub struct BuddyPart {
    pub name: String,
    pub handle: RigidBodyHandle,
    pub shape: BuddyPartShape,
}

#[derive(Debug)]
pub struct Buddy {
    // Limb handles
    pub torso: RigidBodyHandle,
    pub head: RigidBodyHandle,
    pub front_arm_upper: RigidBodyHandle,
    pub front_arm_lower: RigidBodyHandle,
    pub back_arm_upper: RigidBodyHandle,
    pub back_arm_lower: RigidBodyHandle,
    pub front_leg_upper: RigidBodyHandle,
    pub front_leg_lower: RigidBodyHandle,
    pub back_leg_upper: RigidBodyHandle,
    pub back_leg_lower: RigidBodyHandle,
    
    // Joint handles
    pub neck_joint: ImpulseJointHandle,
    pub front_shoulder_joint: ImpulseJointHandle,
    pub front_elbow_joint: ImpulseJointHandle,
    pub back_shoulder_joint: ImpulseJointHandle,
    pub back_elbow_joint: ImpulseJointHandle,
    pub front_hip_joint: ImpulseJointHandle,
    pub front_knee_joint: ImpulseJointHandle,
    pub back_hip_joint: ImpulseJointHandle,
    pub back_knee_joint: ImpulseJointHandle,
    
    // Legacy: keep for visualization
    parts: Vec<BuddyPart>,
}

impl Buddy {
    pub fn spawn(
        origin: Vector<Real>,
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        impulse_joint_set: &mut ImpulseJointSet,
    ) -> Self {
        let mut parts = Vec::new();

        let torso_half = [TORSO_WIDTH / 2.0, TORSO_HEIGHT / 2.0];
        let torso_handle = Self::insert_box_part(
            "torso",
            origin,
            torso_half,
            rigid_body_set,
            collider_set,
            &mut parts,
        );

        let head_center = origin + vector![0.0, TORSO_HEIGHT / 2.0 + NECK_LENGTH + HEAD_RADIUS];
        let head_handle = Self::insert_ball_part(
            "head",
            head_center,
            HEAD_RADIUS,
            rigid_body_set,
            collider_set,
            &mut parts,
        );
        let neck_joint = Self::motor_joint(
            impulse_joint_set,
            torso_handle,
            head_handle,
            point![0.0, TORSO_HEIGHT / 2.0],
            point![0.0, -HEAD_RADIUS],
        );

        let shoulder_world_y = origin.y + TORSO_HEIGHT / 2.0 - SHOULDER_CLEARANCE;
        let (front_arm_upper, front_arm_lower, front_shoulder_joint, front_elbow_joint) = Self::spawn_arm(
            "front_arm",
            origin,
            torso_handle,
            shoulder_world_y,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
            &mut parts,
        );
        let (back_arm_upper, back_arm_lower, back_shoulder_joint, back_elbow_joint) = Self::spawn_arm(
            "back_arm",
            origin,
            torso_handle,
            shoulder_world_y,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
            &mut parts,
        );

        let (front_leg_upper, front_leg_lower, front_hip_joint, front_knee_joint) = Self::spawn_leg(
            "front_leg",
            origin,
            torso_handle,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
            &mut parts,
        );
        let (back_leg_upper, back_leg_lower, back_hip_joint, back_knee_joint) = Self::spawn_leg(
            "back_leg",
            origin,
            torso_handle,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
            &mut parts,
        );

        Self {
            torso: torso_handle,
            head: head_handle,
            front_arm_upper,
            front_arm_lower,
            back_arm_upper,
            back_arm_lower,
            front_leg_upper,
            front_leg_lower,
            back_leg_upper,
            back_leg_lower,
            neck_joint,
            front_shoulder_joint,
            front_elbow_joint,
            back_shoulder_joint,
            back_elbow_joint,
            front_hip_joint,
            front_knee_joint,
            back_hip_joint,
            back_knee_joint,
            parts,
        }
    }

    pub fn parts(&self) -> &[BuddyPart] {
        &self.parts
    }

    pub fn joints(&self) -> [ImpulseJointHandle; 9] {
        [
            self.neck_joint,
            self.front_shoulder_joint,
            self.front_elbow_joint,
            self.back_shoulder_joint,
            self.back_elbow_joint,
            self.front_hip_joint,
            self.front_knee_joint,
            self.back_hip_joint,
            self.back_knee_joint,
        ]
    }

    fn spawn_arm(
        label: &str,
        torso_center: Vector<Real>,
        torso_handle: RigidBodyHandle,
        shoulder_world_y: Real,
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        impulse_joint_set: &mut ImpulseJointSet,
        parts: &mut Vec<BuddyPart>,
    ) -> (RigidBodyHandle, RigidBodyHandle, ImpulseJointHandle, ImpulseJointHandle) {
        let shoulder_x = torso_center.x + TORSO_WIDTH / 2.0 + ARM_THICKNESS / 2.0;
        let upper_half = [ARM_THICKNESS / 2.0, UPPER_ARM_LENGTH / 2.0];
        let upper_center = vector![shoulder_x, shoulder_world_y - UPPER_ARM_LENGTH / 2.0];
        let upper_handle = Self::insert_box_part(
            format!("{label}_upper"),
            upper_center,
            upper_half,
            rigid_body_set,
            collider_set,
            parts,
        );

        let lower_half = [ARM_THICKNESS / 2.0, LOWER_ARM_LENGTH / 2.0];
        let lower_center = vector![
            shoulder_x,
            shoulder_world_y - UPPER_ARM_LENGTH - LOWER_ARM_LENGTH / 2.0
        ];
        let lower_handle = Self::insert_box_part(
            format!("{label}_lower"),
            lower_center,
            lower_half,
            rigid_body_set,
            collider_set,
            parts,
        );

        let shoulder_joint = Self::motor_joint(
            impulse_joint_set,
            torso_handle,
            upper_handle,
            point![0.0, TORSO_HEIGHT / 2.0 - SHOULDER_CLEARANCE],
            point![0.0, UPPER_ARM_LENGTH / 2.0],
        );
        let elbow_joint = Self::motor_joint(
            impulse_joint_set,
            upper_handle,
            lower_handle,
            point![0.0, -UPPER_ARM_LENGTH / 2.0],
            point![0.0, LOWER_ARM_LENGTH / 2.0],
        );
        (upper_handle, lower_handle, shoulder_joint, elbow_joint)
    }

    fn spawn_leg(
        label: &str,
        torso_center: Vector<Real>,
        torso_handle: RigidBodyHandle,
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        impulse_joint_set: &mut ImpulseJointSet,
        parts: &mut Vec<BuddyPart>,
    ) -> (RigidBodyHandle, RigidBodyHandle, ImpulseJointHandle, ImpulseJointHandle) {
        let hip_world_y = torso_center.y - TORSO_HEIGHT / 2.0 + HIP_CLEARANCE;
        let hip_x = torso_center.x;
        let upper_half = [LEG_THICKNESS / 2.0, UPPER_LEG_LENGTH / 2.0];
        let upper_center = vector![hip_x, hip_world_y - UPPER_LEG_LENGTH / 2.0];
        let upper_handle = Self::insert_box_part(
            format!("{label}_upper"),
            upper_center,
            upper_half,
            rigid_body_set,
            collider_set,
            parts,
        );

        let lower_half = [LEG_THICKNESS / 2.0, LOWER_LEG_LENGTH / 2.0];
        let lower_center = vector![
            hip_x,
            hip_world_y - UPPER_LEG_LENGTH - LOWER_LEG_LENGTH / 2.0
        ];
        let lower_handle = Self::insert_box_part(
            format!("{label}_lower"),
            lower_center,
            lower_half,
            rigid_body_set,
            collider_set,
            parts,
        );

        let hip_joint = Self::motor_joint(
            impulse_joint_set,
            torso_handle,
            upper_handle,
            point![0.0, -TORSO_HEIGHT / 2.0 + HIP_CLEARANCE],
            point![0.0, UPPER_LEG_LENGTH / 2.0],
        );
        let knee_joint = Self::motor_joint(
            impulse_joint_set,
            upper_handle,
            lower_handle,
            point![0.0, -UPPER_LEG_LENGTH / 2.0],
            point![0.0, LOWER_LEG_LENGTH / 2.0],
        );
        (upper_handle, lower_handle, hip_joint, knee_joint)
    }

    fn insert_box_part(
        name: impl Into<String>,
        center: Vector<Real>,
        half_extents: [Real; 2],
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        parts: &mut Vec<BuddyPart>,
    ) -> RigidBodyHandle {
        let body = RigidBodyBuilder::dynamic()
            .translation(center)
            .linear_damping(BUDDY_LINEAR_DAMPING)
            .angular_damping(BUDDY_ANGULAR_DAMPING)
            .build();
        let handle = rigid_body_set.insert(body);
        let collider = ColliderBuilder::cuboid(half_extents[0], half_extents[1])
            .density(BUDDY_DENSITY)
            .friction(0.9)
            .collision_groups(InteractionGroups::new(
                Group::from_bits_truncate(GROUP_BUDDY),
                Group::from_bits_truncate(GROUP_WORLD),
            ))
            .build();
        collider_set.insert_with_parent(collider, handle, rigid_body_set);
        parts.push(BuddyPart {
            name: name.into(),
            handle,
            shape: BuddyPartShape::Box { half_extents },
        });
        handle
    }

    fn insert_ball_part(
        name: impl Into<String>,
        center: Vector<Real>,
        radius: Real,
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        parts: &mut Vec<BuddyPart>,
    ) -> RigidBodyHandle {
        let body = RigidBodyBuilder::dynamic()
            .translation(center)
            .linear_damping(BUDDY_LINEAR_DAMPING)
            .angular_damping(BUDDY_ANGULAR_DAMPING)
            .build();
        let handle = rigid_body_set.insert(body);
        let collider = ColliderBuilder::ball(radius)
            .density(BUDDY_DENSITY)
            .friction(0.9)
            .restitution(0.1)
            .collision_groups(InteractionGroups::new(
                Group::from_bits_truncate(GROUP_BUDDY),
                Group::from_bits_truncate(GROUP_WORLD),
            ))
            .build();
        collider_set.insert_with_parent(collider, handle, rigid_body_set);
        parts.push(BuddyPart {
            name: name.into(),
            handle,
            shape: BuddyPartShape::Circle { radius },
        });
        handle
    }

    fn motor_joint(
        impulse_joint_set: &mut ImpulseJointSet,
        parent: RigidBodyHandle,
        child: RigidBodyHandle,
        anchor_parent: Point<Real>,
        anchor_child: Point<Real>,
    ) -> ImpulseJointHandle {
        let joint = RevoluteJointBuilder::new()
            .local_anchor1(anchor_parent)
            .local_anchor2(anchor_child)
            .motor_model(MotorModel::ForceBased)
            .motor(0.0, 0.0, JOINT_STIFFNESS, JOINT_DAMPING)
            .motor_max_force(JOINT_MAX_FORCE)
            .limits([-std::f32::consts::PI * 0.75, std::f32::consts::PI * 0.75])
            .build();
        impulse_joint_set.insert(parent, child, joint, true)
    }
}
