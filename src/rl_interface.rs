use rapier2d::prelude::*;
use crate::buddy::Buddy;

/// Hierarchical sense data for a single limb
#[derive(Debug, Clone)]
pub struct LimbSense {
    pub absolute_angle: AngleSense,
    pub angular_velocity: f32,
    pub absolute_velocity: VelocitySense,
}

#[derive(Debug, Clone)]
pub struct AngleSense {
    pub sin: f32,
    pub cos: f32,
}

#[derive(Debug, Clone)]
pub struct VelocitySense {
    pub x: f32,
    pub y: f32,
}

/// Hierarchical sense data for a single joint
#[derive(Debug, Clone)]
pub struct JointSense {
    pub angle: f32,
    pub angular_velocity: f32,
}

/// Complete hierarchical sense data for the buddy
#[derive(Debug, Clone)]
pub struct BuddySense {
    pub torso: LimbSense,
    pub head: LimbSense,
    pub front_arm_upper: LimbSense,
    pub front_arm_lower: LimbSense,
    pub back_arm_upper: LimbSense,
    pub back_arm_lower: LimbSense,
    pub front_leg_upper: LimbSense,
    pub front_leg_lower: LimbSense,
    pub back_leg_upper: LimbSense,
    pub back_leg_lower: LimbSense,
    pub neck_joint: JointSense,
    pub front_shoulder_joint: JointSense,
    pub front_elbow_joint: JointSense,
    pub back_shoulder_joint: JointSense,
    pub back_elbow_joint: JointSense,
    pub front_hip_joint: JointSense,
    pub front_knee_joint: JointSense,
    pub back_hip_joint: JointSense,
    pub back_knee_joint: JointSense,
}

/// Hierarchical action data for the buddy
#[derive(Debug, Clone)]
pub struct BuddyAction {
    pub neck_joint: f32,
    pub front_shoulder_joint: f32,
    pub front_elbow_joint: f32,
    pub back_shoulder_joint: f32,
    pub back_elbow_joint: f32,
    pub front_hip_joint: f32,
    pub front_knee_joint: f32,
    pub back_hip_joint: f32,
    pub back_knee_joint: f32,
}

/// IO interface for reinforcement learning
pub struct BuddyIO;

impl BuddyIO {
    /// Sense the buddy's current state and return hierarchical data
    pub fn sense(
        buddy: &Buddy,
        rigid_body_set: &RigidBodySet,
        _impulse_joint_set: &ImpulseJointSet,
    ) -> BuddySense {
        // Helper to get limb sense data from a handle
        let get_limb_sense = |handle: RigidBodyHandle| -> LimbSense {
            let body = rigid_body_set.get(handle).unwrap();
            let angle = body.rotation().angle();
            let linvel = body.linvel();
            
            LimbSense {
                absolute_angle: AngleSense {
                    sin: angle.sin(),
                    cos: angle.cos(),
                },
                angular_velocity: body.angvel(),
                absolute_velocity: VelocitySense {
                    x: linvel.x,
                    y: linvel.y,
                },
            }
        };
        
        // Helper to get joint sense data
        let get_joint_sense = |body1_handle: RigidBodyHandle, body2_handle: RigidBodyHandle| -> JointSense {
            // Get the actual angle between the two bodies
            let body1 = rigid_body_set.get(body1_handle).unwrap();
            let body2 = rigid_body_set.get(body2_handle).unwrap();
            let angle = body2.rotation().angle() - body1.rotation().angle();
            let angular_velocity = body2.angvel() - body1.angvel();
            
            JointSense {
                angle,
                angular_velocity,
            }
        };
        
        BuddySense {
            torso: get_limb_sense(buddy.torso),
            head: get_limb_sense(buddy.head),
            front_arm_upper: get_limb_sense(buddy.front_arm_upper),
            front_arm_lower: get_limb_sense(buddy.front_arm_lower),
            back_arm_upper: get_limb_sense(buddy.back_arm_upper),
            back_arm_lower: get_limb_sense(buddy.back_arm_lower),
            front_leg_upper: get_limb_sense(buddy.front_leg_upper),
            front_leg_lower: get_limb_sense(buddy.front_leg_lower),
            back_leg_upper: get_limb_sense(buddy.back_leg_upper),
            back_leg_lower: get_limb_sense(buddy.back_leg_lower),
            neck_joint: get_joint_sense(buddy.torso, buddy.head),
            front_shoulder_joint: get_joint_sense(buddy.torso, buddy.front_arm_upper),
            front_elbow_joint: get_joint_sense(buddy.front_arm_upper, buddy.front_arm_lower),
            back_shoulder_joint: get_joint_sense(buddy.torso, buddy.back_arm_upper),
            back_elbow_joint: get_joint_sense(buddy.back_arm_upper, buddy.back_arm_lower),
            front_hip_joint: get_joint_sense(buddy.torso, buddy.front_leg_upper),
            front_knee_joint: get_joint_sense(buddy.front_leg_upper, buddy.front_leg_lower),
            back_hip_joint: get_joint_sense(buddy.torso, buddy.back_leg_upper),
            back_knee_joint: get_joint_sense(buddy.back_leg_upper, buddy.back_leg_lower),
        }
    }
    
    /// Flatten hierarchical sense data to a flat array for RL
    pub fn flatten_sense(sense: &BuddySense) -> Vec<f32> {
        let mut flat = Vec::new();
        
        // Helper to flatten limb sense
        let flatten_limb = |limb: &LimbSense, flat: &mut Vec<f32>| {
            flat.push(limb.absolute_angle.sin);
            flat.push(limb.absolute_angle.cos);
            flat.push(limb.angular_velocity);
            flat.push(limb.absolute_velocity.x);
            flat.push(limb.absolute_velocity.y);
        };
        
        // Helper to flatten joint sense
        let flatten_joint = |joint: &JointSense, flat: &mut Vec<f32>| {
            flat.push(joint.angle);
            flat.push(joint.angular_velocity);
        };
        
        // Flatten all limbs
        flatten_limb(&sense.torso, &mut flat);
        flatten_limb(&sense.head, &mut flat);
        flatten_limb(&sense.front_arm_upper, &mut flat);
        flatten_limb(&sense.front_arm_lower, &mut flat);
        flatten_limb(&sense.back_arm_upper, &mut flat);
        flatten_limb(&sense.back_arm_lower, &mut flat);
        flatten_limb(&sense.front_leg_upper, &mut flat);
        flatten_limb(&sense.front_leg_lower, &mut flat);
        flatten_limb(&sense.back_leg_upper, &mut flat);
        flatten_limb(&sense.back_leg_lower, &mut flat);
        
        // Flatten all joints
        flatten_joint(&sense.neck_joint, &mut flat);
        flatten_joint(&sense.front_shoulder_joint, &mut flat);
        flatten_joint(&sense.front_elbow_joint, &mut flat);
        flatten_joint(&sense.back_shoulder_joint, &mut flat);
        flatten_joint(&sense.back_elbow_joint, &mut flat);
        flatten_joint(&sense.front_hip_joint, &mut flat);
        flatten_joint(&sense.front_knee_joint, &mut flat);
        flatten_joint(&sense.back_hip_joint, &mut flat);
        flatten_joint(&sense.back_knee_joint, &mut flat);
        
        flat
    }
    
    /// Convert flat action array to hierarchical action struct
    pub fn unflatten_action(flat_action: &[f32]) -> BuddyAction {
        assert_eq!(flat_action.len(), 9, "Action array must have 9 elements (one per joint)");
        
        BuddyAction {
            neck_joint: flat_action[0],
            front_shoulder_joint: flat_action[1],
            front_elbow_joint: flat_action[2],
            back_shoulder_joint: flat_action[3],
            back_elbow_joint: flat_action[4],
            front_hip_joint: flat_action[5],
            front_knee_joint: flat_action[6],
            back_hip_joint: flat_action[7],
            back_knee_joint: flat_action[8],
        }
    }
    
    /// Apply action to the buddy by setting joint angular velocities
    pub fn apply_action(
        action: &BuddyAction,
        buddy: &Buddy,
        impulse_joint_set: &mut ImpulseJointSet,
    ) {
        // Helper to set motor velocity for a joint
        let mut set_motor = |joint_handle: ImpulseJointHandle, velocity: f32| {
            if let Some(joint) = impulse_joint_set.get_mut(joint_handle) {
                if let Some(rev) = joint.data.as_revolute_mut() {
                    rev.data.set_motor_velocity(JointAxis::AngX, velocity, 1.0);
                }
            }
        };
        
        let factor = 1.0;
        let max_velocity = 20.0;
        set_motor(buddy.neck_joint, (action.neck_joint * factor).clamp(-max_velocity, max_velocity));
        set_motor(buddy.front_shoulder_joint, (action.front_shoulder_joint * factor).clamp(-max_velocity, max_velocity));
        set_motor(buddy.front_elbow_joint, (action.front_elbow_joint * factor).clamp(-max_velocity, max_velocity));
        set_motor(buddy.back_shoulder_joint, (action.back_shoulder_joint * factor).clamp(-max_velocity, max_velocity));
        set_motor(buddy.back_elbow_joint, (action.back_elbow_joint * factor).clamp(-max_velocity, max_velocity));
        set_motor(buddy.front_hip_joint, (action.front_hip_joint * factor).clamp(-max_velocity, max_velocity));
        set_motor(buddy.front_knee_joint, (action.front_knee_joint * factor).clamp(-max_velocity, max_velocity));
        set_motor(buddy.back_hip_joint, (action.back_hip_joint * factor).clamp(-max_velocity, max_velocity));
        set_motor(buddy.back_knee_joint, (action.back_knee_joint * factor).clamp(-max_velocity, max_velocity));
    }
    
    /// Get the size of the flattened sense array
    pub fn sense_size() -> usize {
        // 10 limbs * 5 values each + 9 joints * 2 values each
        10 * 5 + 9 * 2
    }
    
    /// Get the size of the action array
    pub fn action_size() -> usize {
        9 // One angular velocity per joint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sense_size() {
        assert_eq!(BuddyIO::sense_size(), 68);
    }
    
    #[test]
    fn test_action_size() {
        assert_eq!(BuddyIO::action_size(), 9);
    }
    
    #[test]
    fn test_unflatten_action() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let action = BuddyIO::unflatten_action(&flat);
        
        assert_eq!(action.neck_joint, 1.0);
        assert_eq!(action.front_shoulder_joint, 2.0);
        assert_eq!(action.front_elbow_joint, 3.0);
        assert_eq!(action.back_shoulder_joint, 4.0);
        assert_eq!(action.back_elbow_joint, 5.0);
        assert_eq!(action.front_hip_joint, 6.0);
        assert_eq!(action.front_knee_joint, 7.0);
        assert_eq!(action.back_hip_joint, 8.0);
        assert_eq!(action.back_knee_joint, 9.0);
    }
}
