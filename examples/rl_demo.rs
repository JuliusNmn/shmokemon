use two_dbuddy::{SimulationWorld, BuddyIO};

fn main() {
    let mut world = SimulationWorld::new();
    
    println!("Buddy RL Interface Demo");
    println!("=======================\n");
    
    // Get sense dimensions
    println!("Sense array size: {}", BuddyIO::sense_size());
    println!("Action array size: {}\n", BuddyIO::action_size());
    
    // Run a few simulation steps
    for step in 0..5 {
        println!("Step {}:", step);
        
        // Get hierarchical sense data
        let sense = world.buddy_sense();
        println!("  Torso angle: sin={:.3}, cos={:.3}", 
            sense.torso.absolute_angle.sin, 
            sense.torso.absolute_angle.cos);
        println!("  Torso velocity: x={:.3}, y={:.3}", 
            sense.torso.absolute_velocity.x, 
            sense.torso.absolute_velocity.y);
        
        // Get flat sense array
        let flat_sense = world.buddy_sense_flat();
        println!("  Flat sense array length: {}", flat_sense.len());
        println!("  First 5 values: {:?}", &flat_sense[0..5]);
        
        // Create a simple action (all joints move slowly)
        let action = vec![0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0];
        world.apply_buddy_action_flat(&action);
        
        // Step the simulation
        world.step();
        
        println!();
    }
    
    println!("\nExample of hierarchical sense structure:");
    let sense = world.buddy_sense();
    println!("Front arm upper limb:");
    println!("  absolute_angle: {{ sin: {:.3}, cos: {:.3} }}", 
        sense.front_arm_upper.absolute_angle.sin,
        sense.front_arm_upper.absolute_angle.cos);
    println!("  angular_velocity: {:.3}", sense.front_arm_upper.angular_velocity);
    println!("  absolute_velocity: {{ x: {:.3}, y: {:.3} }}", 
        sense.front_arm_upper.absolute_velocity.x,
        sense.front_arm_upper.absolute_velocity.y);
    
    println!("\nFront shoulder joint:");
    println!("  angle: {:.3}", sense.front_shoulder_joint.angle);
    println!("  angular_velocity: {:.3}", sense.front_shoulder_joint.angular_velocity);
}
