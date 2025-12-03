use two_dbuddy::{Brain, BuddyIO, SimulationWorld};
use std::time::Instant;

const DEFAULT_STEPS: usize = 240;
const PRINT_INTERVAL: usize = 30;

fn main() {
    let steps = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok())
        .unwrap_or(DEFAULT_STEPS);

    println!("Running offline Rapier2D simulation for {steps} steps...");

    let start_time = Instant::now();
    let mut world = SimulationWorld::new();
    for step in 0..steps {
        let mut brain = Brain::new(
        4.0, 5.0, 0.9, 0.9,
    );
        let sense = world.buddy_sense();
        let sense_flat = BuddyIO::flatten_sense(&sense);
        let action_flat = brain.forward(&sense_flat);
        let action = BuddyIO::unflatten_action(&action_flat);
        world.apply_buddy_action(&action);

        world.step();

        if step % PRINT_INTERVAL == 0 || step == steps - 1 {
            if let Some(head) = world.buddy_head_state() {
                println!(
                    "step {:>4}: head=({:+.2}, {:+.2}) vel=({:+.2}, {:+.2})",
                    step + 1,
                    head.position[0],
                    head.position[1],
                    head.velocity[0],
                    head.velocity[1],
                );
            }
        }
    }

    let duration = start_time.elapsed();
    let steps_per_second = steps as f64 / duration.as_secs_f64();
    
    println!("Simulation complete.");
    println!("Total time: {:.2} seconds", duration.as_secs_f64());
    println!("Steps per second: {:.2}", steps_per_second);
}
