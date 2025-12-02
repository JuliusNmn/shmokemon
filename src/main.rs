use two_dbuddy::SimulationWorld;

const DEFAULT_STEPS: usize = 240;
const PRINT_INTERVAL: usize = 30;

fn main() {
    let steps = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok())
        .unwrap_or(DEFAULT_STEPS);

    println!("Running offline Rapier2D simulation for {steps} steps...");

    let mut world = SimulationWorld::new();

    for step in 0..steps {
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

    println!("Simulation complete.");
}
