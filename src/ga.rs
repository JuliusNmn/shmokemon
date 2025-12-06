use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use crate::{Brain, SimulationWorld};

const FRAMES_PER_ROLLOUT: usize = 200;
const ANGLE_PENALTY: f32 = 0.5;
const VEL_PENALTY: f32 = 0.1;
const ANGVEL_PENALTY: f32 = 0.1;
const MUTATION_RATE: f32 = 0.2;
const MUTATION_STRENGTH: f32 = 0.2;

fn evaluate_brain_upright(brain: &Brain) -> f32 {
    let mut world = SimulationWorld::new();
    let mut total_score = 0.0f32;

    for _ in 0..FRAMES_PER_ROLLOUT {
        let sense = world.buddy_sense_flat();
        let action = brain.forward(&sense);
        world.apply_buddy_action_flat(&action);
        world.step();

        let head_state = world.buddy_head_state();
        let torso_state = world.buddy_torso_state();

        if let (Some(head), Some(torso)) = (head_state, torso_state) {
            let head_y = head.position[1];
            let torso_angle = torso.rotation;
            let torso_vx = torso.velocity[0];
            let torso_vy = torso.velocity[1];

            let torso_angvel = world
                .rigid_body_set
                .get(world.buddy.torso)
                .map(|b| b.angvel() as f32)
                .unwrap_or(0.0);

            let angle_term = ANGLE_PENALTY * torso_angle.abs();
            let vel_term = VEL_PENALTY * (torso_vx * torso_vx + torso_vy * torso_vy);
            let angvel_term = ANGVEL_PENALTY * (torso_angvel * torso_angvel);

            let frame_score = head_y - angle_term - vel_term - angvel_term;
            total_score += frame_score;
        }
    }

    total_score
}

pub fn train_stand_upright(
    population_size: usize,
    top_k: usize,
    generations: usize,
    output_path: &str,
    seed_brain_path: Option<&str>,
) {
    assert!(population_size > 0);
    assert!(top_k > 0 && top_k <= population_size);

    let mut rng = rand::thread_rng();
    let mut population: Vec<Brain> = if let Some(path) = seed_brain_path {
        match Brain::from_file(path) {
            Ok(seed_brain) => {
                let mut pop = Vec::with_capacity(population_size);
                pop.push(seed_brain);
                while pop.len() < population_size {
                    let parent_idx = rng.gen_range(0..pop.len());
                    let parent = &pop[parent_idx];
                    pop.push(parent.mutated(MUTATION_RATE, MUTATION_STRENGTH));
                }
                pop
            }
            Err(err) => {
                eprintln!("failed to load seed brain from {}: {} - falling back to random init", path, err);
                (0..population_size)
                    .map(|_| Brain::new(3.0, 4.0, 0.95, 0.95))
                    .collect()
            }
        }
    } else {
        (0..population_size)
            .map(|_| Brain::new(3.0, 4.0, 0.95, 0.95))
            .collect()
    };

    let mut best_overall_score = f32::NEG_INFINITY;
    let mut cumulative_steps: usize = 0;

    for gen in 0..generations {
        let gen_start = Instant::now();

        let mut scored: Vec<(f32, Brain)> = population
            .into_par_iter()
            .map(|brain| {
                let score = evaluate_brain_upright(&brain);
                (score, brain)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let best_score = scored.first().map(|(s, _)| *s).unwrap_or(f32::NEG_INFINITY);

        // Simulation statistics for this generation
        let steps_this_gen: usize = population_size * FRAMES_PER_ROLLOUT;
        cumulative_steps += steps_this_gen;
        let elapsed = gen_start.elapsed();
        let secs = elapsed.as_secs_f64().max(1e-9);
        let steps_per_sec = (steps_this_gen as f64) / secs;

        println!(
            "generation {gen}: best_score = {best_score}, steps_this_gen = {steps_this_gen}, cumulative_steps = {cumulative_steps}, steps_per_sec = {steps_per_sec:.2}"
        );

        if best_score > best_overall_score {
            best_overall_score = best_score;
            if let Some((_, best_brain)) = scored.first() {
                let _ = best_brain.save_to_file(output_path);
            }
        }

        let survivors: Vec<Brain> = scored
            .into_iter()
            .take(top_k)
            .map(|(_, brain)| brain)
            .collect();

        let survivor_count = survivors.len();
        let mut new_population: Vec<Brain> = Vec::with_capacity(population_size);

        // Elitism: keep each survivor and add a mutated child
        for survivor in survivors {
            let child = survivor.mutated(MUTATION_RATE, MUTATION_STRENGTH);
            new_population.push(survivor);
            new_population.push(child);
            new_population.push(Brain::new(3.0, 4.0, 0.95, 0.95));
        }

        // Fill the rest of the population with mutated copies of random elites
        while new_population.len() < population_size {
            let idx = rng.gen_range(0..survivor_count);
            let parent = &new_population[idx];
            new_population.push(parent.mutated(MUTATION_RATE, MUTATION_STRENGTH));
        }

        population = new_population;
    }
}
