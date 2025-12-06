use std::env;
use two_dbuddy::train_stand_upright;

fn parse_arg_usize(args: &[String], index: usize, default: usize) -> usize {
    args.get(index)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_arg_string(args: &[String], index: usize, default: &str) -> String {
    args.get(index).cloned().unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Args: [0]=bin, [1]=population, [2]=top_k, [3]=generations, [4]=output_path
    let population = parse_arg_usize(&args, 1, 64);
    let top_k = parse_arg_usize(&args, 2, 8);
    let generations = parse_arg_usize(&args, 3, 50);
    let output_path = parse_arg_string(&args, 4, "best_brain_stand_upright.bin");

    println!(
        "Running GA training: population={}, top_k={}, generations={}, output_path={}",
        population, top_k, generations, output_path
    );

    train_stand_upright(population, top_k, generations, &output_path);
}
