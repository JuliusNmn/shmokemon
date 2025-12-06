use std::env;
use two_dbuddy::train_stand_upright;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Defaults
    let mut population: usize = 64;
    let mut top_k: usize = 8;
    let mut generations: usize = 50;
    let mut output_path: String = "best_brain_stand_upright.bin".to_string();
    let mut seed_brain_path: Option<String> = None;
    let mut use_mps: bool = false;

    // Simple flag parser: supports
    // --population <usize>
    // --top-k <usize>
    // --generations <usize>
    // --output <path>
    // --seed <path>
    // --mps
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--population" => {
                if let Some(val) = args.get(i + 1) {
                    if let Ok(parsed) = val.parse::<usize>() {
                        population = parsed;
                    }
                    i += 2;
                } else {
                    break;
                }
            }
            "--top-k" => {
                if let Some(val) = args.get(i + 1) {
                    if let Ok(parsed) = val.parse::<usize>() {
                        top_k = parsed;
                    }
                    i += 2;
                } else {
                    break;
                }
            }
            "--generations" => {
                if let Some(val) = args.get(i + 1) {
                    if let Ok(parsed) = val.parse::<usize>() {
                        generations = parsed;
                    }
                    i += 2;
                } else {
                    break;
                }
            }
            "--output" => {
                if let Some(val) = args.get(i + 1) {
                    output_path = val.clone();
                    i += 2;
                } else {
                    break;
                }
            }
            "--seed" => {
                if let Some(val) = args.get(i + 1) {
                    seed_brain_path = Some(val.clone());
                    i += 2;
                } else {
                    break;
                }
            }
            "--mps" => {
                use_mps = true;
                i += 1;
            }
            _ => {
                // Unknown argument, skip it
                i += 1;
            }
        }
    }

    // Args: [0]=bin, [1]=population, [2]=top_k, [3]=generations, [4]=output_path, [5]=optional seed_brain_path

    println!(
        "Running GA training: population={}, top_k={}, generations={}, output_path={}, seed_brain_path={}, use_mps={}",
        population,
        top_k,
        generations,
        output_path,
        seed_brain_path.as_deref().unwrap_or("<none>"),
        use_mps,
    );

    if use_mps {
        // This is picked up in brain.rs to select the MPS LibTorch device
        // instead of the default CPU device.
        env::set_var("TWO_DBUDDY_USE_MPS", "1");
    }

    train_stand_upright(
        population,
        top_k,
        generations,
        &output_path,
        seed_brain_path.as_deref(),
    );
}
