use macroquad::prelude::*;
use two_dbuddy::{
    BuddyPartShape, SimulationWorld, FLOOR_HALF_EXTENTS, FLOOR_HEIGHT, PIXELS_PER_METER,
    BuddyIO, BuddyAction, Brain, SPARSITY_INPUT_HIDDEN, SPARSITY_HIDDEN_OUTPUT,
};

/// Slider state for brain initialization parameters
struct BrainInitParams {
    ih_gain: f32,
    ho_gain: f32,
    ih_sparsity: f32,
    ho_sparsity: f32,
    /// Which slider is currently being dragged (if any)
    active_slider: Option<usize>,
}

impl Default for BrainInitParams {
    fn default() -> Self {
        Self {
            ih_gain: 4.0,
            ho_gain: 5.0,
            ih_sparsity: SPARSITY_INPUT_HIDDEN,
            ho_sparsity: SPARSITY_HIDDEN_OUTPUT,
            active_slider: None,
        }
    }
}

const FRAMES_PER_STEP: usize = 2;

#[macroquad::main("Rapier2D + Macroquad playground")]
async fn main() {
    request_new_screen_size(1400.0, 1000.0);
    
    let step_limit = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok());

    let mut init_params = BrainInitParams::default();
    let mut world = SimulationWorld::new();
    let mut brain = Brain::new(
        init_params.ih_gain as f64,
        init_params.ho_gain as f64,
        init_params.ih_sparsity,
        init_params.ho_sparsity,
    );
    let mut steps = 0usize;
    let mut frame = 0usize;
    loop {
        if let Some(limit) = step_limit {
            if steps >= limit {
                break;
            }
        }
        if (frame % FRAMES_PER_STEP) == 0 {
            // Get sense data and use brain to generate action
            let sense = world.buddy_sense();
            let sense_flat = BuddyIO::flatten_sense(&sense);
            let action_flat = brain.forward(&sense_flat);
            let action = BuddyIO::unflatten_action(&action_flat);
            world.apply_buddy_action(&action);
            
            world.step();
            steps += 1;
        }
        
        // Get current sense and action for visualization
        let sense = world.buddy_sense();
        let sense_flat = BuddyIO::flatten_sense(&sense);
        let (action_flat, hidden_activations) = brain.forward_with_activations(&sense_flat);
        let action = BuddyIO::unflatten_action(&action_flat);
        
        // Check for button clicks
        if draw_world(&world, steps, step_limit, &sense_flat, &action, &brain, &hidden_activations) {
            // Restart button clicked - reset simulation with same brain
            world = SimulationWorld::new();
            steps = 0;
            frame = 0;
        }
        
        // Draw sliders above reset buttons
        draw_init_param_sliders(&mut init_params);
        
        if draw_reset_button() {
            // Reset brain button clicked - new brain with current slider values
            brain = Brain::new(
                init_params.ih_gain as f64,
                init_params.ho_gain as f64,
                init_params.ih_sparsity,
                init_params.ho_sparsity,
            );
            world = SimulationWorld::new();
            steps = 0;
            frame = 0;
        }
        
        frame += 1;

        next_frame().await;
    }
}

fn draw_world(
    world: &SimulationWorld, 
    steps: usize, 
    limit: Option<usize>,
    sense_flat: &[f32],
    action: &BuddyAction,
    brain: &Brain,
    hidden_activations: &[f32],
) -> bool {
    clear_background(Color::from_rgba(12, 16, 24, 255));
    draw_floor();
    draw_buddy(world);

    let status = match limit {
        Some(max) => format!("step {steps} / {max}"),
        None => format!("step {steps}"),
    };

    draw_text(&status, 20.0, 32.0, 28.0, WHITE);
    draw_text(&format!("fps {}", get_fps()), 20.0, 60.0, 24.0, LIGHTGRAY);
    
    // Draw sense data visualization
    draw_sense_visualization(sense_flat);
    
    // Draw action data visualization
    draw_action_visualization(action);
    
    // Draw neural network visualizations
    draw_network_visualizations(brain, hidden_activations);
    
    draw_text(
        "press Ctrl+C to exit",
        20.0,
        screen_height() - 20.0,
        20.0,
        GRAY,
    );
    
    // Draw restart button and return if clicked
    draw_restart_button()
}

fn draw_floor() {
    let width = (FLOOR_HALF_EXTENTS[0] as f32 * 2.0) * PIXELS_PER_METER;
    let height = (FLOOR_HALF_EXTENTS[1] as f32 * 2.0) * PIXELS_PER_METER;
    let x = screen_width() * 0.5 - width * 0.5;
    let y_center = world_to_screen_y(FLOOR_HEIGHT as f32);
    let y = y_center - height * 0.5;

    draw_rectangle(x, y, width, height, DARKGRAY);
}

fn draw_buddy(world: &SimulationWorld) {
    for part in world.buddy().parts() {
        if let Some(snapshot) = world.body_snapshot(part.handle) {
            match part.shape {
                BuddyPartShape::Circle { radius } => draw_circle_part(&snapshot, radius as f32),
                BuddyPartShape::Box { half_extents } => {
                    draw_box_part(&snapshot, [half_extents[0] as f32, half_extents[1] as f32])
                }
            }
        }
    }
}

fn draw_circle_part(snapshot: &two_dbuddy::RigidBodySnapshot, radius: f32) {
    let center = world_to_screen(snapshot.position);
    draw_circle(center.x, center.y, radius * PIXELS_PER_METER, YELLOW);
}

fn draw_box_part(snapshot: &two_dbuddy::RigidBodySnapshot, half_extents: [f32; 2]) {
    let half_extents_vec = Vec2::new(half_extents[0], half_extents[1]);
    let local_corners = [
        vec2(-half_extents_vec.x, -half_extents_vec.y),
        vec2(half_extents_vec.x, -half_extents_vec.y),
        vec2(half_extents_vec.x, half_extents_vec.y),
        vec2(-half_extents_vec.x, half_extents_vec.y),
    ];

    let rotation = Mat2::from_angle(snapshot.rotation);
    let translation = vec2(snapshot.position[0], snapshot.position[1]);

    let mut screen_points = [Vec2::ZERO; 4];
    for (idx, corner) in local_corners.iter().enumerate() {
        let rotated = rotation * *corner;
        let world = translation + rotated;
        screen_points[idx] = world_to_screen([world.x, world.y]);
    }

    draw_triangle(screen_points[0], screen_points[1], screen_points[2], ORANGE);
    draw_triangle(screen_points[0], screen_points[2], screen_points[3], ORANGE);
}

fn world_to_screen(position: [f32; 2]) -> Vec2 {
    vec2(
        screen_width() * 0.5 + position[0] * PIXELS_PER_METER,
        screen_height() * 0.5 - position[1] * PIXELS_PER_METER,
    )
}

fn world_to_screen_y(y: f32) -> f32 {
    screen_height() * 0.5 - y * PIXELS_PER_METER
}

/// Visualizes sense data in a compact format with vertical bars for each limb.
/// 
/// Sense data structure (68 values total):
/// - Limbs (10 limbs × 5 values = 50 values):
///   Each limb: [sin(angle), cos(angle), angular_velocity, velocity_x, velocity_y]
fn draw_sense_visualization(sense_flat: &[f32]) {
    let panel_x = screen_width() - 280.0;
    let panel_y = 20.0;
    let panel_width = 260.0;
    let panel_height = 400.0;
    
    // Draw background panel
    draw_rectangle(panel_x, panel_y, panel_width, panel_height, Color::from_rgba(0, 0, 0, 180));
    
    // Title
    draw_text("LIMB SENSE DATA", panel_x + 10.0, panel_y + 20.0, 18.0, YELLOW);
    
    // Draw slanted labels at the top
    let label_y = panel_y + 45.0;
    let label_start_x = panel_x + 70.0;
    let label_spacing = 30.0;
    
    let labels = ["Sin", "Cos", "AngVel", "Vel X", "Vel Y"];
    let colors = [
        Color::from_rgba(100, 150, 255, 255),
        Color::from_rgba(150, 100, 255, 255),
        Color::from_rgba(150, 200, 255, 255),
        Color::from_rgba(255, 200, 100, 255),
        Color::from_rgba(255, 150, 100, 255),
    ];
    
    for (i, (label, color)) in labels.iter().zip(colors.iter()).enumerate() {
        let x = label_start_x + i as f32 * label_spacing;
        // Draw slanted text (macroquad doesn't support rotation, so we'll draw it at an angle visually)
        draw_text(label, x, label_y, 11.0, *color);
    }
    
    // Selected limbs to display
    let limbs = [
        ("Torso", 0),
        ("Head", 5),
        ("F.Arm.U", 10),
        ("F.Arm.L", 15),
        ("B.Leg.U", 40),
        ("B.Leg.L", 45),
    ];
    
    let limb_height = 40.0;
    let limb_width = 220.0;
    let mut y_offset = panel_y + 65.0;
    
    for (name, offset) in limbs.iter() {
        draw_limb_sense_compact(
            panel_x + 10.0,
            y_offset,
            limb_width,
            limb_height,
            name,
            &sense_flat[*offset..*offset + 5]
        );
        y_offset += limb_height + 5.0;
    }
}

fn draw_limb_sense_compact(x: f32, y: f32, _width: f32, height: f32, name: &str, data: &[f32]) {
    // data[0] = sin(angle), data[1] = cos(angle), data[2] = ang_vel, data[3] = vel_x, data[4] = vel_y
    
    // Draw limb name
    draw_text(name, x, y + height / 2.0 + 5.0, 14.0, WHITE);
    
    // Draw 5 vertical bars
    let bar_start_x = x + 60.0;
    let bar_width = 20.0;
    let bar_spacing = 30.0;
    let max_bar_height = height - 5.0;
    
    let values = [data[0], data[1], data[2], data[3], data[4]];
    let max_values = [1.0, 1.0, 10.0, 5.0, 5.0]; // Max values for normalization
    let colors = [
        Color::from_rgba(100, 150, 255, 200),
        Color::from_rgba(150, 100, 255, 200),
        Color::from_rgba(150, 200, 255, 200),
        Color::from_rgba(255, 200, 100, 200),
        Color::from_rgba(255, 150, 100, 200),
    ];
    
    for i in 0..5 {
        let bar_x = bar_start_x + i as f32 * bar_spacing;
        let normalized = (values[i] / max_values[i]).clamp(-1.0, 1.0);
        let bar_height = normalized * max_bar_height / 2.0;
        let center_y = y + max_bar_height / 2.0;
        
        // Draw background
        draw_rectangle(bar_x, y, bar_width, max_bar_height, Color::from_rgba(40, 40, 40, 200));
        
        // Draw center line
        draw_line(bar_x, center_y, bar_x + bar_width, center_y, 1.0, GRAY);
        
        // Draw value bar
        if bar_height > 0.0 {
            draw_rectangle(bar_x, center_y - bar_height, bar_width, bar_height, colors[i]);
        } else {
            draw_rectangle(bar_x, center_y, bar_width, -bar_height, colors[i]);
        }
    }
}

fn draw_action_visualization(action: &BuddyAction) {
    let panel_x = screen_width() - 360.0;
    let panel_y = 440.0;
    let panel_width = 340.0;
    let panel_height = 210.0;
    
    // Draw background panel
    draw_rectangle(panel_x, panel_y, panel_width, panel_height, Color::from_rgba(0, 0, 0, 180));
    
    // Title
    draw_text("ACTION DATA (9 joints)", panel_x + 10.0, panel_y + 25.0, 20.0, GREEN);
    
    let actions = [
        ("Neck", action.neck_joint),
        ("F.Shoulder", action.front_shoulder_joint),
        ("F.Elbow", action.front_elbow_joint),
        ("B.Shoulder", action.back_shoulder_joint),
        ("B.Elbow", action.back_elbow_joint),
        ("F.Hip", action.front_hip_joint),
        ("F.Knee", action.front_knee_joint),
        ("B.Hip", action.back_hip_joint),
        ("B.Knee", action.back_knee_joint),
    ];
    
    // Draw bars for each joint
    let bar_width = 28.0;
    let bar_spacing = 32.0;
    let max_bar_height = 60.0;
    let base_y = panel_y + 140.0;
    
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    
    for (i, (name, value)) in actions.iter().enumerate() {
        let x = panel_x + 10.0 + i as f32 * bar_spacing;
        let normalized = value.clamp(-10.0, 10.0) / 10.0; // Normalize to -1..1
        let bar_height = normalized * max_bar_height;
        
        let color = if normalized > 0.0 {
            Color::from_rgba(100, 255, 100, 200)
        } else {
            Color::from_rgba(255, 150, 100, 200)
        };
        
        if bar_height > 0.0 {
            draw_rectangle(x, base_y - bar_height, bar_width, bar_height, color);
        } else {
            draw_rectangle(x, base_y, bar_width, -bar_height, color);
        }
        
        // Draw joint label (rotated would be nice but keep it simple)
        draw_text(name, x - 5.0, panel_y + 50.0 + (i % 2) as f32 * 12.0, 12.0, DARKGRAY);
        
        min_value = min_value.min(*value);
        max_value = max_value.max(*value);
    }
    
    // Print min/max values
    draw_text(&format!("Min: {:.2}", min_value), panel_x + 10.0, panel_y + 170.0, 12.0, DARKGRAY);
    draw_text(&format!("Max: {:.2}", max_value), panel_x + 10.0, panel_y + 190.0, 12.0, DARKGRAY);
    
    // Draw center line
    draw_line(panel_x + 10.0, base_y, panel_x + panel_width - 10.0, base_y, 1.0, GRAY);
    
    draw_text(
        "Angular velocities (rad/s)",
        panel_x + 10.0,
        panel_y + 200.0,
        14.0,
        DARKGRAY,
    );
}

/// Draw neural network weight and activation visualizations
fn draw_network_visualizations(brain: &Brain, hidden_activations: &[f32]) {
    let panel_x = 20.0;
    let panel_y = 100.0;
    let spacing = 50.0;  // Increased spacing for larger labels
    let scale = 4.0;
    
    // Draw input->hidden weights (68x50)
    let y1 = panel_y;
    let input_hidden_weights = brain.get_input_hidden_weights();
    draw_weight_matrix(
        panel_x,
        y1,
        &input_hidden_weights,
        68,
        50,
        "Input→Hidden Weights (68x50)"
    );
    
    // Draw hidden activations (1x50)
    let y2 = y1 + 68.0 * scale + spacing;
    draw_activation_vector(
        panel_x,
        y2,
        hidden_activations,
        "Hidden Activations (1x50)"
    );
    
    // Draw hidden->output weights (50x9) - moved below activations
    let y3 = y2 + 1.0 * scale + spacing;
    let hidden_output_weights = brain.get_hidden_output_weights();
    draw_weight_matrix(
        panel_x,
        y3,
        &hidden_output_weights,
        50,
        9,
        "Hidden→Output Weights (50x9)"
    );
}

/// Convert weight matrix to blue/red colored image and draw it
/// Blue = positive weights (+1), Red = negative weights (-1)
fn draw_weight_matrix(x: f32, y: f32, weights: &[f32], rows: usize, cols: usize, title: &str) {
    // Create RGBA byte array for the image
    let mut bytes = vec![0u8; rows * cols * 4];
    
    // Find min and max values
    let min_val = weights.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    // Find max absolute value for symmetric normalization
    let max_abs = weights.iter().copied().map(|w| w.abs()).fold(0.0f32, f32::max).max(0.001);
    
    // Fill byte array with blue/red color scheme
    for i in 0..rows {
        for j in 0..cols {
            let weight = weights[i * cols + j];
            let normalized = weight / max_abs; // Range: -1 to +1
            
            let idx = (i * cols + j) * 4;
            
            if normalized > 0.0 {
                // Positive weights: blue
                let intensity = (normalized * 255.0) as u8;
                bytes[idx] = 0;           // R
                bytes[idx + 1] = 0;       // G
                bytes[idx + 2] = intensity; // B
                bytes[idx + 3] = 255;     // A
            } else {
                // Negative weights: red
                let intensity = (-normalized * 255.0) as u8;
                bytes[idx] = intensity;   // R
                bytes[idx + 1] = 0;       // G
                bytes[idx + 2] = 0;       // B
                bytes[idx + 3] = 255;     // A
            }
        }
    }
    
    // Create texture from bytes
    let texture = Texture2D::from_rgba8(cols as u16, rows as u16, &bytes);
    texture.set_filter(FilterMode::Nearest);
    
    // Draw title
    draw_text(title, x, y - 10.0, 28.0, WHITE);
    
    // Draw min/max values
    let range_text = format!("Min: {:.4}, Max: {:.4}", min_val, max_val);
    draw_text(&range_text, x, y - 35.0, 24.0, GRAY);
    
    // Draw the texture scaled 4x
    draw_texture_ex(
        &texture,
        x,
        y,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(cols as f32 * 4.0, rows as f32 * 4.0)),
            ..Default::default()
        },
    );
}

/// Draw activation vector as a horizontal bar (width 50)
/// Blue intensity shows activation strength
fn draw_activation_vector(x: f32, y: f32, activations: &[f32], title: &str) {
    let width = activations.len();
    
    // Create RGBA byte array for Nx1 image (horizontal)
    let mut bytes = vec![0u8; width * 4];
    
    // Find min and max values
    let min_val = activations.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = activations.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    // Find max for normalization (ReLU outputs are >= 0)
    let max_activation = 10.0;
    
    // Fill byte array with color based on activation values
    for i in 0..width {
        let activation = activations[i];
        let normalized = (activation / max_activation).clamp(-1.0, 1.0);
        let idx = i * 4;
        if activation > 0.0 {
            bytes[idx] = ((1.0 - normalized) * 255.0) as u8; // R (red for positive)
            bytes[idx + 1] = 0; // G (black for positive)
            bytes[idx + 2] = 0; // B (black for positive)
        } else if activation < 0.0 {
            bytes[idx] = 0; // R (black for negative)
            bytes[idx + 1] = 0; // G (black for negative)
            bytes[idx + 2] = ((1.0 - (-normalized)) * 255.0) as u8; // B (blue for negative)
        } else {
            bytes[idx] = 0; // R (black for 0)
            bytes[idx + 1] = 0; // G (black for 0)
            bytes[idx + 2] = 0; // B (black for 0)
        }
        bytes[idx + 3] = 255; // A
    }
    
    // Create texture from bytes (width pixels wide, 1 pixel tall)
    let texture = Texture2D::from_rgba8(width as u16, 1, &bytes);
    texture.set_filter(FilterMode::Nearest);
    
    // Draw title
    draw_text(title, x, y - 10.0, 28.0, WHITE);
    
    // Draw min/max values
    let range_text = format!("Min: {:.4}, Max: {:.4}", min_val, max_val);
    draw_text(&range_text, x, y - 35.0, 24.0, GRAY);
    
    // Draw the texture scaled 4x
    draw_texture_ex(
        &texture,
        x,
        y,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(width as f32 * 4.0, 1.0 * 4.0)),
            ..Default::default()
        },
    );
}

/// Draw sliders for brain initialization parameters
fn draw_init_param_sliders(params: &mut BrainInitParams) {
    let panel_x = screen_width() - 280.0;
    let panel_y = screen_height() - 260.0;
    let panel_width = 260.0;
    let panel_height = 150.0;
    
    // Draw background panel
    draw_rectangle(panel_x, panel_y, panel_width, panel_height, Color::from_rgba(0, 0, 0, 180));
    draw_rectangle_lines(panel_x, panel_y, panel_width, panel_height, 1.0, Color::from_rgba(80, 80, 80, 255));
    
    // Title
    draw_text("Brain Init Params", panel_x + 10.0, panel_y + 20.0, 18.0, Color::from_rgba(200, 180, 255, 255));
    
    let slider_x = panel_x + 100.0;
    let slider_width = 100.0;
    let slider_height = 8.0;
    let label_x = panel_x + 10.0;
    let value_x = panel_x + 210.0;
    
    let sliders = [
        ("IH Gain", &mut params.ih_gain, 0.0, 10.0, 0),
        ("HO Gain", &mut params.ho_gain, 0.0, 10.0, 1),
        ("IH Sparsity", &mut params.ih_sparsity, 0.0, 1.0, 2),
        ("HO Sparsity", &mut params.ho_sparsity, 0.0, 1.0, 3),
    ];
    
    let mouse_pos = mouse_position();
    let mouse_pressed = is_mouse_button_down(MouseButton::Left);
    let mouse_just_released = is_mouse_button_released(MouseButton::Left);
    
    // Release active slider if mouse released
    if mouse_just_released {
        params.active_slider = None;
    }
    
    for (i, (label, value, min, max, idx)) in sliders.into_iter().enumerate() {
        let y = panel_y + 35.0 + i as f32 * 28.0;
        
        // Draw label
        draw_text(label, label_x, y + 12.0, 14.0, WHITE);
        
        // Draw slider track
        let track_y = y + 5.0;
        draw_rectangle(slider_x, track_y, slider_width, slider_height, Color::from_rgba(60, 60, 60, 255));
        
        // Calculate slider position
        let normalized = (*value - min) / (max - min);
        let handle_x = slider_x + normalized * slider_width;
        let handle_width = 10.0;
        let handle_height = 16.0;
        
        // Draw filled portion
        draw_rectangle(slider_x, track_y, normalized * slider_width, slider_height, Color::from_rgba(120, 100, 200, 255));
        
        // Check if mouse is over the slider track (expand hit area vertically)
        let hit_area_y = track_y - 8.0;
        let hit_area_height = slider_height + 16.0;
        let is_over_track = mouse_pos.0 >= slider_x && mouse_pos.0 <= slider_x + slider_width
            && mouse_pos.1 >= hit_area_y && mouse_pos.1 <= hit_area_y + hit_area_height;
        
        // Start dragging if mouse pressed over track
        if mouse_pressed && is_over_track && params.active_slider.is_none() {
            params.active_slider = Some(idx);
        }
        
        // Update value if this slider is being dragged
        if params.active_slider == Some(idx) && mouse_pressed {
            let new_normalized = ((mouse_pos.0 - slider_x) / slider_width).clamp(0.0, 1.0);
            *value = min + new_normalized * (max - min);
        }
        
        // Draw handle
        let handle_color = if params.active_slider == Some(idx) {
            Color::from_rgba(180, 160, 255, 255)
        } else if is_over_track {
            Color::from_rgba(160, 140, 220, 255)
        } else {
            Color::from_rgba(140, 120, 200, 255)
        };
        draw_rectangle(
            handle_x - handle_width / 2.0,
            track_y - (handle_height - slider_height) / 2.0,
            handle_width,
            handle_height,
            handle_color,
        );
        
        // Draw value
        let value_text = if max <= 1.0 {
            format!("{:.2}", *value)
        } else {
            format!("{:.1}", *value)
        };
        draw_text(&value_text, value_x, y + 12.0, 14.0, Color::from_rgba(180, 180, 180, 255));
    }
}

/// Draw restart button and return true if clicked
fn draw_restart_button() -> bool {
    let button_x = screen_width() - 280.0;
    let button_y = screen_height() - 100.0;
    let button_width = 120.0;
    let button_height = 35.0;
    
    let mouse_pos = mouse_position();
    let is_hovered = mouse_pos.0 >= button_x && mouse_pos.0 <= button_x + button_width
        && mouse_pos.1 >= button_y && mouse_pos.1 <= button_y + button_height;
    
    let button_color = if is_hovered {
        Color::from_rgba(80, 120, 200, 255)
    } else {
        Color::from_rgba(60, 100, 180, 255)
    };
    
    draw_rectangle(button_x, button_y, button_width, button_height, button_color);
    draw_rectangle_lines(button_x, button_y, button_width, button_height, 2.0, WHITE);
    draw_text("Restart", button_x + 28.0, button_y + 23.0, 20.0, WHITE);
    
    is_hovered && is_mouse_button_pressed(MouseButton::Left)
}

/// Draw reset brain button and return true if clicked
fn draw_reset_button() -> bool {
    let button_x = screen_width() - 150.0;
    let button_y = screen_height() - 100.0;
    let button_width = 130.0;
    let button_height = 35.0;
    
    let mouse_pos = mouse_position();
    let is_hovered = mouse_pos.0 >= button_x && mouse_pos.0 <= button_x + button_width
        && mouse_pos.1 >= button_y && mouse_pos.1 <= button_y + button_height;
    
    let button_color = if is_hovered {
        Color::from_rgba(200, 80, 80, 255)
    } else {
        Color::from_rgba(180, 60, 60, 255)
    };
    
    draw_rectangle(button_x, button_y, button_width, button_height, button_color);
    draw_rectangle_lines(button_x, button_y, button_width, button_height, 2.0, WHITE);
    draw_text("Reset Brain", button_x + 12.0, button_y + 23.0, 20.0, WHITE);
    
    is_hovered && is_mouse_button_pressed(MouseButton::Left)
}
