use burn::nn::{Linear, LinearConfig, Initializer};
use burn::tensor::Tensor;
use burn::module::Param;
use burn_ndarray::{NdArray, NdArrayDevice};
use rand::Rng;
use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufWriter};

#[cfg(feature = "mps")]
use burn_tch::{LibTorch, LibTorchDevice};

// Network architecture constants
pub const INPUT_SIZE: usize = 68;  // From BuddyIO::sense_size()
pub const HIDDEN_SIZE: usize = 50;
pub const OUTPUT_SIZE: usize = 9;  // From BuddyIO::action_size()

// Sparsity parameter: probability that a weight is set to zero (0.0 = dense, 1.0 = all zeros)
// We want about 80% of weights to be zero in both matrices
pub const SPARSITY_INPUT_HIDDEN: f32 = 0.9;
pub const SPARSITY_HIDDEN_OUTPUT: f32 = 0.9;

/// Type alias for the backend we're using
/// Uses MPS (Metal) if available, otherwise falls back to CPU (NdArray)
#[cfg(feature = "mps")]
type B = LibTorch;

#[cfg(not(feature = "mps"))]
type B = NdArray;

/// Helper to get the appropriate device
#[cfg(feature = "mps")]
fn get_device() -> LibTorchDevice {
    LibTorchDevice::Mps
}

#[cfg(not(feature = "mps"))]
fn get_device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

/// Brain for the buddy using a simple feedforward neural network
/// Architecture: INPUT_SIZE -> HIDDEN_SIZE -> OUTPUT_SIZE
/// Uses ReLU activation in hidden layer, linear output
#[derive(Debug)]
pub struct Brain {
    /// Linear layer from input to hidden (INPUT_SIZE -> HIDDEN_SIZE)
    input_hidden: Linear<B>,
    /// Linear layer from hidden to output (HIDDEN_SIZE -> OUTPUT_SIZE)
    hidden_output: Linear<B>,
}

impl Brain {
    /// Create a new brain with random weights using Burn's smart initialization
    /// 
    /// # Arguments
    /// * `ih_gain` - Gain for input->hidden layer initialization
    /// * `ho_gain` - Gain for hidden->output layer initialization
    /// * `ih_sparsity` - Probability that input->hidden weights are zero (0.0 = dense, 1.0 = all zeros)
    /// * `ho_sparsity` - Probability that hidden->output weights are zero (0.0 = dense, 1.0 = all zeros)
    /// 
    /// Uses XavierUniform initialization which is appropriate for angular velocities
    /// as it maintains smaller initial weights to avoid extreme rotations.
    /// Initializer draws from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
    pub fn new(ih_gain: f64, ho_gain: f64, ih_sparsity: f32, ho_sparsity: f32) -> Self {
        let device = get_device();
        
        // Initialize layers with XavierUniform
        let mut input_hidden = LinearConfig::new(INPUT_SIZE, HIDDEN_SIZE)
            .with_initializer(Initializer::XavierUniform { gain: ih_gain })
            .init(&device);
        
        let mut hidden_output = LinearConfig::new(HIDDEN_SIZE, OUTPUT_SIZE)
            .with_initializer(Initializer::XavierUniform { gain: ho_gain })
            .init(&device);
        
        // Apply random sparsity to input->hidden weights if sparsity > 0
        if ih_sparsity > 0.0 {
            let ih_weights = input_hidden.weight.val();
            let ih_mask = Self::create_random_mask(INPUT_SIZE, HIDDEN_SIZE, ih_sparsity, &device);
            let ih_sparse_weights = ih_weights * ih_mask;
            input_hidden.weight = Param::from_tensor(ih_sparse_weights);
        }
        
        // Apply random sparsity to hidden->output weights if sparsity > 0
        if ho_sparsity > 0.0 {
            let ho_weights = hidden_output.weight.val();
            let ho_mask = Self::create_random_mask(HIDDEN_SIZE, OUTPUT_SIZE, ho_sparsity, &device);
            let ho_sparse_weights = ho_weights * ho_mask;
            hidden_output.weight = Param::from_tensor(ho_sparse_weights);
        }
        
        Self {
            input_hidden,
            hidden_output,
        }
    }
    
    /// Create a new brain with dense random weights (no sparsity)
    /// Uses default gains of 1.0 for input->hidden and 2.0 for hidden->output
    pub fn new_random() -> Self {
        Self::new(1.0, 2.0, 0.0, 0.0)
    }
    
    /// Create a new brain with random sparse initialization
    /// Uses the sparsity constants defined at module level (~90% zeros)
    /// and higher gains to compensate for sparsity
    pub fn new_random_sparse() -> Self {
        Self::new(4.0, 5.0, SPARSITY_INPUT_HIDDEN, SPARSITY_HIDDEN_OUTPUT)
    }
    
    /// Create a random sparsity mask
    /// Returns a tensor where elements are 1.0 if they should be kept, 0.0 if zeroed.
    /// Each entry is independently set to zero with probability `sparsity`.
    fn create_random_mask(rows: usize, cols: usize, sparsity: f32, device: &<B as burn::tensor::backend::Backend>::Device) -> Tensor<B, 2> {
        let mut rng = rand::thread_rng();
        let mut mask_data = Vec::with_capacity(rows * cols);

        for _ in 0..(rows * cols) {
            let r: f32 = rng.gen();
            if r < sparsity {
                mask_data.push(0.0);
            } else {
                mask_data.push(1.0);
            }
        }

        let mask_1d = Tensor::<B, 1>::from_data(mask_data.as_slice(), device);
        mask_1d.reshape([rows, cols])
    }
    
    /// Forward pass through the network
    /// Takes a sense array (68 values) and outputs an action array (9 values)
    /// Outputs are linear (no activation function)
    pub fn forward(&self, sense: &[f32]) -> Vec<f32> {
        assert_eq!(sense.len(), INPUT_SIZE, "Sense array must have {} elements", INPUT_SIZE);
        
        let device = get_device();
        
        // Convert input to Burn tensor [1, INPUT_SIZE]
        let input_tensor: Tensor<B, 1> = Tensor::from_floats(sense, &device);
        let input_tensor: Tensor<B, 2> = input_tensor.reshape([1, INPUT_SIZE]);
        
        // Forward pass: input -> hidden (with ReLU) -> output
        let hidden = self.input_hidden.forward(input_tensor);
        //let hidden = burn::tensor::activation::relu(hidden);
        let output = self.hidden_output.forward(hidden);
        
        // Convert output tensor back to Vec<f32>
        let output_data = output.into_data();
        output_data.to_vec::<f32>().unwrap()
    }
    
    /// Forward pass that also returns hidden layer activations
    /// Returns (output, hidden_activations)
    pub fn forward_with_activations(&self, sense: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(sense.len(), INPUT_SIZE, "Sense array must have {} elements", INPUT_SIZE);
        
        let device = get_device();
        
        // Convert input to Burn tensor [1, INPUT_SIZE]
        let input_tensor: Tensor<B, 1> = Tensor::from_floats(sense, &device);
        let input_tensor: Tensor<B, 2> = input_tensor.reshape([1, INPUT_SIZE]);
        
        // Forward pass with intermediate activations
        let hidden = self.input_hidden.forward(input_tensor);
        // let hidden = burn::tensor::activation::relu(hidden_pre);
        let output = self.hidden_output.forward(hidden.clone());
        
        // Convert tensors back to Vec<f32>
        let output_vec = output.into_data().to_vec::<f32>().unwrap();
        let hidden_vec = hidden.into_data().to_vec::<f32>().unwrap();
        
        (output_vec, hidden_vec)
    }
    
    /// Get total number of weights in the network
    pub fn weight_count(&self) -> usize {
        // Each linear layer has weights (in × out) and biases (out)
        let input_hidden_params = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE;
        let hidden_output_params = HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE;
        input_hidden_params + hidden_output_params
    }
    
    /// Extract input->hidden weights as a flat Vec<f32> for visualization
    /// Returns weights in row-major order: [INPUT_SIZE x HIDDEN_SIZE]
    pub fn get_input_hidden_weights(&self) -> Vec<f32> {
        // Access the weight tensor from the Linear layer
        // Linear layer weights are stored as [in_features, out_features] = [INPUT_SIZE, HIDDEN_SIZE]
        let weight_tensor = self.input_hidden.weight.val();
        
        // Convert to Vec<f32>
        let data = weight_tensor.into_data();
        let weights = data.to_vec::<f32>().unwrap();
        
        // Transpose from [HIDDEN_SIZE, INPUT_SIZE] to [INPUT_SIZE, HIDDEN_SIZE]
        let mut transposed = vec![0.0; INPUT_SIZE * HIDDEN_SIZE];
        for i in 0..INPUT_SIZE {
            for j in 0..HIDDEN_SIZE {
                transposed[i * HIDDEN_SIZE + j] = weights[j * INPUT_SIZE + i];
            }
        }
        
        transposed
    }
    
    /// Extract hidden->output weights as a flat Vec<f32> for visualization
    /// Returns weights in row-major order: [HIDDEN_SIZE x OUTPUT_SIZE]
    pub fn get_hidden_output_weights(&self) -> Vec<f32> {
        // Access the weight tensor from the Linear layer
        let weight_tensor = self.hidden_output.weight.val();
        
        // Convert to Vec<f32>
        let data = weight_tensor.into_data();
        let weights = data.to_vec::<f32>().unwrap();
        
        // Transpose from [OUTPUT_SIZE, HIDDEN_SIZE] to [HIDDEN_SIZE, OUTPUT_SIZE]
        let mut transposed = vec![0.0; HIDDEN_SIZE * OUTPUT_SIZE];
        for i in 0..HIDDEN_SIZE {
            for j in 0..OUTPUT_SIZE {
                transposed[i * OUTPUT_SIZE + j] = weights[j * HIDDEN_SIZE + i];
            }
        }
        
        transposed
    }

    /// Save all network parameters (weights and biases) to a binary file.
    /// Layout (all values as little-endian f32):
    /// [input_hidden.weight] [input_hidden.bias] [hidden_output.weight] [hidden_output.bias]
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);

        // Helper to write a slice of f32 as raw little-endian bytes
        fn write_f32_slice<W: Write>(writer: &mut W, data: &[f32]) -> io::Result<()> {
            for &v in data {
                writer.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }

        // Extract all parameters as flat Vec<f32>
        let ih_w = self
            .input_hidden
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let ih_b = self
            .input_hidden
            .bias
            .as_ref()
            .expect("input_hidden bias missing")
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let ho_w = self
            .hidden_output
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let ho_b = self
            .hidden_output
            .bias
            .as_ref()
            .expect("hidden_output bias missing")
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Sanity check on expected sizes
        debug_assert_eq!(ih_w.len(), INPUT_SIZE * HIDDEN_SIZE);
        debug_assert_eq!(ih_b.len(), HIDDEN_SIZE);
        debug_assert_eq!(ho_w.len(), HIDDEN_SIZE * OUTPUT_SIZE);
        debug_assert_eq!(ho_b.len(), OUTPUT_SIZE);

        write_f32_slice(&mut file, &ih_w)?;
        write_f32_slice(&mut file, &ih_b)?;
        write_f32_slice(&mut file, &ho_w)?;
        write_f32_slice(&mut file, &ho_b)?;

        file.flush()
    }

    /// Load a brain from a binary weights file previously written by `save_to_file`.
    /// Validates that the number of weights matches the expected architecture.
    pub fn from_file(path: &str) -> io::Result<Self> {
        let device = get_device();

        // Read entire file
        let mut reader = BufReader::new(File::open(path)?);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;

        // Total expected number of f32 values
        let expected_ih_w = INPUT_SIZE * HIDDEN_SIZE;
        let expected_ih_b = HIDDEN_SIZE;
        let expected_ho_w = HIDDEN_SIZE * OUTPUT_SIZE;
        let expected_ho_b = OUTPUT_SIZE;
        let expected_total = expected_ih_w + expected_ih_b + expected_ho_w + expected_ho_b;

        if buffer.len() != expected_total * 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Weights file has incorrect size: got {} bytes, expected {} bytes",
                    buffer.len(),
                    expected_total * 4
                ),
            ));
        }

        // Helper to read f32 values from raw bytes
        fn read_f32_slice(bytes: &[u8]) -> Vec<f32> {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out
        }

        // Split buffer into the four parameter groups
        let mut offset_bytes = 0usize;
        let take_bytes = |count_f32: usize, offset: &mut usize| -> &[u8] {
            let byte_len = count_f32 * 4;
            let start = *offset;
            let end = start + byte_len;
            *offset = end;
            &buffer[start..end]
        };

        let ih_w_bytes = take_bytes(expected_ih_w, &mut offset_bytes);
        let ih_b_bytes = take_bytes(expected_ih_b, &mut offset_bytes);
        let ho_w_bytes = take_bytes(expected_ho_w, &mut offset_bytes);
        let ho_b_bytes = take_bytes(expected_ho_b, &mut offset_bytes);

        let ih_w = read_f32_slice(ih_w_bytes);
        let ih_b = read_f32_slice(ih_b_bytes);
        let ho_w = read_f32_slice(ho_w_bytes);
        let ho_b = read_f32_slice(ho_b_bytes);

        // Create a new brain so that we can query the expected tensor shapes
        let mut brain = Brain::new(1.0, 1.0, 0.0, 0.0);

        // Rebuild tensors with the same shapes as the randomly initialized ones
        let ih_w_shape = brain.input_hidden.weight.val().shape();
        let ih_b_shape = brain
            .input_hidden
            .bias
            .as_ref()
            .expect("input_hidden bias missing")
            .val()
            .shape();
        let ho_w_shape = brain.hidden_output.weight.val().shape();
        let ho_b_shape = brain
            .hidden_output
            .bias
            .as_ref()
            .expect("hidden_output bias missing")
            .val()
            .shape();

        // Build tensors from flat data as 1D, then reshape to the original shapes.
        // This avoids rank mismatches in Burn's `from_floats` API.
        let ih_w_tensor = Tensor::<B, 1>::from_floats(ih_w.as_slice(), &device).reshape(ih_w_shape);
        let ih_b_tensor = Tensor::<B, 1>::from_floats(ih_b.as_slice(), &device).reshape(ih_b_shape);
        let ho_w_tensor = Tensor::<B, 1>::from_floats(ho_w.as_slice(), &device).reshape(ho_w_shape);
        let ho_b_tensor = Tensor::<B, 1>::from_floats(ho_b.as_slice(), &device).reshape(ho_b_shape);

        brain.input_hidden.weight = Param::from_tensor(ih_w_tensor);
        brain.input_hidden.bias = Some(Param::from_tensor(ih_b_tensor));
        brain.hidden_output.weight = Param::from_tensor(ho_w_tensor);
        brain.hidden_output.bias = Some(Param::from_tensor(ho_b_tensor));

        Ok(brain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_brain_creation() {
        let brain = Brain::new_random();
        // Brain should be created successfully with Burn modules
        assert_eq!(brain.weight_count(), INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE);
    }
    
    #[test]
    fn test_brain_forward() {
        let brain = Brain::new_random();
        let sense = vec![0.0; INPUT_SIZE];
        let output = brain.forward(&sense);
        
        assert_eq!(output.len(), OUTPUT_SIZE);
        
        // Outputs should be finite numbers
        for &val in &output {
            assert!(val.is_finite(), "Output {} is not finite", val);
        }
    }
    
    #[test]
    fn test_weight_count() {
        let brain = Brain::new_random();
        let expected = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE;
        assert_eq!(brain.weight_count(), expected);
    }
    
    #[test]
    fn test_sparse_initialization() {
        let brain = Brain::new_random_sparse();
        
        // Test that the brain works with sparse weights
        let sense = vec![0.5; INPUT_SIZE];
        let output = brain.forward(&sense);
        assert_eq!(output.len(), OUTPUT_SIZE);
        
        // Get weights and count zeros
        let ih_weights = brain.get_input_hidden_weights();
        let ho_weights = brain.get_hidden_output_weights();
        
        let ih_zeros = ih_weights.iter().filter(|&&w| w == 0.0).count();
        let ho_zeros = ho_weights.iter().filter(|&&w| w == 0.0).count();
        
        // With random sparsity (~80% zeros), we should have significant sparsity
        // For input->hidden: expect between 70% and 90% zeros
        let ih_sparsity = ih_zeros as f32 / ih_weights.len() as f32;
        assert!(ih_sparsity > 0.7 && ih_sparsity < 0.9,
                "Input-hidden sparsity {} should be between 0.7 and 0.9", ih_sparsity);
        
        // For hidden->output: expect between 70% and 90% zeros
        let ho_sparsity = ho_zeros as f32 / ho_weights.len() as f32;
        assert!(ho_sparsity > 0.7 && ho_sparsity < 0.9,
                "Hidden-output sparsity {} should be between 0.7 and 0.9", ho_sparsity);
        
        println!("Input-hidden sparsity: {:.2}%", ih_sparsity * 100.0);
        println!("Hidden-output sparsity: {:.2}%", ho_sparsity * 100.0);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let brain = Brain::new_random_sparse();

        let path = "test_brain_weights.bin";
        brain.save_to_file(path).expect("failed to save brain weights");

        let loaded = Brain::from_file(path).expect("failed to load brain weights");

        // Clean up test file
        let _ = fs::remove_file(path);

        let sense = vec![0.1; INPUT_SIZE];
        let out_original = brain.forward(&sense);
        let out_loaded = loaded.forward(&sense);

        assert_eq!(out_original.len(), out_loaded.len());
        for (a, b) in out_original.iter().zip(out_loaded.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch after load: {} vs {}", a, b);
        }
    }
}