use burn::nn::{Linear, LinearConfig, Initializer};
use burn::tensor::Tensor;
use burn::module::Param;
use burn_ndarray::{NdArray, NdArrayDevice};
use rand::Rng;

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
    /// Uses XavierUniform initialization which is appropriate for angular velocities
    /// as it maintains smaller initial weights to avoid extreme rotations
    pub fn new_random() -> Self {
        let device = get_device();
        
        // Create linear layers with XavierUniform initialization
        // This initializer draws from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
        // Helps maintain stable gradients and prevents extreme initial angular velocities
        let input_hidden = LinearConfig::new(INPUT_SIZE, HIDDEN_SIZE)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(&device);
        
        let hidden_output = LinearConfig::new(HIDDEN_SIZE, OUTPUT_SIZE)
            .with_initializer(Initializer::XavierUniform { gain: 2.0 })
            .init(&device);
        
        Self {
            input_hidden,
            hidden_output,
        }
    }
    
    /// Create a new brain with random sparse initialization
    /// Each weight is independently set to zero with probability given by the
    /// sparsity constants above, so that on average ~80% of weights are zero.
    pub fn new_random_sparse() -> Self {
        let device = get_device();
        
        // Initialize layers with XavierUniform
        let mut input_hidden = LinearConfig::new(INPUT_SIZE, HIDDEN_SIZE)
            .with_initializer(Initializer::XavierUniform { gain: 4.0 })
            .init(&device);
        
        let mut hidden_output = LinearConfig::new(HIDDEN_SIZE, OUTPUT_SIZE)
            .with_initializer(Initializer::XavierUniform { gain: 5.0 })
            .init(&device);
        
        // Apply random sparsity to input->hidden weights
        // Weight tensor shape is [in_features, out_features] = [INPUT_SIZE, HIDDEN_SIZE]
        let ih_weights = input_hidden.weight.val();
        let ih_mask = Self::create_random_mask(INPUT_SIZE, HIDDEN_SIZE, SPARSITY_INPUT_HIDDEN, &device);
        let ih_sparse_weights = ih_weights * ih_mask;
        input_hidden.weight = Param::from_tensor(ih_sparse_weights);
        
        // Apply random sparsity to hidden->output weights
        // Weight tensor shape is [in_features, out_features] = [HIDDEN_SIZE, OUTPUT_SIZE]
        let ho_weights = hidden_output.weight.val();
        let ho_mask = Self::create_random_mask(HIDDEN_SIZE, OUTPUT_SIZE, SPARSITY_HIDDEN_OUTPUT, &device);
        let ho_sparse_weights = ho_weights * ho_mask;
        hidden_output.weight = Param::from_tensor(ho_sparse_weights);
        
        Self {
            input_hidden,
            hidden_output,
        }
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
    
    #[cfg(not(feature = "mps"))]
    fn create_random_mask(rows: usize, cols: usize, sparsity: f32, device: &NdArrayDevice) -> Tensor<B, 2> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
}