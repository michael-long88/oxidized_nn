use rand::Rng;
use std::{slice, vec, collections::HashSet};


/// A struct to hold the configuration for the neural network.
/// 
/// This struct implements a `default()` function that returns a default configuration
/// consisting of:
/// n_folds: 5,
/// learning_rate: 0.1,
/// learning_rate_decay: 0.01,
/// n_epochs: 1000,
/// n_layers: 5,
/// n_hidden_nodes: vec![],
#[derive(Debug, Clone)]
pub struct NeuralNetConfig {
    n_folds: usize,
    learning_rate: f64,
    learning_rate_decay: f64,
    n_epochs: usize,
    n_layers: usize,
    n_hidden_nodes: Vec<usize>,
}

impl Default for NeuralNetConfig {
    fn default() -> Self {
        NeuralNetConfig {
            n_folds: 5,
            learning_rate: 0.1,
            learning_rate_decay: 0.01,
            n_epochs: 1000,
            n_layers: 5,
            n_hidden_nodes: vec![],
        }
    }
}

impl NeuralNetConfig {
    /// Necessary function to set the number of hidden nodes in each hidden layer.
    /// 
    /// This function should be run before initializing the neural network.
    pub fn set_hidden_nodes(&mut self) {
        let mut hidden_nodes = vec![10; self.n_layers];
        for hidden_node_count in hidden_nodes.iter_mut().take(self.n_layers) {
            *hidden_node_count += 1;
        }
        self.n_hidden_nodes = hidden_nodes;
    }
}

/// Enum to hold the different types of layers in a neural network.
#[derive(Clone)]
pub enum Layer {
    Hidden(Vec<Node>),
    Output(Vec<Node>)
}

/// Allows us to use `into_iter()` on the `Layer` enum.
impl IntoIterator for Layer {
    type Item = Node;
    type IntoIter = vec::IntoIter<Node>;
  
    fn into_iter(self) -> vec::IntoIter<Node> {
        let vec = match self {
            Layer::Hidden(vec) => vec,
            Layer::Output(vec) => vec,
        };
        
        vec.into_iter()
    }
}

/// Allows us to use `into_iter()` on the `Layer` enum.
impl<'a> IntoIterator for &'a Layer {
  type Item = &'a Node;
  type IntoIter = slice::Iter<'a, Node>;

    fn into_iter(self) -> slice::Iter<'a, Node> {
        let slice = match self {
            Layer::Hidden(vec) => vec.as_slice(),
            Layer::Output(vec) => vec.as_slice(),
        };
        
        slice.iter()
    }
}

/// A struct to hold a single node in a neural network.
/// 
/// This struct implements a `default()` function that returns a default configuration
/// consisting of:
/// weights: vec![],
/// output: 0.0,
/// delta: 0.0,
#[derive(Clone)]
pub struct Node {
    pub weights: Vec<f64>,
    pub output: f64,
    pub delta: f64,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            weights: vec![],
            output: 0.0,
            delta: 0.0,
        }
    }
}

/// Struct to hold a neural network.
/// 
/// This struct implements a `default()` function that returns a default configuration
/// consisting of:
/// hidden_layers: Vec::new(),
/// output_layer: Layer::Output(Vec::new()),
/// input_layer: Vec::new(),
pub struct NeuralNetwork {
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
    input_layer: Vec<f64>,
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        NeuralNetwork {
            hidden_layers: Vec::new(),
            output_layer: Layer::Output(Vec::new()),
            input_layer: Vec::new(),
        }
    }
}

/// Creates a new neural network with the given config.
/// 
/// A neural network consists of an input layer, one or more hidden layers, and an output layer.
/// 
/// # Examples
/// 
/// ```
/// use oxidized_nn::*;
/// 
/// 
/// let dataset = vec![
///     [0.678470254957507, 0.8342028985507247, 0.9502341282805183, 0.834307116104869, 0.7818001487726258, 0.1731315042573321, 0.8091603053435115, 1.0],
///     [0.601038715769594, 0.7971014492753623, 0.9210497658717195, 0.8107865168539328, 0.7146045127696502, 0.41780983916745507, 0.773587786259542, 1.0],
///     [0.832389046270066, 0.9263768115942029, 0.9444625939235544, 0.9274906367041198, 0.8829655343416811, 0.4820245979186377, 0.9251908396946564, 2.0],
///     [0.7950897072710104, 0.9084057971014493, 0.9390177501905694, 0.8985767790262171, 0.8638730473592858, 0.5528618732261117, 0.8972519083969467, 2.0]
/// ];
/// let classifications = vec![1, 2];
/// let n_classifications = classifications.len();
/// 
/// let mut nn_config = NeuralNetConfig::default();
/// nn_config.set_hidden_nodes();
/// 
/// let mut neural_network = NeuralNetwork::default();
/// neural_network = initialize_neural_network(nn_config, neural_network, dataset[0].len(), n_classifications);
/// ```
pub fn initialize_neural_network(config: NeuralNetConfig, mut neural_network: NeuralNetwork, num_inputs: usize, num_outputs: usize) -> NeuralNetwork {
    let mut rng = rand::thread_rng();

    for hidden_layer_index in 0..config.n_layers {
        let mut hidden_layer = vec![];
        
        for _ in 0..config.n_hidden_nodes[hidden_layer_index] {
            // The number of nodes in each hidden layer is equal to the number of 
            // nodes in the previous hidden layer, plus one for the bias node
            let n_previous_layer_nodes = if hidden_layer_index == 0 {
                num_inputs + 1
            } else {
                config.n_hidden_nodes[hidden_layer_index - 1] + 1
            };

            let mut node = Node::default();

            // Each node's weights are initialized with a random value between 0 and 1
            for _ in 0..n_previous_layer_nodes {
                node.weights.push(rng.gen());
            }

            hidden_layer.push(node);
        }
        neural_network.hidden_layers.push(Layer::Hidden(hidden_layer));
    }

    let mut output_layer = vec![];

    for _ in 0..num_outputs {
        let mut node = Node::default();

            // Each node's weights are initialized with a random value between 0 and 1
            for _ in 0..config.n_hidden_nodes[0] {
                node.weights.push(rng.gen());
            }

        output_layer.push(node);
    }

    neural_network.output_layer = Layer::Output(output_layer);

    neural_network
}

/// Activation function used to smooth activation values.
/// 
/// There are a number of functions that can be used here, but for the 
/// purpose of this project, we will use the sigmoid function.
pub fn activation_function(activation_value: f64) -> f64 {
    1.0 / (1.0 + (-activation_value).exp())
}


/// Compute derivate of neuron output.
///  
/// For sigmoid derivative, the derivative is d/dx f(x) = f(x)(1-f(x))
pub fn transfer_derivative(output: f64) -> f64 {
    output * (1.0 - output)
}

/// Computes the activation value of a neuron given incoming activations, weights, and bias.
pub fn activate(neuron: &Node, input_values: &[f64]) -> f64 {
    // The last weight in the list is the bias
    let mut activation = neuron.weights[neuron.weights.len() - 1];

    for (index, weight) in input_values.iter().enumerate().take(neuron.weights.len() - 1) {
        activation += weight * input_values[index];
    }

    activation_function(activation)
}

impl NeuralNetwork {
    /// Backpropagates the error from output layer through hidden layers to input layer
    /// 
    /// The error in the output layer is actual classification minus predicted classification
    /// times the derivative of the ouput.
    /// 
    /// In a hidden layer, for each neuron, we take the weight of that neuron's link that's going to
    /// the next layer and multiply it by the error of the next layer neuron that it's linked to. 
    /// This is repeated and summed for each link connecting the neuron to the next layer.
    pub fn backpropagate_error(&mut self, output_values: Vec<f64>) {
        let output_layer = match &mut self.output_layer {
            Layer::Output(output_layer) => output_layer,
            _ => panic!("Output layer is not of type Layer::Output"),
        };
    
        // Error in the output layer is actual classification minus predicted classification
        // times the derivative of the ouput.
        for node_index in 0..output_layer.len() {
            let error = output_values[node_index] - output_layer[node_index].output;
            output_layer[node_index].delta = error * transfer_derivative(output_layer[node_index].output);
        }
    
        let last_hidden_layer_index = self.hidden_layers.len() - 1;
    
        for hidden_layer_index in (0..self.hidden_layers.len()).rev() {
            let mut hidden_layer = match &mut self.hidden_layers[hidden_layer_index] {
                Layer::Hidden(hidden_layer) => hidden_layer.clone(),
                _ => panic!("Hidden layer is not of type Layer::Hidden"),
            };
    
            let mut next_hidden_layer = Layer::Output(output_layer.clone());
    
            if hidden_layer_index != last_hidden_layer_index {
                next_hidden_layer = self.hidden_layers[hidden_layer_index + 1].clone()
            }
    
            let next_hidden_layer = match next_hidden_layer {
                Layer::Hidden(next_hidden_layer) => next_hidden_layer,
                Layer::Output(next_hidden_layer) => next_hidden_layer
            };

            // Error in each hidden layer node is the accumulated (weight * error) of each node in the next layer
            // times the derivative of the hidden layer node.
            for (hidden_layer_node_index, hidden_layer_node) in hidden_layer.iter_mut().enumerate() {
                let mut error = 0.0;
    
                next_hidden_layer.iter().for_each(|next_layer_node| {
                    error += next_layer_node.weights[hidden_layer_node_index] * next_layer_node.delta;
                });
    
                hidden_layer_node.delta = error * transfer_derivative(hidden_layer_node.output);
            }
            
            self.hidden_layers[hidden_layer_index] = Layer::Hidden(hidden_layer);
        }
    }

    /// Forward propagates the input row of data through the neural network.
    /// 
    /// Forward propagates input row of data, passing it through hidden layers in network, and
    /// using results to define the final output layer value.
    pub fn forward_propagate(&mut self, input_values: Vec<f64>) {
        let mut outputs = input_values;
    
        for layer in self.hidden_layers.iter_mut() {
            let hidden_layer = match layer {
                Layer::Hidden(hidden_layer) => hidden_layer,
                _ => panic!("Hidden layer is not of type Layer::Hidden"),
            };
    
            let mut new_inputs: Vec<f64> = Vec::new();
    
            for neuron in hidden_layer.iter_mut() {
                let activation_cost = activate(neuron, &outputs);
                let output_value = activation_function(activation_cost);
                neuron.output = output_value;
                new_inputs.push(output_value)
            }
    
            outputs = new_inputs;
        }

        let output_layer = match &mut self.output_layer {
            Layer::Output(output_layer) => output_layer,
            _ => panic!("Output layer is not of type Layer::Output"),
        };

        for neuron in &mut output_layer.iter_mut() {
            let activation_cost = activate(neuron, &outputs);
            let output_value = activation_function(activation_cost);
            neuron.output = output_value;
        }
    }

    /// Given the input values and the learning rate, update the weights of the neural network.
    pub fn update_weights(&mut self, input_values: &[f64], learning_rate: f64) {
        for hidden_layer_index in 0..self.hidden_layers.len() {
            let mut hidden_layer = match &mut self.hidden_layers[hidden_layer_index] {
                Layer::Hidden(hidden_layer) => hidden_layer.clone(),
                _ => panic!("Hidden layer is not of type Layer::Hidden"),
            };
    
            let inputs = if hidden_layer_index == 0 {
                input_values[..input_values.len() - 1].to_vec()
            } else {
                let previous_layer = self.hidden_layers[hidden_layer_index - 1].clone();
                previous_layer.into_iter().map(|neuron| neuron.output).collect()
            };
    
            // For each neuron, update its weight by adding the learning rate times
            // slope in error from expected values times previous layer's input
            for neuron in hidden_layer.iter_mut() {
                let weight_length = neuron.weights.len();

                for (index, weight) in neuron.weights.iter_mut().enumerate().take(weight_length - 2) {
                    *weight += learning_rate * neuron.delta * inputs[index];
                }

                neuron.weights[weight_length - 1] += learning_rate * neuron.delta;
            }
    
            self.hidden_layers[hidden_layer_index] = Layer::Hidden(hidden_layer);
        }
    }

    /// Given a row of input values, predict the output of the row.
    pub fn predict(&mut self, input_values: Vec<f64>) -> usize {
        self.forward_propagate(input_values);

        let output_layer = match &self.output_layer {
            Layer::Output(output_layer) => output_layer,
            _ => panic!("Output layer is not of type Layer::Output"),
        };

        let max_value = output_layer.iter()
            .max_by(|a, b| a.output.partial_cmp(&b.output).unwrap())
            .unwrap()
            .output;
        output_layer.iter().position(|neuron| neuron.output.eq(&max_value)).unwrap()
    }

    /// Train the neural network using stochastic gradient descent.
    /// 
    /// 1. Forward propagate training row through the network.
    /// 2. One hot encode expected output classification.
    /// 3. Backpropagate error through network to get error gradient (delta).
    /// 4. Update weights based on error gradient.
    /// 5. Decay learning rate.
    pub fn train_network(&mut self, training_set: Vec<Vec<f64>>, learning_rate: f64, decay: f64, n_epochs: usize, n_classifications: usize) {
        let mut new_learning_rate = learning_rate;

        for epoch_index in 0..n_epochs {
            let mut sse = 0.;

            for (row_index, input_values) in training_set.iter().enumerate() {
                // Step 1: Forward propagate training row through the network.
                self.forward_propagate(input_values.to_vec());

                // Step 2: One hot encode expected output classification.
                let mut output_values = vec![0.; n_classifications];
                let last_value = input_values[input_values.len() - 1] as usize;
                output_values[last_value - 1] = 1.;

                let output_layer = match &mut self.output_layer {
                    Layer::Output(output_layer) => output_layer,
                    _ => panic!("Output layer is not of type Layer::Output"),
                };

                // Compute sum of error at each neuron, square to make positive
                for value_index in 0..output_values.len() {
                    let error = output_values[value_index] - output_layer[value_index].output;
                    sse += error * error;
                }

                // Step 3: Backpropagate error through network to get error gradient (delta).
                self.backpropagate_error(output_values.to_vec());

                // Step 4: Update weights based on error gradient.
                self.update_weights(input_values, new_learning_rate);

                // Step 5: Decay learning rate.
                new_learning_rate = learning_rate * (1. / (1. + decay * row_index as f64));
            }

            if epoch_index % 100 == 0 || epoch_index == n_epochs - 1 {
                println!("Epoch: {}, Learning Rate: {}, SSE: {}", epoch_index, new_learning_rate, sse);

                if epoch_index == n_epochs - 1 {
                    println!("\n");
                }
            }
        }
    }
}

/// Split a dataset into n folds for n-fold cross validation.
pub fn cross_validation_split(dataset: &[Vec<f64>], n_folds: usize) -> Vec<Vec<Vec<f64>>> {
    let mut dataset_copy = dataset.to_vec();
    let mut dataset_split = Vec::new();
    let fold_size = dataset.len() / n_folds;

    for _ in 0..n_folds {
        let mut fold = Vec::new();
        while fold.len() < fold_size {
            let index = rand::thread_rng().gen_range(0..dataset_copy.len());
            fold.push(dataset_copy[index].clone());
            dataset_copy.remove(index);
        }
        dataset_split.push(fold);
    }

    dataset_split
}

/// Run a neural network on a dataset.
/// 
/// This function runs through the whole process of creating a neural network, training it, and
/// testing it on a dataset. It returns a vector of the accuracy scores for each fold.
pub fn run_network(dataset: &[Vec<f64>], outputs: Vec<i8>, config: NeuralNetConfig) -> Vec<f64> {
    let unique_classifications = outputs
        .iter()
        .collect::<HashSet<_>>();
    let n_classifications = unique_classifications.len();

    let n_folds = config.n_folds;
    let n_epochs = config.n_epochs;
    let learning_rate = config.learning_rate;
    let decay = config.learning_rate_decay;

    let mut neural_network = NeuralNetwork::default();
    neural_network = initialize_neural_network(config, neural_network, dataset[0].len(), n_classifications);

    let folds = cross_validation_split(dataset, n_folds);
    let mut scores: Vec<f64> = Vec::new();

    for (fold_index, fold) in folds.iter().enumerate() {
        println!("Fold: {}", fold_index + 1); 
        
        // Create a training set for all fiolds except the current fold. The current fold is
        // used as the test set.
        let mut training_set = folds.to_vec();
        training_set.remove(fold_index);

        // Flatten the training set from a 3D vector to a 2D vector.
        let flattened_training_set = training_set
            .into_iter()
            .flatten()
            .collect::<Vec<Vec<f64>>>();

        neural_network.train_network(flattened_training_set, learning_rate, decay, n_epochs, n_classifications);

        let mut correct = 0;
        for row in fold.iter() {
            let prediction = neural_network.predict(row.to_vec());
            if prediction == row[row.len() - 1] as usize {
                correct += 1;
            }
        }

        let accuracy = (correct as f64 / fold.len() as f64) * 100.0;
        scores.push(accuracy);
    }

    scores
}