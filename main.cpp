#include "Eigen/Dense"
#include <iostream>
#include <vector>

#define LAYER_DEBUG 1
#define NETWORK_DEBUG 0

double sigmoid(double input) {
	return 1 / (1 + std::exp(-input));
}

double hyperbolic_tanget(double input) {
	return ((std::exp(input) - std::exp(-input)) / (std::exp(input) + std::exp(-input)));
}

Eigen::VectorXd activationFunction(Eigen::VectorXd vector) {
	Eigen::VectorXd tempVec = Eigen::VectorXd::Random(vector.size());
	for (Eigen::Index row = 0; row < vector.size(); row++) {
		tempVec.coeffRef(row) = tanh(vector[row]);
	}
	return tempVec;
}

class Layer {
private:
	Eigen::MatrixXd m_weights;
	Eigen::VectorXd m_output;
	Eigen::VectorXd m_inputs;
public:
	Layer(size_t number_of_inputs, size_t number_of_neurons, size_t number_of_outputs) {

#if NETWORK_DEBUG
		std::cout << "Generating layer with " << number_of_inputs << " inputs and " <<
			number_of_neurons << " neurons and " << number_of_outputs << " outputs " << '\n';
#endif

		m_weights = Eigen::MatrixXd::Random(number_of_neurons, number_of_inputs);
		m_output = Eigen::VectorXd::Random(number_of_outputs);
	}
	Eigen::VectorXd forwardPropagation(Eigen::VectorXd& input) {
		m_inputs = input;

#if NETWORK_DEBUG
		std::cout << "Activation function parameters: Weights: " << m_weights.size() << " Inputs: " << input.size() << '\n';
#endif

		Eigen::VectorXd tempMat = m_weights * input;
		m_output = tempMat = activationFunction(tempMat);
		return tempMat;
	}
	Eigen::VectorXd& getInputs() {
		return m_inputs;
	}
	Eigen::VectorXd& getOutputs() {
		return m_output;
	}
	Eigen::VectorXd getWeights() {
		return m_weights;
	}
	~Layer() = default;
};

class NeuralNetwork {
private:
	std::vector<Layer> m_layers_vector;
	std::vector<uint32_t> m_topology;
	Eigen::VectorXd m_output;

	double m_learning_rate;
	double bias = 0;

public:
	NeuralNetwork(std::vector<uint32_t>& topology, double learning_rate) :
		m_topology{ topology }, m_learning_rate{ learning_rate } {
		for (size_t index = 0; index < m_topology.size(); index++) {
			if (index == 0)
				m_layers_vector.emplace_back(topology[index], topology[index], topology[index + 1]);
			else if (index < (m_topology.size() - 1))
				m_layers_vector.emplace_back(topology[index], topology[index], topology[index + 1]);
			else
				m_layers_vector.emplace_back(topology[index], topology[index], topology.back());
		}
	}
	~NeuralNetwork() = default;

	void forwardPropagation(Eigen::VectorXd& input) {
		std::cout << "InputF: \n" << input << '\n';
		Eigen::VectorXd previous_layer_output = m_layers_vector.front().forwardPropagation(input);

#if LAYER_DEBUG
		std::cout << "\nFirstOutput: \n" << previous_layer_output;
#endif

		for (size_t layer = 1; layer < m_layers_vector.size(); layer++) {

#if LAYER_DEBUG
			std::cout << "\nOutput: \n" << previous_layer_output;
#endif

			previous_layer_output = m_layers_vector[layer].forwardPropagation(previous_layer_output);
		}

#if LAYER_DEBUG
		std::cout << "\nLastOutput: \n" << previous_layer_output;
#endif

		m_output = previous_layer_output;
	}
	void backwardsPropagation(Eigen::VectorXd& answer) {
		std::cout << "\nInputA: \n" << answer << '\n';

		std::vector<Eigen::VectorXd> layers_deltas;
		Eigen::VectorXd temp_last_layer_deltas = Eigen::VectorXd::Random(m_topology.back());
		for (size_t index = 0; index < m_output.size(); index++) {
			temp_last_layer_deltas.coeffRef(index) = m_output[index] - answer[index];
		}
		layers_deltas.push_back(temp_last_layer_deltas);
		std::cout << "\n Deltas: \n" << temp_last_layer_deltas << '\n';

		Eigen::MatrixXd temp_layer_deltas = Eigen::MatrixXd::Random(m_topology.back() * m_topology.front());
		for (size_t index = m_topology.size() - 2; index > 0; index--) {
			//temp_layer_deltas.coeffRef(index) = temp_layer_deltas[index + 1] *
			//(m_layers_vector[index].getWeights().transpose());
		}
		for (size_t layer = 0; layer < m_layers_vector.size(); layer++) {

		}
	}
	Eigen::VectorXd& getOutput() {
		return m_output;
	}
};

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(NULL);

	std::vector<Eigen::VectorXd> inputs;
	std::vector<Eigen::VectorXd> answers;

	Eigen::VectorXd tempA = (Eigen::VectorXd(2) << 1, 1).finished();
	Eigen::VectorXd tempB = (Eigen::VectorXd(2) << 0, 1).finished();

	inputs.push_back(tempA);
	answers.push_back(tempB);

	std::vector<uint32_t> topology = { 2, 2, 2, 2 };
	NeuralNetwork nn(topology, 0.02);

	size_t iterations = 2;
	for (size_t iteration = 0; iteration < iterations; iteration++) {
		for (size_t sample = 0; sample < inputs.size(); sample++) {
			nn.forwardPropagation(inputs[sample]);
			nn.backwardsPropagation(answers[sample]);
		}
	}
}