#include "nnpp.hpp"

int main() {
	std::cout << "Running perceptron xor test" << '\n';
	std::vector<uint> layer = { 2, 3, 1 };
	NeuralNetwork<float> nn(layer);

	std::vector<float> data = { 1.0f, 2.0f, 3.0f,
								-1.0f, 0.5f, 0.0f,
								1.5f, 4.0f, -1.0f };
	nn.initData(data);
	nn.printData();
	std::cout << "---------" << '\n';

	std::vector<float> in0 = { 0.0f, 0.0f };
	std::vector<float> in1 = { 1.0f, 0.0f };
	std::vector<float> in2 = { 0.0f, 1.0f };
	std::vector<float> in3 = { 1.0f, 1.0f };
	auto out = nn.feed(in0);
	std::cout << "Out for: " << in0[0] << ", " << in0[1] << " -> " << out[0] << '\n';
	assert(out[0] == 0.0f);
	out = nn.feed(in1);
	std::cout << "Out for: " << in1[0] << ", " << in1[1] << " -> " << out[0] << '\n';
	assert(out[0] == 6.5f);
	out = nn.feed(in2);
	std::cout << "Out for: " << in2[0] << ", " << in2[1] << " -> " << out[0] << '\n';
	assert(out[0] == 0.5f);
	out = nn.feed(in3);
	std::cout << "Out for: " << in3[0] << ", " << in3[1] << " -> " << out[0] << '\n';
	assert(out[0] == 7.0f);
	std::cout << "Done!" << '\n';
}
