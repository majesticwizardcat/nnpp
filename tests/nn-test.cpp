#include "nnpp.hpp"

int main() {
	std::cout << "Running NN test... " << '\n';
	std::vector<uint> layers = { 2, 3, 2, 1, 2, 3, 1, 4,
		3, 2, 1, 1, 1, 1, 10, 100, 2, 1, 1, 1, 1, 1, 5, 5, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10 };
	NeuralNetwork<float> nn(layers);

	std::cout << "NN has data size: " << nn.getDataSize() << '\n';
	std::cout << "Initiating data to 1.0f" << '\n';
	nn.initDataVal(1.0f);
	
	std::vector<float> input = { 1.0f, 1.0f };
	std::cout << "Fedding input... " << '\n';
	auto out = nn.feed(input);
	for (float o : out) {
		assert(o == 86400000.0f); 
	}
	std::cout << "Passed!" << '\n';

	std::cout << "Saving to disk..." << '\n';
	assert(nn.saveToFile("nn-test.nnpp"));
	std::cout << "Loading from disk..." << '\n';
	assert(nn.loadFromFile("nn-test.nnpp"));

	std::cout << "Rerunning input test: " << '\n';
	std::cout << "Fedding input... " << '\n';
	out = nn.feed(input);
	for (float o : out) {
		assert(o == 86400000.0f); 
	}
	std::cout << "Passed!" << '\n';
	std::cout << "Done!" << '\n';
}
