#include "nnpp.hpp"

int main() {
	std::cout << "Starting NNAi test" << '\n';
	std::vector<std::vector<uint>> layers = { { 1, 2, 1 }, { 2, 3, 3 }, { 5, 2, 3, 1 } };
	NNAi<float> nnai(layers);

	nnai.initRandomUniform(0.0f, 1.0f);
	nnai.initVal(1.0f);
	nnai.initBiasesVal(0.0f);

	NNPPStackVector<float> out;
	NNPPStackVector<float> input0 = { 1.0f };
	NNPPStackVector<float> input1 = { 1.0f, 1.0f };
	NNPPStackVector<float> input2 = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
	
	std::cout << "Testing feeding..." << '\n';
	out = nnai.feedAt(0, input0);

	for (float o : out) {
		assert(o == 2.0f);
	}

	out = nnai.feedAt(1, input1);

	for (float o : out) {
		assert(o == 6.0f);
	}

	out = nnai.feedAt(2, input2);

	for (float o : out) {
		assert(o == 30.0f);
	}

	assert(nnai.saveToFile("nnai-test.nnpp"));
	assert(nnai.loadFromFile("nnai-test.nnpp"));

	std::cout << "Retesting feeding after save/load..." << '\n';
	out = nnai.feedAt(0, input0);

	for (float o : out) {
		assert(o == 2.0f);
	}

	out = nnai.feedAt(1, input1);

	for (float o : out) {
		assert(o == 6.0f);
	}

	out = nnai.feedAt(2, input2);

	for (float o : out) {
		assert(o == 30.0f);
	}

	std::cout << "Done!" << '\n';
}
