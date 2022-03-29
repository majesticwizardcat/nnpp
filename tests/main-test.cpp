#include "nnpp.hpp"

#include <iostream>

static const uint SESSIONS = 200000;

int main() {
	std::cout << "Running test Main test..." << '\n';
	const float minValue = -10.0f;
	const float maxValue = 10.0f;
	std::vector<std::vector<uint>> layers = { { 4, 30, 20, 35, 50, 30, 10, 1 } };
	NNPopulation<float> nnp("test", 300, layers, minValue, maxValue);
	nnp.createRandom(minValue, maxValue);
	std::vector<float> input0 = { 0.0f, 0.0f, 0.0f, 0.0f };
	std::vector<float> input1 = { 1.0f, 0.0f, 1.0f, 1.0f };
	std::vector<float> input2 = { 0.0f, 1.0f, 1.0f, 0.0f };
	std::vector<float> input3 = { 1.0f, 1.0f, 1.0f, 1.0f };
	std::vector<float> exp0 = { -10.5f };
	std::vector<float> exp1 = { 2.6f };
	std::vector<float> exp2 = { 3.7f };
	std::vector<float> exp3 = { 4.8f };
	std::vector<TrainerDataSet<float>> tests;
	tests.emplace_back(input0, exp0, 0);
	tests.emplace_back(input1, exp1, 0);
	tests.emplace_back(input2, exp2, 0);
	tests.emplace_back(input3, exp3, 0);

	float lastbest = 1.0f;
	std::cout << "Starting training test" << '\n';
	for (uint i = 0; i < SESSIONS; ++i) {
		SimpleTrainer<float> trainer(1, 4, &nnp, tests);
		if (lastbest != nnp.getMaxScoreNNAi().getScore()) {
			std::cout << "Best score: " << nnp.getMaxScoreNNAi().getScore() << '\n';
			lastbest = nnp.getMaxScoreNNAi().getScore();
		}
		if (i % 1000 == 0) {
			std::cout << "Completed: " << i << " out of: " << SESSIONS << '\n';
		}
		trainer.run();
		nnp.saveToDisk();
		nnp = NNPopulation<float>("test");
	}
	std::cout << "Finished training test" << '\n';

	nnp.printInfo();

	const NNAi<float>& best = nnp.getBestNNAiConstRef();
	std::cout << "Best score: " << best.getScore() << '\n';
	std::cout << "Best score layers: " << '\n';
	best.printLayerSizes();
	std::cout << "Outputs after training: " << '\n';
	for (const auto& t : tests) {
		std::vector<float> out = best.feedAt(t.aiIndex, t.input);
		std::cout << "Input: ";
		for (float i : t.input) {
			std::cout << i << ' ';
		}
		std::cout << "Output: ";
		for (float o : out) {
			std::cout << o << ' ';
		}
		std::cout << "Expected: ";
		for (float e : t.expected) {
			std::cout << e << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
	std::cout << "Test completed!" << '\n';
}
