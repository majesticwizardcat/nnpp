#include "nnpp.hpp"

#include <iostream>
#include <chrono>

static const uint SESSIONS = 200000;

int main() {
	assert(false); // Do not do perf tests on Debug

	const float minValue = -10.0f;
	const float maxValue = 10.0f;
	std::vector<std::vector<uint>> layers = { { 4, 300, 200, 350, 500, 300, 100, 1 } };
	NNPopulation<float> nnp("test", 300, layers, minValue, maxValue);
	nnp.createRandom(minValue, maxValue);
	NNPPStackVector<float> input0 = { 0.0f, 0.0f, 0.0f, 0.0f };
	NNPPStackVector<float> input1 = { 1.0f, 0.0f, 1.0f, 1.0f };
	NNPPStackVector<float> input2 = { 0.0f, 1.0f, 1.0f, 0.0f };
	NNPPStackVector<float> input3 = { 1.0f, 1.0f, 1.0f, 1.0f };
	NNPPStackVector<float> exp0 = { -10.5f };
	NNPPStackVector<float> exp1 = { 2.6f };
	NNPPStackVector<float> exp2 = { 3.7f };
	NNPPStackVector<float> exp3 = { 4.8f };
	std::vector<TrainerDataSet<float>> tests;

	tests.emplace_back(input0, exp0, 0);
	tests.emplace_back(input1, exp1, 0);
	tests.emplace_back(input2, exp2, 0);
	tests.emplace_back(input3, exp3, 0);

	std::cout << "Starting perf test" << '\n';
	auto start = std::chrono::high_resolution_clock::now();
	SimpleTrainer<float> trainer(SESSIONS, 1, &nnp, tests);
	trainer.run(false, false);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<float> time = end - start;
	std::cout << "Finished perf test. Time taken: " << time.count() << " seconds." << '\n';
}
