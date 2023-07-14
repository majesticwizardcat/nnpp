#include "nnpp.hpp"

#include <iostream>
#include <thread>
#include <chrono>

class TestTrainer : public NNPPTrainer<float> {
public:
	TestTrainer(uint sessions, uint threads, NNPopulation<float>* const population)
		: NNPPTrainer<float>(sessions, threads, population, getDefaultEvolutionInfoFloat()) { }

	inline uint getSessionsFinished() const {
		return m_sessionsFinished;
	}
	
protected:
	std::vector<NNPPTrainingUpdate<float>> runSession(NeuronBuffer<float>& threadLocalNeuronBuffer) {
		std::vector<NNPPTrainingUpdate<float>> results;
		std::random_device dev;
		std::uniform_int_distribution<uint> dist(1, 5000);
		std::this_thread::sleep_for(std::chrono::milliseconds(dist(dev)));
		m_sessionsFinished++;
		return results;
	}

	uint sessionsTillEvolution() const {
		assert(tillNextGen() >= m_trainee->getSessionsTrainedThisGen());
		return tillNextGen() - m_trainee->getSessionsTrainedThisGen();
	}

private:
	std::atomic<uint> m_sessionsFinished;

	uint tillNextGen() const {
		return 100;
	}
};

int main() {
	std::cout << "Starting training test.." << '\n';
	std::chrono::time_point start = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<uint>> layers = {{ 2, 3, 1 }};
	NNPopulation<float> population("trainer-test", 100, layers);
	population.createRandom(0.0f, 1.0f);
	TestTrainer trainer(1053, 7, &population);
	trainer.run(true);
	assert(population.getGenerartion() == 10);
	assert(population.getSessionsTrained() == 1053);
	assert(population.getSessionsTrainedThisGen() == 53);
	assert(trainer.getSessionsFinished() == 1053);
	std::chrono::duration<float, std::milli> dur = std::chrono::high_resolution_clock::now() - start;
	std::cout << "Training test completed! Time: " << (dur.count() / 1000) << '\n';
}
