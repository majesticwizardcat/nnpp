#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include <atomic>
#include <utility>
#include <mutex>
#include <array>
#include <cassert>
#include <string>

#include "sauce.hpp"
#include "pcg_random.hpp"

namespace nnpp {

static constexpr uint32_t VERSION_NN = 1;
static constexpr const char* HEADER_STR_NN = "NAPPNN";
static constexpr uint32_t VERSION_NNAI = 1;
static constexpr const char* HEADER_STR_NNAI = "NNPPNNAI";
static constexpr uint32_t VERSION_NNPP = 1;
static constexpr const char* HEADER_STR_NNPP = "NNPP";

static constexpr uint64_t MAX_INPUT_OUTPUT_NEURONS = 1024;
static constexpr uint32_t MAX_NEURONS_PER_LAYER = 1 << 16;
static constexpr float DEFAULT_WEIGHT_MUTATION_CHANCE = 0.05f;
static constexpr uint64_t DEFAULT_MAX_LAYER_MUTATION = 5;
static constexpr float DEFAULT_LAYER_MUTATION_CHANCE = 0.3f;
static constexpr float DEFAULT_LAYER_ADDITION_CHANCE = 0.5f;
static constexpr float TARGET_DECREASE_RATE = 0.0005f;
static constexpr float DEFAULT_CHILD_REGRESSION_PERCENTAGE = 0.95f;
static constexpr uint64_t DEFUALT_MIN_TRAINING_SESSIONS_REQUIRED = 1;
static constexpr float DEFAULT_MIN_MUTATION_VALUE_FLOAT = -1.0f;
static constexpr float DEFAULT_MAX_MUTATION_VALUE_FLOAT = 1.0f;

inline constexpr float normalize(const float value, const float min, const float max) {
	return (value - min) / (max - min);
}

template <typename T> struct EvolutionInfo {
	T minMutationValue;
	T maxMutationValue;
	float childRegressionPercentage;
	float weightMutationChance;
	float layerMutationChance;
	float layerAdditionChance;
	uint64_t maxLayersMutation;
	uint64_t minTrainingSessionsRequired;
};

inline static constexpr EvolutionInfo<float> getDefaultEvolutionInfoFloat() {
	return EvolutionInfo<float> {
		  .minMutationValue = DEFAULT_MIN_MUTATION_VALUE_FLOAT
		, .maxMutationValue = DEFAULT_MAX_MUTATION_VALUE_FLOAT
		, .childRegressionPercentage = DEFAULT_CHILD_REGRESSION_PERCENTAGE
		, .weightMutationChance = DEFAULT_WEIGHT_MUTATION_CHANCE
		, .layerMutationChance = DEFAULT_LAYER_MUTATION_CHANCE
		, .layerAdditionChance = DEFAULT_LAYER_ADDITION_CHANCE
		, .maxLayersMutation = DEFAULT_MAX_LAYER_MUTATION
		, .minTrainingSessionsRequired = DEFUALT_MIN_TRAINING_SESSIONS_REQUIRED
	};
}

template <typename T> using NNPPStackVector = sauce::StaticVector<T, MAX_INPUT_OUTPUT_NEURONS>;

// First half of the array is for inputs for the next layer and the other half is for the outputs
template <typename T> using NeuronBuffer = std::vector<T>;

template <typename T> static constexpr inline NeuronBuffer<T> allocNeuronBuffer() {
	return NeuronBuffer<T>(MAX_NEURONS_PER_LAYER * 2);
}

template <typename T> class NeuralNetwork {
public:
	NeuralNetwork() = delete;

	NeuralNetwork(const NeuralNetwork& other) = delete;

	constexpr NeuralNetwork(const std::vector<uint32_t>& layers)
			: m_layerSizes(layers)
			, m_data(nullptr)
			, m_neuronBiases(nullptr) {
		if (layers.size() < 3) {
			std::cout << "Cannot init nn with less than 3 layers" << '\n';
			assert(false);
			return;
		}
		assert(layers[0] <= MAX_INPUT_OUTPUT_NEURONS);
		assert(layers.back() <= MAX_INPUT_OUTPUT_NEURONS);
		initializeNetworkData();
		m_data = std::make_unique<T[]>(getDataSize());
		m_neuronBiases = std::make_unique<T[]>(getNumOfTotalNeurons());
	}

	NeuralNetwork(const std::string_view location)
			: m_data(nullptr)
			, m_neuronBiases(nullptr) {
		if (!loadFromFile(location)) {
			std::cout << "Could not load from file" << '\n';
			assert(false);
		}
	}

	NeuralNetwork(sauce::BufferedFileReader& reader)
			: m_data(nullptr)
			, m_neuronBiases(nullptr) {
		if (!load(reader)) {
			std::cout << "Could not read from stream" << '\n';
			assert(false);
		}
	}

	NeuralNetwork(NeuralNetwork&& other)
			: m_layerSizes(std::move(other.m_layerSizes))
			, m_data(std::move(other.m_data))
			, m_neuronBiases(std::move(other.m_neuronBiases))
			, m_dataSizePerLayer(std::move(other.m_dataSizePerLayer))
			, m_numOfNeuronsPerLayer(std::move(other.m_numOfNeuronsPerLayer)) { }

	inline constexpr NeuralNetwork(const NeuralNetwork<T>& n0, const NeuralNetwork<T>& n1, const EvolutionInfo<T>& evolutionInfo) {
		initFromParents(n0, n1, evolutionInfo);
	}

	constexpr void initFromParents(const NeuralNetwork<T>& n0, const NeuralNetwork<T>& n1, const EvolutionInfo<T>& evolutionInfo) {
		assert(n0.m_layerSizes.size() == n1.m_layerSizes.size());
		clearAll();

		std::uniform_real_distribution<float> realDist(0.0f, 1.0f);
		std::uniform_real_distribution<T> mutationValueDist(evolutionInfo.minMutationValue, evolutionInfo.maxMutationValue);
		std::uniform_int_distribution<uint> layerMutation(0, evolutionInfo.maxLayersMutation);
		pcg_extras::seed_seq_from<std::random_device> seed;
		pcg32_fast dev(seed);

		std::vector<uint32_t> minLayerSizes(n0.m_layerSizes.size());
		m_layerSizes.resize(n0.m_layerSizes.size());
		for (uint32_t i = 0; i < n0.m_layerSizes.size(); ++i) {
			if (i > 0 && i < n0.m_layerSizes.size() - 1) {
				const uint32_t extras = realDist(dev) < evolutionInfo.layerMutationChance ? layerMutation(dev) : 0;
				m_layerSizes[i] = (n0.m_layerSizes[i] + n1.m_layerSizes[i]) / 2;
				if (realDist(dev) < evolutionInfo.layerAdditionChance) {
					m_layerSizes[i] = std::min(MAX_NEURONS_PER_LAYER, m_layerSizes[i] + extras);
				}
				else if (m_layerSizes[i] > extras) {
					m_layerSizes[i] -= extras;
				}
				else {
					m_layerSizes[i] = 1;
				}
			}
			else {
				m_layerSizes[i] = n0.m_layerSizes[i];
			}
			minLayerSizes[i] = std::min(m_layerSizes[i], std::min(n0.m_layerSizes[i], n1.m_layerSizes[i]));
		}

		initializeNetworkData();
		m_data = std::make_unique<T[]>(getDataSize());
		m_neuronBiases = std::make_unique<T[]>(getNumOfTotalNeurons());

		for (uint32_t tl = 1; tl < m_layerSizes.size(); ++tl) {
			for (uint32_t tn = 0; tn < m_layerSizes[tl]; ++tn) {
				for (uint32_t fn = 0; fn < m_layerSizes[tl - 1]; ++fn) {
					const float mutation = realDist(dev);
					T weightValue;
					if (mutation < evolutionInfo.weightMutationChance || tn >= minLayerSizes[tl] || fn >= minLayerSizes[tl - 1]) {
						weightValue = mutationValueDist(dev);
					}
					else {
						weightValue = realDist(dev) < 0.5f ? n0.weightAt(fn, tn, tl) : n1.weightAt(fn, tn, tl);
					}
					setWeightAt(fn, tn, tl, weightValue);
				}
			}
		}

		for (uint32_t l = 0; l < m_layerSizes.size(); ++l) {
			for (uint32_t n = 0; n < m_layerSizes[l]; ++n) {
				const float mutation = realDist(dev);
				T neuronValue;
				if (mutation < evolutionInfo.weightMutationChance || n >= minLayerSizes[l]) {
					neuronValue = mutationValueDist(dev);
				}
				else {
					neuronValue = realDist(dev) < 0.5f ? n0.neuronBiasAt(n, l) : n1.neuronBiasAt(n, l);
				}
				setNeuronBiasAt(n, l, neuronValue);
			}
		}
	}

	inline constexpr void initDataVal(const T& val) {
		for (uint64_t i = 0; i < getDataSize(); ++i) {
			m_data[i] = val;
		}
		initBiasesVal(val);
	}

	inline constexpr void initBiasesVal(const T& val) {
		const uint32_t biasesNum = getNumOfTotalNeurons();
		for (uint32_t i = 0; i < biasesNum; ++i) {
			m_neuronBiases[i] = val;
		}
	}

	inline constexpr void initData(const std::vector<T>& data, const std::vector<T>& biases) {
		assert(data.size() == getDataSize());
		assert(biases.size() == getNumOfTotalNeurons());
		for (uint64_t i = 0; i < getDataSize(); ++i) {
			m_data[i] = data[i];
		}

		const uint32_t biasesNum = getNumOfTotalNeurons();
		for (uint32_t i = 0; i < biasesNum; ++i) {
			m_neuronBiases[i] = biases[i];
		}
	}

	inline constexpr uint64_t getDataSize() const {
		return m_dataSizePerLayer.back();
	}

	inline constexpr uint64_t getDataSizeForLayer(const uint64_t layer) const {
		assert(layer < m_dataSizePerLayer.size());
		return m_dataSizePerLayer[layer];
	}

	inline constexpr uint64_t getNumOfTotalNeurons() const {
		assert(m_numOfNeuronsPerLayer.back() == std::accumulate(m_layerSizes.begin(), m_layerSizes.end(), 0ull));
		return m_numOfNeuronsPerLayer.back();
	}

	inline constexpr void randomizeDataUniform(const T& min, const T& max) {
		std::uniform_real_distribution<T> dist(min, max);
		pcg_extras::seed_seq_from<std::random_device> seed;
		pcg32_fast dev(seed);
		for (uint64_t i = 0; i < getDataSize(); ++i) {
			m_data[i] = dist(dev);
		}

		const uint32_t biasesNum = getNumOfTotalNeurons();
		for (uint32_t i = 0; i < biasesNum; ++i) {
			m_neuronBiases[i] = dist(dev);
		}
	}

	bool save(sauce::BufferedFileWriter& writer) const {
		if (!writer.isOk()) {
			return false;
		}

		// Save header
		static constexpr std::string_view header = HEADER_STR_NN;
		writer.write(header);
		writer.write(VERSION_NN);

		// Save data
		const uint32_t layers = m_layerSizes.size();
		writer.write(layers);

		for (const uint32_t l : m_layerSizes) {
			writer.write(l);
		}

		writer.writeBytes(m_data.get(), sizeof(T) * getDataSize());
		writer.writeBytes(m_neuronBiases.get(), sizeof(T) * getNumOfTotalNeurons());
		return writer.flush();
	}

	inline bool saveToDisk(const std::string_view fileName) const {
		assert(m_data && getDataSize() > 0);
		sauce::BufferedFileWriter writer(fileName);
		return save(writer);
	}

	bool load(sauce::BufferedFileReader& reader) {
		clearAll();

		if (!reader.isOk()) {
			return false;
		}

		static constexpr std::string_view header = HEADER_STR_NN;
		static constexpr uint64_t headerLen = header.size();
		char headerBuffer[headerLen];

		reader.readBytes(headerBuffer, headerLen);
		if (strncmp(headerBuffer, header.data(), headerLen) != 0) {
			std::cout << "Invalid header when reading neural network: " << headerBuffer << " != " << header << '\n';
			return false;
		}

		uint32_t headerVersion = 0;
		reader.read(headerVersion);

		if (headerVersion == 0 || headerVersion > VERSION_NN) {
			std::cout << "Invalid header version when reading neural network: " << headerVersion << '\n';
			return false;
		}

		uint32_t layers;
		reader.read(layers);

		for (uint32_t l = 0; l < layers; ++l) {
			uint32_t cur;
			reader.read(cur);
			m_layerSizes.push_back(cur);
		}

		initializeNetworkData();
		const uint64_t biases = getNumOfTotalNeurons();
		m_data = std::make_unique<T[]>(getDataSize());
		m_neuronBiases = std::make_unique<T[]>(biases);

		reader.readBytes(m_data.get(), sizeof(T) * getDataSize());
		return reader.readBytes(m_neuronBiases.get(), sizeof(T) * biases);
	}

	bool loadFromFile(const std::string_view location) {
		sauce::BufferedFileReader reader(location);
		return load(reader);
	}

	void clearAll() {
		m_layerSizes.clear();
		m_data.release();
		m_neuronBiases.release();
		m_dataSizePerLayer.clear();
		m_numOfNeuronsPerLayer.clear();
	}

	constexpr NNPPStackVector<T> feed(const NNPPStackVector<T>& input, NeuronBuffer<T>& neuronBuffer) const {
		latchInputs(input, neuronBuffer);
		propagate(neuronBuffer);
		return getOutputs(neuronBuffer);
	}

	constexpr bool operator==(const NeuralNetwork& other) const {
		if (getDataSize() != other.getDataSize()
			|| m_layerSizes != other.m_layerSizes) {
			return false;
		}

		for (uint64_t i = 0; i < getDataSize(); ++i) {
			if (m_data[i] != other.m_data[i]) {
				return false;
			}
		}
		return true;
	}

	inline void printLayerSizes() const {
		std::cout << "{ ";
		for (uint32_t l : m_layerSizes) {
			std::cout << l << ' ';
		}
		std::cout << "}" << '\n';
	}

	inline void printData() const {
		for (uint32_t i = 1; i < m_layerSizes.size(); ++i) {
			for (uint32_t n = 0; n < m_layerSizes[i]; ++n) {
				printWeightsAt(n, i);
			}
		}
	}

private:
	std::vector<uint32_t> m_layerSizes;
	std::unique_ptr<T[]> m_data;
	std::unique_ptr<T[]> m_neuronBiases;
	std::vector<uint64_t> m_dataSizePerLayer;
	std::vector<uint32_t> m_numOfNeuronsPerLayer;

	inline constexpr void initializeNetworkData() {
		calculateDataSizes();
		calculateNeuronSizes();
	}

	inline constexpr uint32_t weightIndex(const uint32_t fromNeuron, const uint32_t toNeuron, const uint32_t toLayer) const {
		assert(toLayer > 0);
		assert(toLayer < m_layerSizes.size());
		assert(getDataSizeForLayer(toLayer)
			+ fromNeuron * m_layerSizes[toLayer]
			+ toNeuron < getDataSize());
		return getDataSizeForLayer(toLayer)
			+ fromNeuron * m_layerSizes[toLayer]
			+ toNeuron;
	}

	inline constexpr uint32_t neuronBiasIndex(const uint32_t neuron, const uint32_t layer) const {
		assert(layer >= 0);
		assert(layer < m_numOfNeuronsPerLayer.size());
		assert(m_numOfNeuronsPerLayer[layer] + neuron < getNumOfTotalNeurons());
		return m_numOfNeuronsPerLayer[layer] + neuron;
	}

	inline constexpr const T& weightAt(const uint32_t fromNeuron, const uint32_t toNeuron, const uint32_t toLayer) const {
		return m_data[weightIndex(fromNeuron, toNeuron, toLayer)];
	}

	inline constexpr const T& neuronBiasAt(const uint32_t neuron, const uint32_t layer) const {
		return m_neuronBiases[neuronBiasIndex(neuron, layer)];
	}

	inline constexpr void setWeightAt(const uint32_t fromNeuron, const uint32_t toNeuron, const uint32_t toLayer, const T& value) {
		m_data[weightIndex(fromNeuron, toNeuron, toLayer)] = value;
	}

	inline constexpr void setNeuronBiasAt(const uint32_t neuron, const uint32_t layer, const T& value) {
		m_neuronBiases[neuronBiasIndex(neuron, layer)] = value;
	}

	inline constexpr void latchInputs(const NNPPStackVector<T>& inputs, NeuronBuffer<T>& neurons) const {
		assert(inputs.size() == *m_layerSizes.begin());
		assert(neurons.size() >= inputs.size() * 2);
		for (uint32_t i = 0; i < inputs.size(); ++i) {
			neurons[i] = inputs[i] + neuronBiasAt(i, 0);
		}
	}

	inline constexpr void propagate(NeuronBuffer<T>& neurons) const {
		// half size is where the array for copy dest begins
		const uint64_t halfSize = (neurons.size() >> 1); // size / 2
		for (uint32_t l = 1; l < m_layerSizes.size(); ++l) {
			assert(neurons.size() >= m_layerSizes[l] * 2);
			std::fill_n(neurons.begin() + halfSize, m_layerSizes[l], 0);
			for (uint fromNeuron = 0; fromNeuron < m_layerSizes[l - 1]; ++fromNeuron) {
				for (uint toNeuron = 0; toNeuron < m_layerSizes[l]; ++toNeuron) {
					 neurons[halfSize + toNeuron] += neurons[fromNeuron] * weightAt(fromNeuron, toNeuron, l) + neuronBiasAt(toNeuron, l);
				}
			}

			std::copy(neurons.begin() + halfSize,
					  neurons.begin() + halfSize + m_layerSizes[l],
					  neurons.begin());
		}
	}

	void printWeightsAt(uint32_t neuron, uint32_t layer) const {
		assert(layer >= 1);
		for (uint32_t n = 0; n < m_layerSizes[layer - 1]; ++n) {
			std::cout << weightAt(n, neuron, layer) << ' ';
		}
		std::cout << '\n';
	}

	inline constexpr NNPPStackVector<T> getOutputs(const NeuronBuffer<T>& neurons) const {
		NNPPStackVector<T> out;
		for (uint i = 0; i < m_layerSizes.back(); ++i) {
			out[i] = neurons[i];
		}
		return out;
	}

	inline constexpr void calculateDataSizes() {
		assert(m_layerSizes.size() > 1);
		auto calculateDataSizeForLayer = [this](const uint64_t layer) {
			assert(layer <= m_layerSizes.size());
			uint64_t size = 0;
			for (uint64_t i = 1; i < layer; ++i) {
				size += m_layerSizes[i - 1] * m_layerSizes[i];
			}
			return size;
		};

		m_dataSizePerLayer.resize(m_layerSizes.size() + 1);
		for (uint64_t i = 0; i <= m_layerSizes.size(); ++i) {
			m_dataSizePerLayer[i] = calculateDataSizeForLayer(i);
		}
	}

	inline constexpr void calculateNeuronSizes() {
		m_numOfNeuronsPerLayer.resize(m_layerSizes.size() + 1);
		for (uint32_t i = 0; i < m_layerSizes.size(); ++i) {
			m_numOfNeuronsPerLayer[i] = std::accumulate(m_layerSizes.begin(), m_layerSizes.begin() + i, 0ull);
		}
		m_numOfNeuronsPerLayer[m_layerSizes.size()] = std::accumulate(m_layerSizes.begin(), m_layerSizes.end(), 0ull);
		assert(getNumOfTotalNeurons() == std::accumulate(m_layerSizes.begin(), m_layerSizes.end(), 0ull));
	}
};

template <typename T> class NNAi {
public:
	NNAi() = delete;
	NNAi(const NNAi& other) = delete;

	constexpr NNAi(const uint64_t id, const std::vector<std::vector<uint>>& layers)
			: m_id(id)
			, m_score(0.0f)
			, m_sessionsTrained(0) {
		for (const auto& ls : layers) {
			m_networks.emplace_back(ls);
		}
	}

	NNAi(sauce::BufferedFileReader& reader) {
		if (!load(reader)) {
			std::cout << "Could not load NNAi from stream" << '\n';
			assert(false);
		}
	}

	NNAi(NNAi&& other)
			: m_id(std::move(other.m_id))
			, m_networks(std::move(other.m_networks))
			, m_score(std::move(other.m_score))
			, m_sessionsTrained(std::move(other.m_sessionsTrained)) { }

	inline NNAi& operator=(NNAi&& other) {
		m_id = std::move(other.m_id);
		m_networks = std::move(other.m_networks);
		m_score = std::move(other.m_score);
		m_sessionsTrained = std::move(other.m_sessionsTrained);
		return *this;
	}

	constexpr NNAi(const uint64_t id, const NNAi& nnai0, const NNAi& nnai1, const EvolutionInfo<T>& evolutionInfo) {
		initFromParents(id, nnai0, nnai1, evolutionInfo);
	}

	constexpr void initRandomUniform(const T& min, const T& max) {
		for (auto& nn : m_networks) {
			nn.randomizeDataUniform(min, max);
		}
	}

	constexpr void initVal(const T& val) {
		for (auto& nn : m_networks) {
			nn.initDataVal(val);
		}
	}

	constexpr void initBiasesVal(const T& val) {
		for (auto& nn : m_networks) {
			nn.initBiasesVal(val);
		}
	}

	constexpr void initFromParents(const uint64_t id, const NNAi& nnai0, const NNAi& nnai1, const EvolutionInfo<T>& evolutionInfo) {
		assert(nnai0.getNetworksNumber() == nnai1.getNetworksNumber());
		clearAll();
		m_id = id;
		m_score = (nnai0.getScore() + nnai1.getScore()) * 0.5f * evolutionInfo.childRegressionPercentage;
		for (uint32_t i = 0; i < nnai0.getNetworksNumber(); ++i) {
			m_networks.emplace_back(nnai0.getConstRefAt(i), nnai1.getConstRefAt(i), evolutionInfo);
		}
	}

	inline bool saveToFile(const std::string_view location) const {
		sauce::BufferedFileWriter writer(location);
		return save(writer);
	}

	bool save(sauce::BufferedFileWriter& writer) const {
		if (!writer.isOk()) {
			return false;
		}

		static constexpr std::string_view headerStr = HEADER_STR_NNAI;
		writer.write(headerStr);
		writer.write(VERSION_NNAI);

		const uint32_t nets = m_networks.size();
		writer.write(m_id);
		writer.write(nets);
		writer.write(m_sessionsTrained);
		writer.write(m_score);

		for (const auto& nn : m_networks) {
			if (!nn.save(writer)) {
				return false;
			}
		}

		return writer.flush();
	}

	bool loadFromFile(const std::string_view location) {
		sauce::BufferedFileReader reader(location);
		return load(reader);
	}

	bool load(sauce::BufferedFileReader& reader) {
		clearAll();
		if (!reader.isOk()) {
			return false;
		}

		static constexpr std::string_view headerStr = HEADER_STR_NNAI;
		static constexpr uint64_t headerLen = headerStr.size();

		char headerBuffer[headerLen];
		reader.readBytes(headerBuffer, headerLen);

		if (strncmp(headerBuffer, headerStr.data(), headerLen) != 0) {
			std::cout << "Invalid header reading NNAi: " << headerBuffer << '\n';
			return false;
		}

		uint32_t headerVersion = 0;
		reader.read(headerVersion);

		if (headerVersion == 0 || headerVersion > VERSION_NNAI) {
			std::cout << "Invalid version reading NNAi: " << headerVersion << '\n';
			return false;
		}

		uint32_t nets;
		reader.read(m_id);
		reader.read(nets);
		reader.read(m_sessionsTrained);
		reader.read(m_score);

		for (uint32_t i = 0; i < nets; ++i) {
			m_networks.emplace_back(reader);
		}

		return reader.isOk();
	}

	void clearAll() {
		m_id = 0;
		m_score = 0.0f;
		m_sessionsTrained = 0;
		m_networks.clear();
	}

	inline constexpr NNPPStackVector<T> feedAt(const uint32_t index, const NNPPStackVector<T>& input, NeuronBuffer<T>& neuronBuffer) const {
		assert(index < m_networks.size());
		return m_networks[index].feed(input, neuronBuffer);
	}

	inline constexpr const NeuralNetwork<T>& getConstRefAt(const uint32_t index) const {
		assert(index < m_networks.size());
		return m_networks[index];
	}

	inline constexpr uint64_t getID() const {
		return m_id;
	}

	inline constexpr uint64_t getNetworksNumber() const {
		return m_networks.size();
	}

	inline constexpr uint64_t getSessionsTrained() const {
		return m_sessionsTrained;
	}

	inline constexpr void sessionCompleted() {
		m_sessionsTrained++;
	}

	inline constexpr void updateScoreReplace(float newScore) {
		m_score = newScore;
	}

	inline constexpr void updateScoreDelta(float deltaScore) {
		m_score += deltaScore;
	}

	inline constexpr float getScore() const {
		return m_score;
	}

	inline constexpr float getAvgScore() const {
		return m_sessionsTrained == 0 ? 0.0f : m_score / static_cast<float>(m_sessionsTrained);
	}

	inline constexpr bool operator<(const NNAi& other) const {
		return m_score < other.m_score;
	}

	inline constexpr bool operator>(const NNAi& other) const {
		return m_score > other.m_score;
	}

	void printLayerSizes() const {
		for (const auto& nn : m_networks) {
			nn.printLayerSizes();
		}
	}

	void printData() const {
		for (const auto& nn: m_networks) {
			nn.printData();
		}
	}

private:
	uint64_t m_id;
	std::vector<NeuralNetwork<T>> m_networks;
	float m_score;
	uint32_t m_sessionsTrained;
};

template <typename T> class NNPopulation {
public:
	NNPopulation() = delete;
	NNPopulation(const NNPopulation& other) = delete;

	constexpr NNPopulation(const std::string_view name, const uint32_t size, const std::vector<std::vector<uint32_t>>& layers)
			: m_generation(0)
			, m_sessionsTrained(0)
			, m_sessionsTrainedThisGen(0)
			, m_name(name)
			, m_nextID(0) {
		createPopulation(size, layers);
	}

	constexpr NNPopulation(const std::string_view name)
			: m_name(name)
			, m_nextID(0) {
		loadFromDisk(name);
	}

	NNPopulation(NNPopulation&& other)
			: m_population(std::move(other.m_population))
			, m_generation(std::move(other.m_generation))
			, m_sessionsTrained(std::move(other.m_sessionsTrained))
			, m_name(std::move(other.m_name))
			, m_nextID(std::move(other.m_nextID)) { }

	inline NNPopulation& operator=(NNPopulation&& other) {
		m_population = std::move(other.m_population);
		m_generation = std::move(other.m_generation);
		m_sessionsTrained = std::move(other.m_sessionsTrained);
		m_name = std::move(other.m_name);
		m_nextID = std::move(other.m_nextID);
		return *this;
	}

	inline constexpr void createRandom(const T& min, const T& max) {
		for (auto& nnai : m_population) {
			nnai.initRandomUniform(min, max);
		}
	}

	bool saveToDisk() const {
		sauce::BufferedFileWriter writer(m_name);
		if (!writer.isOk()) {
			return false;
		}

		static constexpr std::string_view headerStr = HEADER_STR_NNPP;
		writer.write(headerStr);
		writer.write(VERSION_NNPP);

		const uint32_t size = m_population.size();
		writer.write(size);
		writer.write(m_generation);
		writer.write(m_sessionsTrained);
		writer.write(m_sessionsTrainedThisGen);

		for (const auto& nn : m_population) {
			if (!nn.save(writer)) {
				return false;
			}
		}

		return writer.flush();
	}

	bool loadFromDisk(const std::string_view location) {
		clearAll();
		m_name = location;

		sauce::BufferedFileReader reader(location);
		if (!reader.isOk()) {
			return false;
		}

		static constexpr std::string_view headerStr = HEADER_STR_NNPP;
		static constexpr uint64_t headerLen = headerStr.size();

		char headerBuffer[headerLen];
		reader.readBytes(headerBuffer, headerLen);

		if (strncmp(headerBuffer, headerStr.data(), headerLen) != 0) {
			std::cout << "Invalid header loading NNPP: " << headerBuffer << '\n';
			return false;
		}

		uint32_t headerVersion = 0;
		reader.read(headerVersion);

		if (headerVersion == 0 || headerVersion > VERSION_NNPP) {
			std::cout << "Invalid header version loading NNPP: " << headerVersion << '\n';
			return false;
		}

		uint32_t size;
		reader.read(size);
		reader.read(m_generation);
		reader.read(m_sessionsTrained);
		reader.read(m_sessionsTrainedThisGen);

		for (uint32_t i = 0; i < size; ++i) {
			m_population.emplace_back(reader);
		}

		return reader.isOk();
	}

	void clearAll() {
		m_name.clear();
		m_population.clear();
		m_generation = 0;
		m_sessionsTrained = 0;
		m_nextID = 0;
	}

	inline constexpr void evolutionCompleted() {
		m_generation++;
		m_sessionsTrainedThisGen = 0;
	}

	inline constexpr void trainSessionCompleted() {
		m_sessionsTrained++;
		m_sessionsTrainedThisGen++;
	}

	inline constexpr uint64_t getGenerartion() const {
		return m_generation;
	}
	
	inline constexpr uint64_t getSessionsTrained() const {
		return m_sessionsTrained;
	}

	inline constexpr uint64_t getSessionsTrainedThisGen() const {
		return m_sessionsTrainedThisGen;
	}

	inline constexpr uint64_t getPopulationSize() const {
		return m_population.size();
	}

	inline constexpr NNAi<T>& getMinScoreNNAi() {
		return *std::min_element(m_population.begin(), m_population.end());
	}

	inline constexpr NNAi<T>& getMaxScoreNNAi() {
		return *std::max_element(m_population.begin(), m_population.end());
	}

	inline constexpr void replace(uint32_t index, NNAi<T>&& replacement) {
		assert(index < m_population.size());
		m_population[index] = std::move(replacement);
	}

	inline constexpr const NNAi<T>& getNNAiAt(const uint32_t index) const {
		assert(index < m_population.size());
		return m_population[index];
	}

	inline constexpr NNAi<T>& getNNAiAt(const uint32_t index) {
		assert(index < m_population.size());
		return m_population[index];
	}

	inline void printInfo() const { 
		std::cout << "Name: " << m_name
			<< ", Population size: " << m_population.size()
			<< ", Generation: " << m_generation
			<< ", Trained sessions: " << m_sessionsTrained
			<< ", Trained sessions this gen: " << m_sessionsTrainedThisGen << '\n';
		const NNAi<T>& best = getBestNNAiConstRef();
		std::cout << "Best score: " << best.getScore()
			<< ", times trained: " << best.getSessionsTrained()
			<< ", with ID: " << best.getID() << '\n';
		float avg = 0;
		for (const auto& nnai : m_population) {
			avg += nnai.getScore();
		}
		avg /= m_population.size();
		std::cout << "Avg score: " << avg << '\n';
	}

	inline constexpr const NNAi<T>& getBestNNAiConstRef() const {
		return *std::max_element(m_population.begin(), m_population.end());
	}

	inline constexpr uint64_t assignNextID() {
		return m_nextID++;
	}

private:
	std::vector<NNAi<T>> m_population;
	uint64_t m_generation;
	uint64_t m_sessionsTrained;
	uint64_t m_sessionsTrainedThisGen;
	std::string m_name;
	uint64_t m_nextID;

	void constexpr createPopulation(uint size, const std::vector<std::vector<uint>>& layers) {
		for (uint i = 0; i < size; ++i) {
			m_population.emplace_back(m_nextID++, layers);
		}
	}
};

template <typename T> struct NNPPTrainingUpdate {
	NNAi<T>& nnai;
	float updateValue;
	bool replaceValue;

	NNPPTrainingUpdate() = delete;
	constexpr NNPPTrainingUpdate(NNAi<T>& nnai, float updateValue, bool replaceValue)
			: nnai(nnai)
			, updateValue(updateValue)
			, replaceValue(replaceValue) { }
};

template <typename T> class NNPPTrainer {
public:
	NNPPTrainer() = delete;

	constexpr NNPPTrainer(const uint64_t sessions, const uint32_t threads, NNPopulation<T>& population, const EvolutionInfo<T>& defaultEvolInfo)
			: m_trainee(population)
			, m_sessions(sessions)
			, m_threads(threads)
			, m_totalSessionsCompleted(0)
			, m_defaultEvolInfo(defaultEvolInfo) {
		m_perThreadNeuronBuffers.reserve(m_threads);
		for (uint32_t i = 0; i < m_threads; ++i) {
			m_perThreadNeuronBuffers.push_back(allocNeuronBuffer<T>());
		}
	}

	virtual ~NNPPTrainer() { }

	void run(const bool verbose = true, const bool autoSave = true) {
		m_totalSessionsCompleted = 0;
		uint64_t sessionsCompleted = 0;
		std::vector<std::thread> workers;
		std::atomic<uint64_t> sessionsCounter = 0;

		auto workFunc = [this, verbose, &sessionsCounter](const uint64_t sessionsToRun, NeuronBuffer<T>* threadLocalNeuronBuffer) {
			while (sessionsCounter++ < sessionsToRun) {
				onSessionComplete(runSession(*threadLocalNeuronBuffer));

				if (verbose) {
					std::cout << "\rCompleted: " << m_totalSessionsCompleted << " out of: " << m_sessions;
					std::cout.flush();
				}
			}
		};

		while (sessionsCompleted < m_sessions) {
			const uint64_t sessionsToRun = std::min(m_sessions - sessionsCompleted, sessionsTillEvolution());
			const uint32_t threadsToUse = std::min(m_threads, static_cast<uint32_t>(sessionsToRun));
			sessionsCounter = 0;

			for (uint32_t i = 0; i < threadsToUse; ++i) {
				workers.emplace_back(workFunc, sessionsToRun, &m_perThreadNeuronBuffers[i]);
			}

			while (!workers.empty()) {
				workers.back().join();
				workers.pop_back();
			}

			if (shouldEvolve()) {
				evolve();
				if (autoSave) {
					m_trainee.saveToDisk();
				}
			}

			sessionsCompleted += sessionsToRun;
		}

		if (verbose) {
			std::cout << '\n';
		}

		if (autoSave) {
			m_trainee.saveToDisk();
		}
	}

	constexpr void evolve() {
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		pcg_extras::seed_seq_from<std::random_device> seed;
		pcg32_fast dev(seed);

		const float minFitness = getFitnessForNNAi(m_trainee.getMinScoreNNAi());
		const float maxFitness = getFitnessForNNAi(m_trainee.getMaxScoreNNAi());
		assert(minFitness <= maxFitness);

		EvolutionInfo<T> evolutionInfo = m_defaultEvolInfo;
		setEvolutionInfo(evolutionInfo);
		assert(evolutionInfo.minMutationValue <= evolutionInfo.maxMutationValue);

		std::vector<uint32_t> replaced;
		replaced.reserve(m_trainee.getPopulationSize());

		for (uint32_t i = 0; i < m_trainee.getPopulationSize(); ++i) {
			if (m_trainee.getNNAiAt(i).getSessionsTrained() < evolutionInfo.minTrainingSessionsRequired) {
				continue;
			}

			float normalizedFitness;
			if (maxFitness > minFitness) {
				normalizedFitness = normalize(getFitnessForNNAi(m_trainee.getNNAiAt(i)), minFitness, maxFitness);
			}
			else {
				normalizedFitness = 1.0f / static_cast<float>(m_trainee.getPopulationSize());
			}
			assert(normalizedFitness >= 0.0f);
			assert(normalizedFitness <= 1.0f);

			if (normalizedFitness < dist(dev)) {
				replaced.push_back(i);
			}
		}
		assert(minFitness == maxFitness || replaced.size() > 0);
		assert(minFitness == maxFitness || replaced.size() != m_trainee.getPopulationSize());

		std::uniform_int_distribution<uint32_t> intDist(0, m_trainee.getPopulationSize() - 1);
		for (const uint32_t r : replaced) {
			m_trainee.replace(r, createEvolvedNNAi(r, intDist, dev, minFitness, maxFitness, dist(dev), evolutionInfo));
		}
		m_trainee.evolutionCompleted();
	}

protected:
	NNPopulation<T>& m_trainee;

	virtual std::vector<NNPPTrainingUpdate<T>> runSession(NeuronBuffer<T>& threadLocalNeuronBuffer) = 0;
	virtual uint64_t sessionsTillEvolution() const = 0;

	virtual float getAvgScoreImportance() const { return 0.0f; }
	virtual void setEvolutionInfo(EvolutionInfo<float>& /*evolutionInfo*/) const { }

private:
	uint64_t m_sessions;
	uint32_t m_threads;
	uint64_t m_totalSessionsCompleted;
	std::mutex m_onSessionCompleteMutex;
	std::vector<NeuronBuffer<T>> m_perThreadNeuronBuffers;
	EvolutionInfo<T> m_defaultEvolInfo;

	constexpr NNAi<T> createEvolvedNNAi(
				  uint32_t index
				, std::uniform_int_distribution<uint32_t>& dist
				, pcg32_fast& dev
				, const float minFitness
				, const float maxFitness
				, const float fitnessTarget
				, const EvolutionInfo<T>& evolutionInfo) const {
		uint32_t nnai0 = index;
		uint32_t nnai1 = index;

		float target = fitnessTarget;
		while (!(nnai0 != index && normalize(getFitnessForNNAi(m_trainee.getNNAiAt(nnai0)), minFitness, maxFitness) >= target)) {
			nnai0 = dist(dev);
			target -= TARGET_DECREASE_RATE;
		}
		assert(normalize(getFitnessForNNAi(m_trainee.getNNAiAt(nnai0)), minFitness, maxFitness) >= target);

		target = fitnessTarget;
		while (!(nnai1 != index && nnai1 != nnai0 && normalize(getFitnessForNNAi(m_trainee.getNNAiAt(nnai1)), minFitness, maxFitness) >= target)) {
			nnai1 = dist(dev);
			target -= TARGET_DECREASE_RATE;
		}
		assert(normalize(getFitnessForNNAi(m_trainee.getNNAiAt(nnai1)), minFitness, maxFitness) >= target);

		assert(index != nnai0);
		assert(index != nnai1);
		assert(nnai0 != nnai1);
		assert(nnai0 < m_trainee.getPopulationSize());
		assert(nnai1 < m_trainee.getPopulationSize());

		return NNAi<T>(m_trainee.assignNextID()
					, m_trainee.getNNAiAt(nnai0)
					, m_trainee.getNNAiAt(nnai1)
					, evolutionInfo);
	}

	inline constexpr void onSessionComplete(const std::vector<NNPPTrainingUpdate<T>>& scoreUpdates) {
		std::lock_guard<std::mutex> lock(m_onSessionCompleteMutex);
		updateScores(scoreUpdates);
		m_trainee.trainSessionCompleted();
		m_totalSessionsCompleted++;
	}

	inline constexpr void updateScores(const std::vector<NNPPTrainingUpdate<T>>& scoreUpdates) const {
		for (const auto& [nnai, deltaScore, replace] : scoreUpdates) {
			nnai.sessionCompleted();
			if (replace) {
				nnai.updateScoreReplace(deltaScore);
			}
			else {
				nnai.updateScoreDelta(deltaScore);
			}
		}
	}

	inline constexpr bool shouldEvolve() const {
		return sessionsTillEvolution() == 0;
	}

	inline constexpr float getFitnessForNNAi(const NNAi<T>& nnai) const {
		const float avgImportance = std::max(0.0f, std::min(1.0f, getAvgScoreImportance()));
		return avgImportance * nnai.getAvgScore() + (1.0f - avgImportance) * nnai.getScore();
	}

};

template <typename T> class TrainerDataSet {
public:
	NNPPStackVector<T> input;
	NNPPStackVector<T> expected;
	uint32_t aiIndex;

	TrainerDataSet() : aiIndex(0) { }
	TrainerDataSet(const NNPPStackVector<T>& input, const NNPPStackVector<T>& expected, uint32_t aiIndex) :
			input(input),
			expected(expected),
			aiIndex(aiIndex) { }
};

template <typename T> class SimpleTrainer : public NNPPTrainer<T> {
public:
	SimpleTrainer() = delete;
	SimpleTrainer(uint32_t sessions, uint32_t threads, NNPopulation<T>& population, const std::vector<TrainerDataSet<T>>& tests, const EvolutionInfo<T>& defaultEvolutionInfo)
			: NNPPTrainer<T>(sessions, threads, population, defaultEvolutionInfo)
			, m_tests(tests) { }

protected:
	std::vector<NNPPTrainingUpdate<T>> runSession(NeuronBuffer<T>& threadLocalNeuronBuffer) override {
		std::vector<NNPPTrainingUpdate<T>> updates;
		for (uint32_t i = 0; i < NNPPTrainer<T>::m_trainee.getPopulationSize(); ++i) {
			NNAi<T>& nnai = NNPPTrainer<T>::m_trainee.getNNAiAt(i);

			if (nnai.getSessionsTrained() > 0) {
				continue;
			}
			float deltaScore = 0.0f;
			for (const auto& [input, exp, indx] : m_tests) {
				NNPPStackVector<T> out = nnai.feedAt(indx, input, threadLocalNeuronBuffer);
				assert(out.size() == exp.size());
				for (uint32_t o = 0; o < exp.size(); ++o) {
					deltaScore += -std::abs(exp[o] - out[o]);
				}
			}
			updates.emplace_back(nnai, deltaScore, true);
			nnai.sessionCompleted();
		}
		return updates;
	}

	uint64_t sessionsTillEvolution() const override {
		assert(NNPPTrainer<T>::m_trainee.getSessionsTrainedThisGen() <= 1);
		return NNPPTrainer<T>::m_trainee.getSessionsTrainedThisGen() - 1;
	}

private:
	std::vector<TrainerDataSet<T>> m_tests;
};

}
