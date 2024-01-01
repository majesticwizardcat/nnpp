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

typedef unsigned int uint;
typedef unsigned long long ulong;

static const uint VERSION_NN = 1;
static const std::string HEADER_STR_NN = "NNPPNN";
static const uint VERSION_NNAI = 2;
static const std::string HEADER_STR_NNAI = "NNPPNNAI";
static const uint VERSION_NNPP = 1;
static const std::string HEADER_STR_NNPP = "NNPP";

static const uint MAX_INPUT_OUTPUT_NEURONS = 1024;
static const uint MAX_NEURONS_PER_LAYER = 1 << 16;
static const float DEFAULT_WEIGHT_MUTATION_CHANCE = 0.05f;
static const uint DEFAULT_MAX_LAYER_MUTATION = 5;
static const float DEFAULT_LAYER_MUTATION_CHANCE = 0.3f;
static const float DEFAULT_LAYER_ADDITION_CHANCE = 0.5f;
static const float TARGET_DECREASE_RATE = 0.0005f;
static const float DEFAULT_CHILD_REGRESSION_PERCENTAGE = 0.95f;
static const uint DEFUALT_MIN_TRAINING_SESSIONS_REQUIRED = 1;
static const float DEFAULT_MIN_MUTATION_VALUE_FLOAT = -1.0f;
static const float DEFAULT_MAX_MUTATION_VALUE_FLOAT = 1.0f;

inline float normalize(float value, float min, float max) {
	return (value - min) / (max - min);
}

template <typename T, ulong N> class StackVector {
	static_assert(std::is_trivially_constructible_v<T>);
	static_assert(std::is_trivially_destructible_v<T>);

public:
	StackVector()
		: m_size(0) { }

	StackVector(const StackVector& other)
		: m_size(other.m_size)
		, m_array(other.m_array) { }

	StackVector(StackVector&& other)
		: m_size(std::move(other.m_size))
		, m_array(std::move(other.m_array)) { }
	
	StackVector(std::initializer_list<T> initValues)
		: m_size(initValues.size()) {
		std::copy(initValues.begin(), initValues.end(), m_array.begin());
	}

	inline void push(const T& val) { m_array[m_size++] = val; }
	inline ulong size() const { return m_size; }
	inline auto begin() const { return m_array.begin(); }
	inline auto end() const { return m_array.begin() + m_size; }
	inline bool empty() const { return m_size == 0; }
	inline void erase(ulong index) { std::swap(m_array[index], m_array[--m_size]); }
	inline void clear() { m_size = 0; }
	inline T& addNew() { return m_array[m_size++]; }
	inline T& operator[](ulong pos) { m_size = std::max(m_size, pos + 1); return m_array[pos]; }
	inline const T& operator[](ulong pos) const { return m_array[pos]; }

	inline StackVector& operator=(StackVector&& other) {
		m_array = std::move(other.m_array);
		m_size = std::move(other.m_size);
		return *this;
	}

private:
	ulong m_size;
	std::array<T, N> m_array;
};

template <typename T> struct EvolutionInfo {
	T minMutationValue;
	T maxMutationValue;
	float childRegressionPercentage;
	float weightMutationChance;
	float layerMutationChance;
	float layerAdditionChance;
	uint maxLayersMutation;
	uint minTrainingSessionsRequired;
};

static EvolutionInfo<float> getDefaultEvolutionInfoFloat() {
	EvolutionInfo<float> evolutionInfo;
	evolutionInfo.minMutationValue = DEFAULT_MIN_MUTATION_VALUE_FLOAT;
	evolutionInfo.maxMutationValue = DEFAULT_MAX_MUTATION_VALUE_FLOAT;
	evolutionInfo.weightMutationChance = DEFAULT_WEIGHT_MUTATION_CHANCE;
	evolutionInfo.layerAdditionChance = DEFAULT_LAYER_ADDITION_CHANCE;
	evolutionInfo.layerMutationChance = DEFAULT_LAYER_MUTATION_CHANCE;
	evolutionInfo.maxLayersMutation = DEFAULT_MAX_LAYER_MUTATION;
	evolutionInfo.minTrainingSessionsRequired = DEFUALT_MIN_TRAINING_SESSIONS_REQUIRED;
	evolutionInfo.childRegressionPercentage = DEFAULT_CHILD_REGRESSION_PERCENTAGE;
	return evolutionInfo;
}

template <typename T> using NNPPStackVector = StackVector<T, MAX_INPUT_OUTPUT_NEURONS>;

// First half of the array is for inputs for the next layer and the other half is for the outputs
template <typename T> using NeuronBuffer = std::vector<T>;

template <typename T> static inline NeuronBuffer<T> allocNeuronBuffer() {
	return NeuronBuffer<T>(MAX_NEURONS_PER_LAYER * 2);
}

template <typename T> class NeuralNetwork {
public:
	NeuralNetwork() = delete;

	NeuralNetwork(const NeuralNetwork& other) = delete;

	NeuralNetwork(const std::vector<uint>& layers)
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
		initInternal();
		m_data = std::make_unique<T[]>(getDataSize());
		m_neuronBiases = std::make_unique<T[]>(getNumOfTotalNeurons());
	}

	NeuralNetwork(NeuralNetwork&& other)
		: m_layerSizes(std::move(other.m_layerSizes))
		, m_dataSizePerLayer(std::move(other.m_dataSizePerLayer))
		, m_data(std::move(other.m_data))
		, m_neuronBiases(std::move(other.m_neuronBiases))
		, m_numOfNeuronsPerLayer(std::move(other.m_numOfNeuronsPerLayer)) {
	}

	NeuralNetwork(const std::string& location)
		: m_data(nullptr)
		, m_neuronBiases(nullptr) {
		if (!loadFromFile(location)) {
			std::cout << "Could not load from file" << '\n';
			assert(false);
		}
	}

	NeuralNetwork(std::ifstream* const stream) 
		: m_data(nullptr)
		, m_neuronBiases(nullptr) {
		if (!readFromStream(stream)) {
			std::cout << "Could not read from stream" << '\n';
			assert(false);
		}
	}

	NeuralNetwork(const NeuralNetwork<T>& n0, const NeuralNetwork<T>& n1, const EvolutionInfo<T>& evolutionInfo) {
		initFromParents(n0, n1, evolutionInfo);
	}

	void initFromParents(const NeuralNetwork<T>& n0, const NeuralNetwork<T>& n1, const EvolutionInfo<T>& evolutionInfo) {
		assert(n0.m_layerSizes.size() == n1.m_layerSizes.size());

		std::uniform_real_distribution<float> realDist(0.0f, 1.0f);
		std::uniform_real_distribution<T> mutationValueDist(evolutionInfo.minMutationValue, evolutionInfo.maxMutationValue);
		std::uniform_int_distribution<uint> layerMutation(0, evolutionInfo.maxLayersMutation);
		std::random_device dev;
		const T* dataN0 = n0.m_data.get();
		const T* dataN1 = n1.m_data.get();
		const T* biasesN0 = n0.m_neuronBiases.get();
		const T* biasesN1 = n1.m_neuronBiases.get();

		clearAll();
		
		std::vector<uint> minLayerSizes(n0.m_layerSizes.size());
		m_layerSizes.resize(n0.m_layerSizes.size());
		for (uint i = 0; i < n0.m_layerSizes.size(); ++i) {
			if (i > 0 && i < n0.m_layerSizes.size() - 1) {
				uint extras = realDist(dev) < evolutionInfo.layerMutationChance ? layerMutation(dev) : 0;
				m_layerSizes[i] = (n0.m_layerSizes[i] + n1.m_layerSizes[i]) / 2;
				if (realDist(dev) < evolutionInfo.layerAdditionChance) {
					m_layerSizes[i] += extras;
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

		initInternal();
		m_data = std::make_unique<T[]>(getDataSize());
		m_neuronBiases = std::make_unique<T[]>(getNumOfTotalNeurons());

		for (uint tl = 1; tl < m_layerSizes.size(); ++tl) {
			for (uint tn = 0; tn < m_layerSizes[tl]; ++tn) {
				for (uint fn = 0; fn < m_layerSizes[tl - 1]; ++fn) {
					T* weightPtr = weightPtrAt(fn, tn, tl);
					float mutation = realDist(dev);
					if (mutation < evolutionInfo.weightMutationChance || tn >= minLayerSizes[tl] || fn >= minLayerSizes[tl - 1]) {
						*weightPtr = mutationValueDist(dev);
					}
					else {
						*weightPtr = realDist(dev) < 0.5f ? n0.weightAt(fn, tn, tl) : n1.weightAt(fn, tn, tl);
					}
				}
			}
		}

		for (uint l = 0; l < m_layerSizes.size(); ++l) {
			for (uint n = 0; n < m_layerSizes[l]; ++n) {
				T* biasPtr = neuronBiasPtrAt(n, l);
				float mutation = realDist(dev);
				if (mutation < evolutionInfo.weightMutationChance || n >= minLayerSizes[l]) {
					*biasPtr = mutationValueDist(dev);
				}
				else {
					*biasPtr = realDist(dev) < 0.5f ? n0.neuronBiasAt(n, l) : n1.neuronBiasAt(n, l);
				}
			}
		}
	}

	void initDataVal(const T& val) {
		for (ulong i = 0; i < getDataSize(); ++i) {
			m_data[i] = val;
		}
		initBiasesVal(val);
	}

	void initBiasesVal(const T& val) {
		uint biasesNum = getNumOfTotalNeurons();
		for (uint i = 0; i < biasesNum; ++i) {
			m_neuronBiases[i] = val;
		}
	}

	void initData(const std::vector<T>& data, const std::vector<T>& biases) {
		assert(data.size() == getDataSize());
		assert(biases.size() == getNumOfTotalNeurons());
		for (ulong i = 0; i < getDataSize(); ++i) {
			m_data[i] = data[i];
		}

		uint biasesNum = getNumOfTotalNeurons();
		for (uint i = 0; i < biasesNum; ++i) {
			m_neuronBiases[i] = biases[i];
		}
	}

	inline const ulong getDataSize() const {
		return m_dataSizePerLayer.back();
	}

	inline const ulong getDataSizeForLayer(ulong layer) const {
		assert(layer < m_dataSizePerLayer.size());
		return m_dataSizePerLayer[layer];
	}

	inline uint getNumOfTotalNeurons() const {
		assert(m_numOfNeuronsPerLayer.back() == std::accumulate(m_layerSizes.begin(), m_layerSizes.end(), 0));
		return m_numOfNeuronsPerLayer.back();
	}

	void randomizeDataUniform(const T& min, const T& max) {
		std::uniform_real_distribution<T> dist(min, max);
		std::random_device dev;
		for (ulong i = 0; i < getDataSize(); ++i) {
			m_data[i] = dist(dev);
		}

		uint biasesNum = getNumOfTotalNeurons();
		for (uint i = 0; i < biasesNum; ++i) {
			m_neuronBiases[i] = dist(dev);
		}
	}

	bool saveToStream(std::ofstream* saveFile) const {
		if (!*saveFile) {
			return false;
		}

		if (!saveHeader(saveFile)) {
			return false;
		}

		uint size = m_layerSizes.size();
		if (!saveFile->write(reinterpret_cast<const char*>(&size), sizeof(uint))) {
			return false;
		}

		for (uint l : m_layerSizes) {
			if (!saveFile->write(reinterpret_cast<const char*>(&l), sizeof(l))) {
				return false;
			}
		}

		if (!saveFile->write(reinterpret_cast<const char*>(m_data.get()), sizeof(T) * getDataSize())
			|| !saveFile->write(reinterpret_cast<const char*>(m_neuronBiases.get()), sizeof(T) * getNumOfTotalNeurons())) {
			return false;
		}
		return true;
	}

	bool saveToFile(const std::string& fileName) const {
		assert(m_data && getDataSize() > 0);
		std::ofstream saveFile(fileName, std::ios::out | std::ios::binary);
		return saveToStream(&saveFile);
	}

	bool readFromStream(std::ifstream* file) {
		clearAll();
		if (!*file) {
			return false;
		}

		uint headerVersion = 0;
		if (!readHeader(file, &headerVersion)
			|| headerVersion > VERSION_NN) {
			std::cout << "Header version: " << headerVersion << '\n';
			return false;
		}

		switch(headerVersion) {
		case 1:
			return loadVersion1(file);
		default:
			break;
		}
		return false;
	}

	bool loadFromFile(const std::string& location) {
		std::ifstream file(location, std::ios::in | std::ios::binary);
		return readFromStream(&file);
	}

	void clearAll() {
		m_layerSizes.clear();
		m_data.release();
		m_neuronBiases.release();
		m_dataSizePerLayer.clear();
		m_numOfNeuronsPerLayer.clear();
	}

	NNPPStackVector<T> feed(const NNPPStackVector<T>& input, NeuronBuffer<T>& neuronBuffer) const {
		latchInputs(input, neuronBuffer);
		propagate(neuronBuffer);
		return getOutputs(neuronBuffer);
	}

	bool operator==(const NeuralNetwork& other) const {
		if (getDataSize() != other.getDataSize()
			|| m_layerSizes != other.m_layerSizes) {
			return false;
		}

		for (ulong i = 0; i < getDataSize(); ++i) {
			if (m_data[i] != other.m_data[i]) {
				return false;
			}
		}
		return true;
	}

	inline void printLayerSizes() const {
		std::cout << "{ ";
		for (uint l : m_layerSizes) {
			std::cout << l << ' ';
		}
		std::cout << "}" << '\n';
	}

	inline void printData() const {
		for (uint i = 1; i < m_layerSizes.size(); ++i) {
			for (uint n = 0; n < m_layerSizes[i]; ++n) {
				printWeightsAt(n, i);
			}
		}
	}

private:
	std::vector<uint> m_layerSizes;
	std::unique_ptr<T[]> m_data;
	std::unique_ptr<T[]> m_neuronBiases;
	std::vector<ulong> m_dataSizePerLayer;
	std::vector<uint> m_numOfNeuronsPerLayer;

	bool saveHeader(std::ofstream* saveFile) const {
		return saveFile->write(reinterpret_cast<const char*>(HEADER_STR_NN.data()), sizeof(HEADER_STR_NN.size()))
			&& saveFile->write(reinterpret_cast<const char*>(&VERSION_NN), sizeof(uint));
	}

	bool readHeader(std::ifstream* file, uint* headerVersion) const {
		std::string headerStr;
		headerStr.resize(HEADER_STR_NN.size());
		if (!file->read(reinterpret_cast<char*>(headerStr.data()), sizeof(HEADER_STR_NN.size()))) {
			return false;
		}
		return headerStr == HEADER_STR_NN
			&& file->read(reinterpret_cast<char*>(headerVersion), sizeof(uint));
	}

	bool loadVersion1(std::ifstream* file) {
		uint layers;
		if (!file->read(reinterpret_cast<char*>(&layers), sizeof(uint))) {
			return false;
		}
		for (uint l = 0; l < layers; ++l) {
			uint cur = 0;
			if (!file->read(reinterpret_cast<char*>(&cur), sizeof(uint))) {
				return false;
			}
			m_layerSizes.push_back(cur);
		}

		initInternal();
		uint biases = getNumOfTotalNeurons();
		m_data = std::make_unique<T[]>(getDataSize());
		m_neuronBiases = std::make_unique<T[]>(biases);
		if (!file->read(reinterpret_cast<char*>(m_data.get()), sizeof(T) * getDataSize())
			|| !file->read(reinterpret_cast<char*>(m_neuronBiases.get()), sizeof(T) * biases)) {
			return false;
		}
		return true;
	}

	inline void initInternal() {
		calculateDataSizes();
		calculateNeuronSizes();
	}

	inline uint weightIndex(uint fromNeuron, uint toNeuron, uint toLayer) const {
		assert(toLayer > 0);
		assert(toLayer < m_layerSizes.size());
		assert(getDataSizeForLayer(toLayer)
			+ fromNeuron * m_layerSizes[toLayer]
			+ toNeuron < getDataSize());
		return getDataSizeForLayer(toLayer)
			+ fromNeuron * m_layerSizes[toLayer]
			+ toNeuron;
	}

	inline uint neuronBiasIndex(uint neuron, uint layer) const {
		assert(layer >= 0);
		assert(layer < m_numOfNeuronsPerLayer.size());
		assert(m_numOfNeuronsPerLayer[layer] + neuron < getNumOfTotalNeurons());
		return m_numOfNeuronsPerLayer[layer] + neuron;
	}

	inline const T& weightAt(uint fromNeuron, uint toNeuron, uint toLayer) const {
		return m_data[weightIndex(fromNeuron, toNeuron, toLayer)];
	}

	inline const T& neuronBiasAt(uint neuron, uint layer) const {
		return m_neuronBiases[neuronBiasIndex(neuron, layer)];
	}

	inline T* weightPtrAt(uint fromNeuron, uint toNeuron, uint toLayer) const {
		return m_data.get() + weightIndex(fromNeuron, toNeuron, toLayer);
	}

	inline T* neuronBiasPtrAt(uint neuron, uint layer) const {
		return m_neuronBiases.get() + neuronBiasIndex(neuron, layer);
	}

	inline void latchInputs(const NNPPStackVector<T>& inputs, NeuronBuffer<T>& neurons) const {
		assert(inputs.size() == *m_layerSizes.begin());
		assert(neurons.size() >= inputs.size() * 2);
		for (uint i = 0; i < inputs.size(); ++i) {
			neurons[i] = inputs[i] + neuronBiasAt(i, 0);
		}
	}

	inline void propagate(NeuronBuffer<T>& neurons) const {
		// half size is where the array for copy dest begins
		const ulong halfSize = neurons.size() / 2;
		for (uint l = 1; l < m_layerSizes.size(); ++l) {
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

	inline void printWeightsAt(uint neuron, uint layer) const {
		assert(layer >= 1);
		for (uint n = 0; n < m_layerSizes[layer - 1]; ++n) {
			std::cout << weightAt(n, neuron, layer) << ' ';
		}
		std::cout << '\n';
	}

	inline NNPPStackVector<T> getOutputs(const NeuronBuffer<T>& neurons) const {
		NNPPStackVector<T> out;
		for (uint i = 0; i < m_layerSizes.back(); ++i) {
			out[i] = neurons[i];
		}
		return out;
	}

	inline void calculateDataSizes() {
		assert(m_layerSizes.size() > 1);
		auto calculateDataSizeForLayer = [this](ulong layer) {
			assert(layer <= m_layerSizes.size());
			ulong size = 0;
			for (ulong i = 1; i < layer; ++i) {
				size += m_layerSizes[i - 1] * m_layerSizes[i];
			}
			return size;
		};

		m_dataSizePerLayer.resize(m_layerSizes.size() + 1);
		for (ulong i = 0; i <= m_layerSizes.size(); ++i) {
			m_dataSizePerLayer[i] = calculateDataSizeForLayer(i);
		}
	}

	inline void calculateNeuronSizes() {
		m_numOfNeuronsPerLayer.resize(m_layerSizes.size() + 1);
		for (uint i = 0; i < m_layerSizes.size(); ++i) {
			m_numOfNeuronsPerLayer[i] = std::accumulate(m_layerSizes.begin(), m_layerSizes.begin() + i, 0);
		}
		m_numOfNeuronsPerLayer[m_layerSizes.size()] = std::accumulate(m_layerSizes.begin(), m_layerSizes.end(), 0);
		assert(getNumOfTotalNeurons() == std::accumulate(m_layerSizes.begin(), m_layerSizes.end(), 0));
	}
};

template <typename T> class NNAi {
public:
	NNAi() = delete;
	NNAi(const NNAi& other) = delete;
	NNAi(ulong id, const std::vector<std::vector<uint>>& layers)
		: m_id(id)
		, m_sessionsTrained(0)
		, m_score(0.0f) {
		for (const auto& ls : layers) {
			m_networks.emplace_back(ls);
		}
	}

	NNAi(std::ifstream* file) {
		if (!loadFromStream(file)) {
			std::cout << "Could not load NNAi from stream" << '\n';
			assert(false);
		}
	}

	NNAi(ulong id, const NNAi& nnai0, const NNAi& nnai1, const EvolutionInfo<T>& evolutionInfo) {
		initFromParents(id, nnai0, nnai1, evolutionInfo);
	}

	NNAi(NNAi&& other)
		: m_id(std::move(other.m_id))
		, m_score(std::move(other.m_score))
		, m_sessionsTrained(std::move(other.m_sessionsTrained))
		, m_networks(std::move(other.m_networks)) {
	}

	void initRandomUniform(const T& min, const T& max) {
		for (auto& nn : m_networks) {
			nn.randomizeDataUniform(min, max);
		}
	}

	void initVal(const T& val) {
		for (auto& nn : m_networks) {
			nn.initDataVal(val);
		}
	}

	void initBiasesVal(const T& val) {
		for (auto& nn : m_networks) {
			nn.initBiasesVal(val);
		}
	}

	void initFromParents(ulong id, const NNAi& nnai0, const NNAi& nnai1, const EvolutionInfo<T>& evolutionInfo) {
		assert(nnai0.getNetworksNumber() == nnai1.getNetworksNumber());
		clearAll();
		m_id = id;
		m_score = (nnai0.getScore() + nnai1.getScore()) * 0.5f * evolutionInfo.childRegressionPercentage;
		for (uint i = 0; i < nnai0.getNetworksNumber(); ++i) {
			m_networks.emplace_back(nnai0.getConstRefAt(i), nnai1.getConstRefAt(i), evolutionInfo);
		}
	}

	inline bool saveToFile(const std::string& location) const {
		std::ofstream file(location, std::ios::out | std::ios::binary);
		return saveToStream(&file);
	}

	bool saveToStream(std::ofstream* file) const {
		if (!*file) {
			return false;
		}

		if (!saveHeader(file)) {
			return false;
		}

		uint nets = m_networks.size();
		if (!file->write(reinterpret_cast<const char*>(&m_id), sizeof(ulong))
			|| !file->write(reinterpret_cast<const char*>(&nets), sizeof(uint))
			|| !file->write(reinterpret_cast<const char*>(&m_sessionsTrained), sizeof(uint))
			|| !file->write(reinterpret_cast<const char*>(&m_score), sizeof(float))) {
			return false;
		}

		for (const auto& nn : m_networks) {
			if (!nn.saveToStream(file)) {
				return false;
			}
		}

		return true;
	}

	bool loadFromFile(const std::string& location) {
		std::ifstream file(location, std::ios::in | std::ios::binary);
		return loadFromStream(&file);
	}

	bool loadFromStream(std::ifstream* file) {
		clearAll();
		if (!*file) {
			return false;
		}

		uint headerVersion = 0;
		if (!readHeader(file, &headerVersion)) {
			return false;
		}

		switch (headerVersion) {
		case 1:
			return loadVersion1(file);
		case 2:
			return loadVersion2(file);
		default:
			break;
		}
		return false;
	}

	void clearAll() {
		m_id = 0;
		m_score = 0.0f;
		m_sessionsTrained = 0;
		m_networks.clear();
	}

	inline NNAi& operator=(NNAi&& other) {
		m_id = std::move(other.m_id);
		m_score = std::move(other.m_score);
		m_sessionsTrained = std::move(other.m_sessionsTrained);
		m_networks = std::move(other.m_networks);
		return *this;
	}

	inline NNPPStackVector<T> feedAt(uint index, const NNPPStackVector<T>& input, NeuronBuffer<T>& neuronBuffer) const {
		assert(index < m_networks.size());
		return m_networks[index].feed(input, neuronBuffer);
	}

	inline const NeuralNetwork<T>& getConstRefAt(uint index) const {
		assert(index < m_networks.size());
		return m_networks[index];
	}

	inline ulong getID() const {
		return m_id;
	}

	inline void setID(ulong newID) {
		m_id = newID;
	}

	inline uint getNetworksNumber() const {
		return m_networks.size();
	}

	inline uint getSessionsTrained() const {
		return m_sessionsTrained;
	}

	inline void sessionCompleted() {
		m_sessionsTrained++;
	}

	inline void updateScoreReplace(float newScore) {
		m_score = newScore;
	}

	inline void updateScoreDelta(float deltaScore) {
		m_score += deltaScore;
	}

	inline float getScore() const {
		return m_score;
	}

	inline float getAvgScore() const {
		return m_sessionsTrained == 0 ? 0.0f : m_score / static_cast<float>(m_sessionsTrained);
	}

	inline bool operator<(const NNAi& other) const {
		return m_score < other.m_score;
	}

	inline bool operator>(const NNAi& other) const {
		return m_score > other.m_score;
	}

	inline void printLayerSizes() const {
		for (const auto& nn : m_networks) {
			nn.printLayerSizes();
		}
	}

	inline void printData() const {
		for (const auto& nn: m_networks) {
			nn.printData();
		}
	}

private:
	ulong m_id;
	std::vector<NeuralNetwork<T>> m_networks;
	float m_score;
	uint m_sessionsTrained;

	bool saveHeader(std::ofstream* saveFile) const {
		return saveFile->write(reinterpret_cast<const char*>(HEADER_STR_NNAI.data()), sizeof(HEADER_STR_NNAI.size()))
			&& saveFile->write(reinterpret_cast<const char*>(&VERSION_NNAI), sizeof(uint));
	}

	bool readHeader(std::ifstream* file, uint* headerVersion) const {
		std::string headerStr;
		headerStr.resize(HEADER_STR_NNAI.size());
		if (!file->read(reinterpret_cast<char*>(headerStr.data()), sizeof(HEADER_STR_NNAI.size()))) {
			return false;
		}
		return headerStr == HEADER_STR_NNAI
			&& file->read(reinterpret_cast<char*>(headerVersion), sizeof(uint));
	}

	bool loadVersion1(std::ifstream* file) {
		uint nets;
		if (!file->read(reinterpret_cast<char*>(&nets), sizeof(uint))
			|| !file->read(reinterpret_cast<char*>(&m_sessionsTrained), sizeof(uint))
			|| !file->read(reinterpret_cast<char*>(&m_score), sizeof(float))) {
			return false;
		}

		for (uint i = 0; i < nets; ++i) {
			m_networks.emplace_back(file);
		}

		return true;
	}

	bool loadVersion2(std::ifstream* file) {
		return file->read(reinterpret_cast<char*>(&m_id), sizeof(ulong)) && loadVersion1(file);
	}
};

template <typename T> class NNPopulation {
public:
	NNPopulation() = delete;
	NNPopulation(const NNPopulation& other) = delete;

	NNPopulation(const std::string& name, uint size, const std::vector<std::vector<uint>>& layers)
		: m_name(name)
		, m_generation(0)
		, m_sessionsTrained(0)
		, m_sessionsTrainedThisGen(0)
		, m_nextID(0) {
		createPopulation(size, layers);
	}

	NNPopulation(const std::string& name)
		: m_name(name)
		, m_nextID(0) {
		loadFromDisk(name);
	}

	NNPopulation(NNPopulation&& other)
		: m_name(std::move(other.m_name))
		, m_sessionsTrained(std::move(other.m_sessionsTrained))
		, m_generation(std::move(m_generation))
		, m_population(std::move(other.m_population))
		, m_nextID(std::move(other.m_nextID)) {
	}

	inline NNPopulation& operator=(NNPopulation&& other) {
		m_name = std::move(other.m_name);
		m_sessionsTrained = std::move(other.m_sessionsTrained);
		m_generation = std::move(other.m_generation);
		m_population = std::move(other.m_population);
		m_nextID = std::move(other.m_nextID);
		return *this;
	}

	void createRandom(const T& min, const T& max) {
		for (auto& nnai : m_population) {
			nnai.initRandomUniform(min, max);
		}
	}

	inline bool saveToDisk() const {
		return saveToDisk(m_name);
	}

	bool saveToDisk(const std::string& location) const {
		std::ofstream file(location, std::ios::out | std::ios::binary);
		if (!file) {
			return false;
		}

		if (!saveHeader(&file)) {
			return false;
		}

		uint size = m_population.size();
		if (!file.write(reinterpret_cast<const char*>(&size), sizeof(uint))
			|| !file.write(reinterpret_cast<const char*>(&m_generation), sizeof(uint))
			|| !file.write(reinterpret_cast<const char*>(&m_sessionsTrained), sizeof(uint))
			|| !file.write(reinterpret_cast<const char*>(&m_sessionsTrainedThisGen), sizeof(uint))) {
			return false;
		}

		for (const auto& nn : m_population) {
			if (!nn.saveToStream(&file)) {
				return false;
			}
		}
		return true;
	}

	bool loadFromDisk(const std::string& location) {
		clearAll();
		m_name = location;
		std::ifstream file(location, std::ios::binary);
		if (!file) {
			return false;
		}

		uint headerVersion = 0;
		if (!readHeader(&file, &headerVersion)) {
			return false;
		}

		bool result = false;
		switch (headerVersion) {
		case 1:
			result = loadVersion1(&file);
			break;
		default:
			return false;
		}

		if (!result) {
			return false;
		}

		recalculateIDs();
		return true;
	}

	void clearAll() {
		m_population.clear();
		m_generation = 0;
		m_sessionsTrained = 0;
		m_nextID = 0;
	}

	inline void evolutionCompleted() {
		m_generation++;
		m_sessionsTrainedThisGen = 0;
	}

	inline void trainSessionCompleted() {
		m_sessionsTrained++;
		m_sessionsTrainedThisGen++;
	}

	inline uint getGenerartion() const {
		return m_generation;
	}
	
	inline uint getSessionsTrained() const {
		return m_sessionsTrained;
	}

	inline uint getSessionsTrainedThisGen() const {
		return m_sessionsTrainedThisGen;
	}

	inline uint getPopulationSize() const {
		return m_population.size();
	}

	inline NNAi<T>* const getNNAiPtrAt(uint index) {
		assert(index < m_population.size());
		return &m_population[index];
	}

	inline NNAi<T>& getMinScoreNNAi() {
		return *std::min_element(m_population.begin(), m_population.end());
	}

	inline NNAi<T>& getMaxScoreNNAi() {
		return *std::max_element(m_population.begin(), m_population.end());
	}

	inline void replace(uint index, NNAi<T> replacement) {
		assert(index < m_population.size());
		m_population[index] = std::move(replacement);
	}

	inline const NNAi<T>& getConstRefAt(uint index) const {
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

	inline const NNAi<T>& getBestNNAiConstRef() const {
		return *std::max_element(m_population.begin(), m_population.end());
	}

	inline ulong assignNextID() {
		return m_nextID++;
	}

private:
	std::vector<NNAi<T>> m_population;
	uint m_generation;
	uint m_sessionsTrained;
	uint m_sessionsTrainedThisGen;
	std::string m_name;
	ulong m_nextID;

	void createPopulation(uint size, const std::vector<std::vector<uint>>& layers) {
		for (uint i = 0; i < size; ++i) {
			m_population.emplace_back(m_nextID++, layers);
		}
	}

	void recalculateIDs() {
		if (m_population.empty()) {
			return;
		}

		auto cmpID = [&](const NNAi<T>& n0, const NNAi<T>& n1) {
			return n0.getID() < n1.getID();
		};
		std::sort(m_population.begin(), m_population.end(), cmpID);

		m_nextID = m_population[0].getID();
		for (size_t i = 1; i < m_population.size(); ++i) {
			if (m_population[i].getID() <= m_nextID) {
				m_population[i].setID(m_nextID + 1);
			}
			m_nextID = m_population[i].getID();
		}
		m_nextID++;
	}

	bool saveHeader(std::ofstream* saveFile) const {
		return saveFile->write(reinterpret_cast<const char*>(HEADER_STR_NNPP.data()), sizeof(HEADER_STR_NNPP.size()))
			&& saveFile->write(reinterpret_cast<const char*>(&VERSION_NNPP), sizeof(uint));
	}

	bool readHeader(std::ifstream* file, uint* headerVersion) const {
		std::string headerStr;
		headerStr.resize(HEADER_STR_NNPP.size());
		if (!file->read(reinterpret_cast<char*>(headerStr.data()), sizeof(HEADER_STR_NNPP.size()))) {
			return false;
		}
		return headerStr == HEADER_STR_NNPP
			&& file->read(reinterpret_cast<char*>(headerVersion), sizeof(uint));
	}

	bool loadVersion1(std::ifstream* file) {
		uint size = 0;
		if (!file->read(reinterpret_cast<char*>(&size), sizeof(uint))
			|| !file->read(reinterpret_cast<char*>(&m_generation), sizeof(uint))
			|| !file->read(reinterpret_cast<char*>(&m_sessionsTrained), sizeof(uint))
			|| !file->read(reinterpret_cast<char*>(&m_sessionsTrainedThisGen), sizeof(uint))) {
			return false;
		}

		for (uint i = 0; i < size; ++i) {
			m_population.emplace_back(file);
		}
		return true;
	}
};

template <typename T> struct NNPPTrainingUpdate {
	NNAi<T>* const nnai;
	float updateValue;
	bool replaceValue;

	NNPPTrainingUpdate<T>() = delete;
	NNPPTrainingUpdate<T>(NNAi<T>* const nnai, float updateValue, bool replaceValue)
		: nnai(nnai)
		, updateValue(updateValue)
		, replaceValue(replaceValue) { }
};

template <typename T> class NNPPTrainer {
public:
	NNPPTrainer() = delete;
	NNPPTrainer(uint sessions, uint threads, NNPopulation<T>* const population, const EvolutionInfo<T>& defaultEvolInfo)
			: m_sessions(sessions)
			, m_threads(threads)
			, m_totalSessionsCompleted(0)
			, m_trainee(population)
			, m_defaultEvolInfo(defaultEvolInfo) {
		assert(population);
		for (uint i = 0; i < m_threads; ++i) {
			m_perThreadNeuronBuffers.push_back(allocNeuronBuffer<T>());
		}
	}

	virtual ~NNPPTrainer() { }

	void run(bool verbose) {
		run(verbose, true);
	}

	void run(bool verbose, bool autoSave) {
		m_totalSessionsCompleted = 0;
		uint sessionsCompleted = 0;
		std::vector<std::thread> workers;
		std::atomic<uint> sessionsCounter = 0;

		auto workFunc = [this, verbose, &sessionsCounter](uint sessionsToRun, NeuronBuffer<T>* threadLocalNeuronBuffer) {
			while (sessionsCounter++ < sessionsToRun) {
				onSessionComplete(runSession(*threadLocalNeuronBuffer));
				if (verbose) {
					std::cout << "\rCompleted: " << m_totalSessionsCompleted << " out of: " << m_sessions;
					std::cout.flush();
				}
			}
		};

		while (sessionsCompleted < m_sessions) {
			uint sessionsToRun = std::min(m_sessions - sessionsCompleted, sessionsTillEvolution());
			uint threadsToUse = std::min(m_threads, sessionsToRun);
			sessionsCounter = 0;

			for (uint i = 0; i < threadsToUse; ++i) {
				workers.emplace_back(workFunc, sessionsToRun, &m_perThreadNeuronBuffers[i]);
			}

			while (!workers.empty()) {
				workers.back().join();
				workers.pop_back();
			}

			if (shouldEvolve()) {
				evolve();
				if (autoSave) {
					save();
				}
			}

			sessionsCompleted += sessionsToRun;
		}
		if (verbose) {
			std::cout << '\n';
		}

		if (autoSave) {
			save();
		}
	}

	inline void save() const {
		m_trainee->saveToDisk();
	}

	void evolve() {
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		std::random_device dev;

		float minFitness = getFitnessForNNAi(m_trainee->getMinScoreNNAi());
		float maxFitness = getFitnessForNNAi(m_trainee->getMaxScoreNNAi());
		assert(minFitness <= maxFitness);

		EvolutionInfo<T> evolutionInfo = m_defaultEvolInfo;
		setEvolutionInfo(evolutionInfo);
		assert(evolutionInfo.minMutationValue <= evolutionInfo.maxMutationValue);

		std::vector<uint> replaced;
		for (uint i = 0; i < m_trainee->getPopulationSize(); ++i) {
			if (m_trainee->getConstRefAt(i).getSessionsTrained() < evolutionInfo.minTrainingSessionsRequired) {
				continue;
			}

			float normalizedFitness = 0.0f;
			if (maxFitness > minFitness) {
				normalizedFitness = normalize(getFitnessForNNAi(m_trainee->getConstRefAt(i)), minFitness, maxFitness);
			}
			else {
				normalizedFitness = 1.0f / static_cast<float>(m_trainee->getPopulationSize());
			}
			assert(normalizedFitness >= 0.0f);
			assert(normalizedFitness <= 1.0f);

			if (normalizedFitness < dist(dev)) {
				replaced.push_back(i);
			}
		}
		assert(minFitness == maxFitness || replaced.size() > 0);
		assert(minFitness == maxFitness || replaced.size() != m_trainee->getPopulationSize());

		std::uniform_int_distribution<uint> intDist(0, m_trainee->getPopulationSize() - 1);
		for (uint r : replaced) {
			m_trainee->replace(r, createEvolvedNNAi(r, &intDist, &dev, minFitness, maxFitness, dist(dev), evolutionInfo));
		}
		m_trainee->evolutionCompleted();
	}

protected:
	NNPopulation<T>* const m_trainee;

	virtual std::vector<NNPPTrainingUpdate<T>> runSession(NeuronBuffer<T>& threadLocalNeuronBuffer) = 0;
	virtual uint sessionsTillEvolution() const = 0;

	virtual float getAvgScoreImportance() const { return 0.0f; }
	virtual void setEvolutionInfo(EvolutionInfo<float>& evolutionInfo) const { }

private:
	uint m_sessions;
	uint m_threads;
	uint m_totalSessionsCompleted;
	std::mutex m_onSessionCompleteMutex;
	std::vector<NeuronBuffer<T>> m_perThreadNeuronBuffers;
	EvolutionInfo<T> m_defaultEvolInfo;

	NNAi<T> createEvolvedNNAi(uint index, std::uniform_int_distribution<uint>* const dist,
		std::random_device* const dev, float minFitness, float maxFitness, float fitnessTarget, const EvolutionInfo<T>& evolutionInfo) const {
		uint nnai0 = index;
		uint nnai1 = index;

		float target = fitnessTarget;
		while (!(nnai0 != index && normalize(getFitnessForNNAi(m_trainee->getConstRefAt(nnai0)), minFitness, maxFitness) >= target)) {
			nnai0 = (*dist)(*dev);
			target -= TARGET_DECREASE_RATE;
		}
		assert(normalize(getFitnessForNNAi(m_trainee->getConstRefAt(nnai0)), minFitness, maxFitness) >= target);

		target = fitnessTarget;
		while (!(nnai1 != index && nnai1 != nnai0 && normalize(getFitnessForNNAi(m_trainee->getConstRefAt(nnai1)), minFitness, maxFitness) >= target)) {
			nnai1 = (*dist)(*dev);
			target -= TARGET_DECREASE_RATE;
		}
		assert(normalize(getFitnessForNNAi(m_trainee->getConstRefAt(nnai1)), minFitness, maxFitness) >= target);

		assert(index != nnai0);
		assert(index != nnai1);
		assert(nnai0 != nnai1);
		assert(nnai0 < m_trainee->getPopulationSize());
		assert(nnai1 < m_trainee->getPopulationSize());

		return NNAi<T>(m_trainee->assignNextID()
					, m_trainee->getConstRefAt(nnai0)
					, m_trainee->getConstRefAt(nnai1)
					, evolutionInfo);
	}

	inline void onSessionComplete(const std::vector<NNPPTrainingUpdate<T>>& scoreUpdates) {
		std::lock_guard<std::mutex> lock(m_onSessionCompleteMutex);
		updateScores(scoreUpdates);
		m_trainee->trainSessionCompleted();
		m_totalSessionsCompleted++;
	}

	inline void updateScores(const std::vector<NNPPTrainingUpdate<T>>& scoreUpdates) {
		for (const auto& [nnai, deltaScore, replace] : scoreUpdates) {
			assert(nnai);
			nnai->sessionCompleted();
			if (replace) {
				nnai->updateScoreReplace(deltaScore);
			}
			else {
				nnai->updateScoreDelta(deltaScore);
			}
		}
	}

	inline bool shouldEvolve() const {
		return sessionsTillEvolution() == 0;
	}

	inline float getFitnessForNNAi(const NNAi<T>& nnai) const {
		const float avgImportance = std::max(0.0f, std::min(1.0f, getAvgScoreImportance()));
		return avgImportance * nnai.getAvgScore() + (1.0f - avgImportance) * nnai.getScore();
	}

};

template <typename T> class TrainerDataSet {
public:
	NNPPStackVector<T> input;
	NNPPStackVector<T> expected;
	uint aiIndex;

	TrainerDataSet() : aiIndex(0) { }
	TrainerDataSet(const NNPPStackVector<T>& input, const NNPPStackVector<T>& expected, uint aiIndex) :
		input(input),
		expected(expected),
		aiIndex(aiIndex) { }
};

template <typename T> class SimpleTrainer : public NNPPTrainer<T> {
public:
	SimpleTrainer() = delete;
	SimpleTrainer(uint sessions, uint threads, NNPopulation<T>* population, const std::vector<TrainerDataSet<T>>& tests, const EvolutionInfo<T>& defaultEvolutionInfo)
			: NNPPTrainer<T>(sessions, threads, population, defaultEvolutionInfo)
			, m_tests(tests) { }

protected:
	std::vector<NNPPTrainingUpdate<T>> runSession(NeuronBuffer<T>& threadLocalNeuronBuffer) override {
		std::vector<NNPPTrainingUpdate<T>> updates;
		for (uint i = 0; i < NNPPTrainer<T>::m_trainee->getPopulationSize(); ++i) {
			NNAi<T>* nnai = NNPPTrainer<T>::m_trainee->getNNAiPtrAt(i);
			assert(nnai);
			if (nnai->getSessionsTrained() > 0) {
				continue;
			}
			float deltaScore = 0.0f;
			for (const auto& [input, exp, indx] : m_tests) {
				NNPPStackVector<T> out = nnai->feedAt(indx, input, threadLocalNeuronBuffer);
				assert(out.size() == exp.size());
				for (uint o = 0; o < exp.size(); ++o) {
					deltaScore += -std::abs(exp[o] - out[o]);
				}
			}
			updates.emplace_back(nnai, deltaScore, true);
			nnai->sessionCompleted();
		}
		return updates;
	}

	uint sessionsTillEvolution() const override {
		assert(NNPPTrainer<T>::m_trainee->getSessionsTrainedThisGen() <= 1);
		return NNPPTrainer<T>::m_trainee->getSessionsTrainedThisGen() - 1;
	}

private:
	std::vector<TrainerDataSet<T>> m_tests;
};
