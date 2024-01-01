using _Interfaces;

using SimpleNeuralNetwork.ActivationFunction;
using SimpleNeuralNetwork.Layer;

namespace SimpleNeuralNetwork
{
    public class NeuralLayerFactory
    {
        public NeuralLayer CreateNeuralLayer(int numberOfInputNeurons, IActivationFunction rectifiedActivationFunction, WeightedSumFunction weightedSumFunction)
        {
            NeuralLayer neuralLayer = new NeuralLayer();

            return neuralLayer;
        }
    }
}