using SimpleNeuralNetwork.ActivationFunction;
using SimpleNeuralNetwork.Layer;

namespace SimpleNeuralNetwork
{
    public class NeuralNetwork
    {
        private NeuralLayerFactory _layerFactory;

        internal List<NeuralLayer> _layers;
        internal double _learningRate;
        internal double[][] _expectedResult;

        public NeuralNetwork(int numberOfInputNeurons)
        {
            _layers = new List<NeuralLayer>();
            _layerFactory = new NeuralLayerFactory();

            // Create input layer that will collect inputs.
            CreateInputLayer(numberOfInputNeurons);

            _learningRate = 2.95;
        }

        private void CreateInputLayer(int numberOfInputNeurons)
        {
            var inputLayer = _layerFactory.CreateNeuralLayer(numberOfInputNeurons, new RectifiedActivationFuncion(), new WeightedSumFunction());
            inputLayer.Neurons.ForEach(x => x.AddInputSynapse(0));
            AddLayer(inputLayer);
        }

        public void AddLayer(NeuralLayer newLayer)
        {
            if (_layers.Any())
            {
                var lastLayer = _layers.Last();
                newLayer.ConnectLayers(lastLayer);
            }

            _layers.Add(newLayer);
        }

        public void Train(double[][] inputs, int numberOfEpochs)
        {
            double totalError = 0;

            for (int i = 0; i < numberOfEpochs; i++)
            {
                for (int j = 0; j < inputs.GetLength(0); j++)
                {
                    PushInputValues(inputs[j]);

                    var outputs = new List<double>();

                    // Get outputs.
                    _layers.Last().Neurons.ForEach(x =>
                    {
                        outputs.Add(x.CalculateOutput());
                    });

                    // Calculate error by summing errors on all output neurons.
                    totalError = CalculateTotalError(outputs, j);
                    HandleOutputLayer(j);
                    HandleHiddenLayers();
                }
            }
        }

        private double CalculateTotalError(IList<double> outputs, int row)
        {
            double totalError = 0;

            outputs.ToList().ForEach(output =>
            {
                var error = Math.Pow(output - _expectedResult[row][outputs.IndexOf(output)], 2);
                totalError += error;
            });

            return totalError;
        }

        private void HandleOutputLayer(int row)
        {
            _layers.Last().Neurons.ForEach(neuron =>
            {
                neuron.Inputs.ForEach(connection =>
                {
                    var output = neuron.CalculateOutput();
                    var netInput = connection.GetOutput();

                    var expectedOutput = _expectedResult[row][_layers.Last().Neurons.IndexOf(neuron)];

                    var nodeDelta = (expectedOutput - output) * output * (1 - output);
                    var delta = -1 * netInput * nodeDelta;

                    connection.UpdateWeight(_learningRate, delta);

                    neuron.PreviousPartialDerivate = nodeDelta;
                });
            });
        }

        private void HandleHiddenLayers()
        {
            for (int k = _layers.Count - 2; k > 0; k--)
            {
                _layers[k].Neurons.ForEach(neuron =>
                {
                    neuron.Inputs.ForEach(connection =>
                    {
                        var output = neuron.CalculateOutput();
                        var netInput = connection.GetOutput();
                        double sumPartial = 0;

                        _layers[k + 1].Neurons
                        .ForEach(outputNeuron =>
                        {
                            outputNeuron.Inputs.Where(i => i.IsFromNeuron(neuron.Id))
                            .ToList()
                            .ForEach(outConnection =>
                            {
                                sumPartial += outConnection.PreviousWeight * outputNeuron.PreviousPartialDerivate;
                            });
                        });

                        var delta = -1 * netInput * sumPartial * output * (1 - output);
                        connection.UpdateWeight(_learningRate, delta);
                    });
                });
            }
        }

        public void PushExpectedValues(double[][] expectedOutputs)
        {
            _expectedResult = expectedOutputs;
        }

        public void PushInputValues(double[] inputs)
        {
            _layers.First().Neurons.ForEach(x => x.PushValueOnInput(inputs[_layers.First().Neurons.IndexOf(x)]));
        }

        public IList<double> GetOutput()
        {
            var returnValue = new List<double>();

            _layers.Last().Neurons.ForEach(neuron =>
            {
                returnValue.Add(neuron.CalculateOutput());
            });

            return returnValue;
        }
    }
}