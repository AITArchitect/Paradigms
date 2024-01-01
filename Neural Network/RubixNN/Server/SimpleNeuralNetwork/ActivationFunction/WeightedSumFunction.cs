using _Interfaces;

namespace SimpleNeuralNetwork.ActivationFunction
{
    public class WeightedSumFunction : IInputFunction
    {
        public double CalculateInput(IList<ISynapse> inputs)
        {
            return inputs.Select(x => x.Weight * x.GetOutput()).Sum();
        }
    }
}
