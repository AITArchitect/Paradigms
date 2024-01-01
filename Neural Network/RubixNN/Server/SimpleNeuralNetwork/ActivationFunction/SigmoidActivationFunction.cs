using _Interfaces;

namespace SimpleNeuralNetwork.ActivationFunction
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        private double _coeficient;

        public SigmoidActivationFunction(double coeficient)
        {
            _coeficient = coeficient;
        }

        public double CalculateOutput(double input)
        {
            return (1 / (1 + Math.Exp(-input * _coeficient)));
        }
    }
}
