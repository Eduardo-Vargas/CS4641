package shared;



/**
 * Standard error measure, suitable for use with
 * linear output networks for regression, sigmoid
 * output networks for single class probability,
 * and soft max networks for multi class probabilities.
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CrossEntropyError extends AbstractErrorMeasure {

    /**
     * @see nn.error.ErrorMeasure#error(double[], nn.Pattern[], int)
     */
    public double value(Instance output, Instance example) {
        double actual = Double.parseDouble(output.toString());
        double predicted = Double.parseDouble(example.getLabel().toString());

        double entropyLoss = ((actual * Math.log(predicted)) + ((1 - actual) * Math.log(predicted))) / 270;


        return -entropyLoss;
      }
}
