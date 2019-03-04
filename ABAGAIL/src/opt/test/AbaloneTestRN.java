package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import shared.reader.DataSetLabelBinarySeperator;


import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneTestRN {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 16, hiddenLayer = 1000, outputLayer = 26, trainingIterations = 100;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        //DataSetLabelBinarySeperator.seperateLabels(set);

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < 2; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();





                predicted = getMax(convertStringToDoubleArray(instances[j].getLabel().toString()));//Double.parseDouble(instances[j].getLabel().toString());
                actual = getMax(convertStringToDoubleArray(networks[i].getOutputValues().toString()));//Double.parseDouble(networks[i].getOutputValues().toString());


                if (predicted==actual){
                	correct++;
                } else {
                	incorrect++;
                }

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(network.getOutputValues())); //Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);

            }

            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {
        double[][][] attributes = new double[20000][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/letter-data.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[16]; // 16 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 16; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);


						int c = (int) attributes[i][1][0];

            // Create a double array of length 10, all values are initialized to 0
            double[] classes = new double[26];

            // Set the i'th index to 1.0
            classes[c] = 1.0;
            instances[i].setLabel(new Instance(classes));
        }

        return instances;
    }


    private static double[] convertStringToDoubleArray(String input) {
    	String[] split = input.split(",");
    	double[] result = new double[split.length];
    	for (int i=0; i<split.length; i++){
    		result[i] =  Double.parseDouble(split[i]);
    	}
    	return result;
    }

    private static int getMax(double[] input) {
    	double max = input[0];
    	int index = 0;
    	for (int i = 0; i<input.length; i++) {
    		if (input[i] > max) {
    			max = input[i];
    			index = i;
    		}
    	}
    	return index;
    }
}
