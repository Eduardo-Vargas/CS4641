package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

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
public class AbaloneTestHeart {
    private static int NUM_ATTR = 13;
    private static int NUM_INST = 270;

    private static Instance[] instances = initializeInstances();

    private static int inputLayer = NUM_ATTR, hiddenLayer = 10, outputLayer = 1, trainingIterations = 200;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();


    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static String fileName = "junk.txt";
    private static Date date;

    private static BufferedWriter bufferedWriter;


    public static void main(String[] args) {
//

        /*for(int loops=0; loops<1; loops++) {
            if(loops == 0) {
                hiddenLayer = 10;
                trainingIterations = 200;
            }
            else if(loops == 1) {
                hiddenLayer = 25;
                trainingIterations = 200;
            }
            else if(loops == 2) {
                hiddenLayer = 35;
                trainingIterations = 200;
            }
            else if(loops == 3) {
                hiddenLayer = 50;
                trainingIterations = 200;
            }
            else if(loops == 4) {
                hiddenLayer = 75;
                trainingIterations = 200;
            }
            else if(loops == 5) {
                hiddenLayer = 100;
                trainingIterations = 200;
            }*/
            fileName = "Heart" + hiddenLayer + "HL" + trainingIterations + "E.txt";


            try {
                FileWriter writer = new FileWriter(fileName, false);
                bufferedWriter = new BufferedWriter(writer);
            } catch (IOException e) {
                e.printStackTrace();
            }



        date = new Date();
        System.out.println(date.toString());
        System.out.println("Working on: " + fileName);
        System.out.println();
        results = "";




        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
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

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }

            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

//        System.out.println(results);
//
        try {
            bufferedWriter.write(results);
            bufferedWriter.newLine();
            bufferedWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }


        }





    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
//        System.out.println("\nError results for " + oaName + "\n---------------------------");

        try {
            bufferedWriter.write("\nError results for " + oaName + "\n---------------------------");
            bufferedWriter.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }




        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
//                if(i==0 && j<10) { System.out.println("----"); System.out.println(instances[j]); System.out.println(output); System.out.println(example); }
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
//                if(i==0 && j<10) { System.out.println("++++"); System.out.println(instances[j]); System.out.println(output); System.out.println(example); }

                //error -= instances[j].getLabel() * java.lang.Math.log(( network.getOutputValues())) - (1.0 - instances[j].getLabel()[0]) * java.lang.Math.log((1.0 - network.getOutputValues()[0]));

                //System.out.println(instances[j].getLabel());
                //System.out.println(network.getOutputValues());

                error += measure.value(output, example);

                //error += measure.value(output, example);
//                if(i==0 && j<10) { System.out.println("====" + error); }
            }
              //System.out.println(df.format(error));
//            System.out.println(df.format(error));

        try {
            bufferedWriter.write(df.format(error));
            bufferedWriter.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }


        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[NUM_INST][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/heart-data.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[NUM_ATTR]; // NUM_ATTR attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < NUM_ATTR; j++)
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
            instances[i].setLabel(new Instance((attributes[i][1][0]) - 1));
        }

        return instances;
    }


}
