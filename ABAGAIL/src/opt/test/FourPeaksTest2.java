package opt.test;

import java.util.*;
import java.text.*;

import util.linalg.Vector;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import opt.prob.*;
import shared.*;

/**
 * Optimization implemented through RHC, SA, GA, and MIMIC to find an approximate solution
 * to the continuous peaks problem.
 */
public class FourPeaksTest2 {

    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    private static final int maxNumIterations = 100000;

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"SA", "GA", "MIMIC"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        oa[0] = new SimulatedAnnealing(1E11, .95, hcp);
        oa[1] = new StandardGeneticAlgorithm(200, 100, 10, gap);
        oa[2] = new MIMIC(200, 20, pop);

        for(int i = 0; i < oa.length; i++)
            train(oa[i], ef, oaNames[i]);

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, EvaluationFunction ef, String oaName) {
        System.out.print("\n" + oaName + "\n");

        double optimal = 0.0, start = System.nanoTime(), stamp = 0, end, trainingTime, temp;
        for(int i = 0; i < maxNumIterations; i++) {
            oa.train();
            temp = ef.value(oa.getOptimal());
            System.out.println(temp);

            if(optimal < temp) {
                optimal = temp;
                stamp = System.nanoTime();
            }


        }
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);
        stamp -= start;
        stamp /= Math.pow(10,9);

        results += "\n\nResults for " + oaName + ":\nTraining time: " + df.format(trainingTime) + " seconds."
                + "\nOptimal instance found after " + stamp + " seconds.\nFinal optimal solution found: " + optimal;
    }

}
