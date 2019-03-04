package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
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
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest2 {
    /** The n value */
    private static final int N = 50;

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"SA", "GA", "MIMIC"};
    private static String results = "";
    private static final int maxNumIterations = 5000;
    private static DecimalFormat df = new DecimalFormat("0.000");
    private static DecimalFormat df1 = new DecimalFormat("0.0000");

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        // RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        // FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        // fit.train();
        // System.out.println(ef.value(rhc.getOptimal()));

        // SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        // fit = new FixedIterationTrainer(sa, 200000);
        // fit.train();
        // System.out.println(ef.value(sa.getOptimal()));

        // StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
        // fit = new FixedIterationTrainer(ga, 1000);
        // fit.train();
        // System.out.println(ef.value(ga.getOptimal()));

        // for mimic we use a sort encoding
        // ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        // odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        // MIMIC mimic = new MIMIC(200, 100, pop);
        // fit = new FixedIterationTrainer(mimic, 1000);
        // fit.train();
        // System.out.println(ef.value(mimic.getOptimal()));

        oa[0] = new SimulatedAnnealing(1E12, .95, hcp);
        oa[1] = new StandardGeneticAlgorithm(200, 150, 20, gap);
        oa[2] = new MIMIC(200, 34, pop);

        for(int i = 2; i < oa.length; i++)
            train(oa[i], ef, oaNames[i]);

        System.out.println(results);

    }

    private static void train(OptimizationAlgorithm oa, EvaluationFunction ef, String oaName) {
        System.out.print("\n" + oaName + "\n");

        double optimal = 0.0, start = System.nanoTime(), stamp = 0, end, trainingTime, temp;
        for(int i = 1; i < maxNumIterations - 1; i++) {
            oa.train();
            temp = ef.value(oa.getOptimal());
            System.out.println(df1.format(temp));

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
