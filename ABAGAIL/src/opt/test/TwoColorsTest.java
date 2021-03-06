package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author Daniel Cohen dcohen@gatech.edu
 * @version 1.0
 */
public class TwoColorsTest {
    /** The number of colors */
    private static final int k = 2;
    /** The N value */
    private static final int N = 100*k;



    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, k+1);
        EvaluationFunction ef = new TwoColorsEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        FixedIterationTrainer fit;
        /*
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 100);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));
        */
        for (int i = 0; i<10;i++) {
            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .5, hcp);
            fit = new FixedIterationTrainer(sa, 1000);
            fit.train();
            System.out.println(ef.value(sa.getOptimal()));
        }
        for (int i = 0; i<10; i++) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()));
        }
        for (int i = 0; i<10;i++) {
            MIMIC mimic = new MIMIC(200, 20, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            System.out.println(ef.value(mimic.getOptimal()));
        }
    }
}
