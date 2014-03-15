__author__ = ('Travis Silvers, firestrand@gmail.com')
"""
 Standard Particle Swarm Optimization (SPSO) originally due to
 Eberhart et al. This the 'Standard PSO 2007' variant from Clerc.
 
 </summary>
 <remarks>
 References:
 (1) Eberhart, R. C. and Kennedy, J. A new optimizer using particle
     swarm theory. Proceedings of the Sixth International Symposium
     on Micromachine and Human Science, Nagoya, Japan pp. 39-43.
 (2) J. Kennedy and R. Eberhart. Particle swarm optimization.
     In Proceedings of IEEE International Conference on Neural
     Networks, volume IV, pages 1942-1948, Perth, Australia, 1995
 (3) Clerc, M. C Source Code downloaded from http://clerc.maurice.free.fr/pso/
"""
import scipy
import logging
import math
from pybrain.optimization.optimizer import ContinuousOptimizer


def fullyConnected(lst):
    return dict((i, lst) for i in lst)

def ring(lst):
    leftist = lst[1:] + lst[0:1]
    rightist = lst[-1:] + lst[:-1]
    return dict((i, (j, k)) for i, j, k in zip(lst, leftist, rightist))

# TODO: implement some better neighborhoods


class StandardParticleSwarmOptimizer(ContinuousOptimizer):
    """ Standard Particle Swarm Optimization
    S := swarm size
    K := maximum number of particles _informed_ by a given one
    p := probability threshold of random topology, typically calculated from K
    w := first cognitive/confidence coefficient
    c := second cognitive/confidence coefficient
    """
    S = 50
    K = 3
    p = 1
    w = 0.72984
    c = 1.193

    boundaries = None

    memory = 2.0
    sociality = 2.0
    inertia = 0.9

    neighbourfunction = None

    mustMaximize = True

    def _setInitEvaluable(self, evaluable):
        if evaluable is not None:
            logging.warning("Initial point provided was ignored.")
        ContinuousOptimizer._setInitEvaluable(self, evaluable)

    def _additionalInit(self):
        self.dim = self.numParameters
        if self.neighbourfunction is None:
            self.neighbourfunction = fullyConnected

        if self.boundaries is None:
            maxs = scipy.array([10] * self.dim)
            mins = scipy.array([-10] * self.dim)
        else:
            mins = scipy.array([min_ for min_, max_ in self.boundaries])
            maxs = scipy.array([max_ for min_, max_ in self.boundaries])

        self.particles = []
        for _ in xrange(self.size):
            startingPosition = scipy.random.random(self.dim)
            startingPosition *= (maxs - mins)
            startingPosition += mins
            self.particles.append(Particle(startingPosition, self.minimize))

        # Global neighborhood
        self.neighbours = self.neighbourfunction(self.particles)

    def best(self, particlelist):
        """Return the particle with the best fitness from a list of particles.
        """
        picker = min if self.minimize else max
        return picker(particlelist, key=lambda p: p.fitness)
"""
        public override Result Optimize(double[] parameters)
        {
            Debug.Assert(parameters != null && parameters.Length == Dimensionality);

            // Retrieve parameter specific to SPSO method.
            int S = (int)System.Math.Round(parameters[0], System.MidpointRounding.AwayFromZero);
            Debug.Assert(S > 0);

            double p = parameters[2]; //This is what matters for informed particles
            double w = parameters[3];
            double c = parameters[4];

            //Initialize Random for each particle
            Random rand = new Random();

            // Get problem-context.
            double[] lowerBound = Problem.LowerBound;
            double[] upperBound = Problem.UpperBound;
            double[] lowerInit = Problem.LowerInit;
            double[] upperInit = Problem.UpperInit;
            int n = Problem.Dimensionality;

            // Allocate agent positions and associated fitnesses.
            double[][] agents = Tools.NewMatrix(S, n);
            double[][] velocities = Tools.NewMatrix(S, n);
            double[][] bestAgentPosition = Tools.NewMatrix(S, n);

            int[,] links = new int[S, S];
            int[] index = new int[S];
            int g;
            double[] px = new double[n];
            double[] gx = new double[n];
            int nEval = 0;

            double[] agentFitness = new double[S];
            double[] bestAgentFitness = new double[S];

            // Initialize
            // Initialize all agents.
            // This counts as iterations below.
            // Position and velocity
            for (int s = 0; s < S; s++)
            {
                for (int d = 0; d < n; d++)
                {
                    agents[s][d] = rand.NextDouble(lowerInit[d], upperInit[d]);
                    velocities[s][d] = (rand.NextDouble(lowerBound[d], upperBound[d]) - agents[s][d]) / 2;
                }
            }

            // First evaluations
            for (int s = 0; s < S; s++)
            {
                agentFitness[s] = Problem.Fitness(agents[s]);
                nEval++;
                agents[s].CopyTo(bestAgentPosition[s], 0);	// Best position = current one
                bestAgentFitness[s] = agentFitness[s];
            }

            // Find the best
            int best = 0;
            double errorPrev = bestAgentFitness[best];

            for (int s = 1; s < S; s++)
            {
                if (bestAgentFitness[s] < errorPrev)
                {
                    best = s;
                    errorPrev = bestAgentFitness[s];
                }
            }

            int initLinks = 1;		// So that information links will beinitialized
            int noStop = 0;
            // ---------------------------------------------- ITERATIONS
            while (noStop == 0)
            {
                if (initLinks == 1)	// Random topology
                {
                    // Who informs who, at random
                    for (int s = 0; s < S; s++)
                    {
                        for (int m = 0; m < S; m++)
                        {
                            if (rand.NextDouble() < p) links[m, s] = 1;	// Probabilistic method
                            else links[m, s] = 0;
                        }
                        links[s, s] = 1;
                    }
                }

                // The swarm MOVES
                for (int i = 0; i < S; i++)
                    index[i] = i;

                for (int s0 = 0; s0 < S; s0++)	// For each particle ...
                {
                    int s = index[s0];
                    // ... find the first informant
                    int s1 = 0;
                    while (links[s1, s] == 0) s1++;
                    if (s1 >= S) s1 = s;

                    // Find the best informant
                    g = s1;
                    for (int m = s1; m < S; m++)
                    {
                        if (links[m, s] == 1 && bestAgentFitness[m] < bestAgentFitness[g])
                            g = m;
                    }

                    //.. compute the new velocity, and move
                    // Exploration tendency
                    if(g!=s)
                    {
                        for (int d = 0; d < n; d++)
                        {
                            velocities[s][d] = w*velocities[s][d];
                            px[d] = bestAgentPosition[s][d] - agents[s][d];
                            gx[d] = bestAgentPosition[g][d] - agents[s][d];
                            velocities[s][d] += rand.NextDouble(0.0, c) * px[d];
                            velocities[s][d] += rand.NextDouble(0.0, c) * gx[d];
                            agents[s][d] = agents[s][d] + velocities[s][d];
                        }
                    }
                    else
                    {
                        for (int d = 0; d < n; d++)
                        {
                            velocities[s][d] = w * velocities[s][d];
                            px[d] = bestAgentPosition[s][d] - agents[s][d];
                            velocities[s][d] += rand.NextDouble(0.0, c) * px[d];
                            agents[s][d] = agents[s][d] + velocities[s][d];
                        }
                    }

                    if (!Problem.RunCondition.Continue(nEval, bestAgentFitness[best]))
                    {
                        //error= fabs(error - pb.objective);
                        goto end;
                    }

                    for (int d = 0; d < n; d++)
                    {
                        if (agents[s][d] < lowerBound[d])
                        {
                            agents[s][d] = lowerBound[d];
                            velocities[s][d] = 0;
                        }

                        if (agents[s][d] > upperBound[d])
                        {
                            agents[s][d] = upperBound[d];
                            velocities[s][d] = 0;
                        }
                    }

                    agentFitness[s] = Problem.Fitness(agents[s]);
                    nEval++;
                    // ... update the best previous position
                    if (agentFitness[s] < bestAgentFitness[s])	// Improvement
                    {
                        agents[s].CopyTo(bestAgentPosition[s], 0);
                        bestAgentFitness[s] = agentFitness[s];
                        // ... update the best of the bests
                        if (bestAgentFitness[s] < bestAgentFitness[best])
                        {
                            best = s;
                        }
                    }
                }			// End of "for (s0=0 ...  "
                // Check if finished
                initLinks = bestAgentFitness[best] < errorPrev ? 0 : 1;
                errorPrev = bestAgentFitness[best];
                // Trace fitness of best found solution.
                Trace(nEval, bestAgentFitness[best]);
            end:
                noStop = Problem.RunCondition.Continue(nEval, bestAgentFitness[best]) ? 0 : 1;
            } // End of "while nostop ...

            // Return best-found solution and fitness.
            return new Result(bestAgentPosition[best], bestAgentFitness[best], nEval);
"""
    def _learnStep(self):
        for particle in self.particles:
            particle.fitness = self._oneEvaluation(particle.position.copy())

        for particle in self.particles:
            bestPosition = self.best(self.neighbours[particle]).position
            diff_social = self.sociality \
                          * scipy.random.random() \
                          * (bestPosition - particle.position)

            diff_memory = self.memory \
                          * scipy.random.random() \
                          * (particle.bestPosition - particle.position)

            particle.velocity *= self.inertia
            particle.velocity += diff_memory + diff_social
            particle.move()

    def calculate_parameters(self, dimensions, num_informed):
        self.S = 10 + 2 * math.sqrt(dimensions) #Swarm size
        self.K = num_informed #number of informed particles
        self.p = 1. - (1. - 1. / float(self.S))**self.K #Probability threshold of random topology
        #(to simulate the global best PSO, set p=1)

        #According to Clerc's Stagnation Analysis
        self.w = 1. / (2. * math.log(2.0))#0.721
        self.c = 0.5 + math.log(2.0)#1.193

    @property
    def batchSize(self):
        return self.size


class Particle(object):
    def __init__(self, start, minimize):
        """Initialize a Particle at the given start vector."""
        self.minimize = minimize
        self.dim = scipy.size(start)
        self.position = start
        self.velocity = scipy.zeros(scipy.size(start))
        self.bestPosition = scipy.zeros(scipy.size(start))
        self._fitness = None
        if self.minimize:
            self.bestFitness = scipy.inf
        else:
            self.bestFitness = -scipy.inf

    def _setFitness(self, value):
        self._fitness = value
        if ((self.minimize and value < self.bestFitness)
            or (not self.minimize and value > self.bestFitness)):
            self.bestFitness = value
            self.bestPosition = self.position.copy()

    def _getFitness(self):
        return self._fitness

    fitness = property(_getFitness, _setFitness)

    def move(self):
        self.position += self.velocity

