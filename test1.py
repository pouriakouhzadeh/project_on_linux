import random
import logging
from deap import base, creator, tools

class GeneticAlgorithm:
    def __init__(self):
        self._initialize_deap()
        self.toolbox = base.Toolbox()
        self.param_space = [(2, 12), (2, 30), (30, 800), (12000, 20000), (1000, 3000), (52, 75), (8, 18)]
        self.toolbox.register("individual", self.generate_individual, self.param_space)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def _initialize_deap(self):
        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def generate_individual(self, param_space):
        individual = []
        for param in param_space[:-1]:
            if isinstance(param[0], (bool, str)):
                individual.append(random.choice(param))
            else:
                try:
                    if param[0] > param[1]:
                        raise ValueError(f"Invalid range: ({param[0]}, {param[1]})")
                    individual.append(random.randint(param[0], param[1]))
                except ValueError as e:
                    logging.error(f"Error with parameter range: {e}")
                    return None

        last_param = param_space[-1]
        if last_param[0] > last_param[1]:
            logging.error(f"Invalid range for hours: ({last_param[0]}, {last_param[1]})")
            return None

        allowed_hours_count = random.randint(last_param[0], last_param[1])
        allowed_hours_set = set(random.sample(range(3, 24), allowed_hours_count))

        individual.append(list(allowed_hours_set))
        return creator.Individual(individual)

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    individual = ga.toolbox.individual()
    print(individual)
