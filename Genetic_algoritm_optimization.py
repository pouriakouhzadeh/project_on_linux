import random
from deap import base, creator, tools
from celery import group
from TR_Coress_Validation import CrossValidation
from TR_MODEL import TrainModels
from TR_MODEL_MODEL import TrainModelsReturn
from tasks import train_model_task
import pandas as pd
import pickle

class GeneticAlgorithm:
    def __init__(self):
        self.CV = CrossValidation()
        self.TR = TrainModels()
        self.TRMM = TrainModelsReturn()

        # Initialize DEAP creator for fitness and individual types
        self._initialize_deap()

        self.toolbox = base.Toolbox()
        # Define parameter space
        param_space = [(2, 12), (2, 30), (30, 800), (2000, 6000), (1000, 3000), (52, 75), (8, 18)]
        # Register functions
        self.toolbox.register("individual", self.generate_individual, param_space)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.create_task_for_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _initialize_deap(self):
        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def df_to_json(self, df):
        return df.to_json(orient='split')

    def save_to_file(self, data, file_name="ga_results.txt"):
        with open(file_name, "a") as file:
            file.write(data + "\n")

    def read_data(self, file_name, tail_size=7000):
        try:
            data = pd.read_csv(file_name)
            data = data.tail(tail_size)
            data.reset_index(inplace=True, drop=True)
            return data
        except FileNotFoundError:
            print(f"File {file_name} not found.")
            return None

    def create_task_for_individual(self, individual, currency_pair):
        currency_data = self.read_data(f"{currency_pair}.csv", 7000)
        return train_model_task.s(self.df_to_json(currency_data), *individual[:-1], individual[-1])

    def generate_individual(self, param_space):
        individual = [random.randint(param[0], param[1]) for param in param_space[:-1]]
        allowed_hours_count = random.randint(param_space[-1][0], min(param_space[-1][1], 21))
        allowed_hours = random.sample(range(3, 24), allowed_hours_count)
        individual.append(allowed_hours)
        return creator.Individual(individual)

    def local_search(self, individual):
        # Perform hill climbing local search
        best_fitness = self.toolbox.evaluate(individual)
        improved = True
        while improved:
            improved = False
            for i in range(len(individual)):
                old_value = individual[i]
                individual[i] = max(0, old_value - 1)  # Make a small change
                fitness = self.toolbox.evaluate(individual)
                if fitness > best_fitness:
                    best_fitness = fitness
                    improved = True
                else:
                    individual[i] = old_value  # Revert change
        return individual,

    def main(self, currency_pair, pop, NG):
        population = self.toolbox.population(n=pop)
        CXPB, MUTPB, NGEN = 0.5, 0.2, NG

        tasks = group(self.create_task_for_individual(ind, currency_pair) for ind in population)
        results = tasks.apply_async()
        fitnesses = results.get()

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        print(f"{currency_pair} >>> Population : {pop} & Generation : 0/{NG} & Best fitness = {max(fitnesses)}")
        for g in range(NGEN):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                self.local_search(ind)  # Apply local search

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            tasks = group(self.create_task_for_individual(ind) for ind in invalid_ind)
            results = tasks.apply_async()
            fitnesses = results.get()

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print(f"{currency_pair} >>> Population : {pop} & Generation : {g}/{NG} & Best fitness = {max(fitnesses)}")

            population[:] = offspring

        best_individuals = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)[:10]
        print("Start calculate CrossValidation and save model...")
        for i, ind in enumerate(best_individuals, 1):
            print(f"Top {i} Individual = ", ind, "Fitness = ", ind.fitness.values[0])
            if ind.fitness.values[0] > 0.7 :
                currency_data = self.read_data(f"{currency_pair}.csv", 7000)
                CrosValidation = self.CV.Train(self.df_to_json(currency_data.copy()), *ind[:-1], ind[-1])
                print(f"CrosValidation = {CrosValidation}")
                if  CrosValidation > (0.7, ) :
                    TrainModel = self.TR.Train(self.df_to_json(currency_data.copy()), *ind[:-1], ind[-1])
                    print(f"TrainModel = {TrainModel}")
                    if  TrainModel > (0.7, ) :
                        model = self.TRMM.Train(self.df_to_json(currency_data.copy()), *ind[:-1], ind[-1])
                        with open(f"/home/pouria/project/trained_models/{currency_pair}.pkl", 'wb') as file:
                            pickle.dump(model, file)
                        with open(f"/home/pouria/project/trained_models/{currency_pair}_parameters.pkl", 'wb') as file:
                            pickle.dump(ind, file)
                        print(f"Model for {currency_pair} saved successfully")
                        return True

            self.save_to_file(f"Top {i} Individual: {ind}, Fitness: {ind.fitness.values[0]}")

        return False

