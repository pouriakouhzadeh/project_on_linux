import random
from deap import base, creator, tools, algorithms
from celery import group
from tasks import train_model_task  # فرض بر این است که شما یک تسک Celery به نام train_model_task تعریف کرده‌اید
import pandas as pd 
import numpy as np
import deap
from TR_Coress_Validation import CrossValidation
from TR_MODEL import TrainModels
from TR_MODEL_MODEL import TrainModelsReturn
import pickle

CV = CrossValidation()
TR = TrainModels()
TRMM = TrainModelsReturn()

# تعریف creator برای ایجاد نوع فیتنس و نوع فرد
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)


if not hasattr(deap.creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def df_to_json(df):
    return df.to_json(orient='split')

def save_to_file(data, file_name="ga_results.txt"):
    with open(file_name, "a") as file:
        file.write(data + "\n")

def read_data(file_name, tail_size=7000):
    try:
        data = pd.read_csv(file_name)
        data = data.tail(tail_size)
        data.reset_index(inplace=True, drop=True)
        return data
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None
    
def create_task_for_individual(individual):
    currency_data = read_data("EURCHF60.csv", 7000)
    return train_model_task.s(df_to_json(currency_data), *individual[:-1], individual[-1])

# تعریف تابع برای ایجاد یک فرد
def generate_individual(param_space):
    individual = [random.randint(param[0], param[1]) for param in param_space[:-1]]
    allowed_hours_count = random.randint(param_space[-1][0], min(param_space[-1][1], 21))
    allowed_hours = random.sample(range(3, 24), allowed_hours_count)

    individual.append(allowed_hours)
    return creator.Individual(individual)


# تعریف toolbox
toolbox = base.Toolbox()
# def Train(self, data,  depth,  page,  feature,   QTY,         iter,        Thereshhold, primit_hours=[]):
param_space = [         (6, 12),(1, 20),(30, 800),(2000, 6000),(1000, 3000), (52, 75), (8, 18)]
toolbox.register("individual", generate_individual, param_space)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", create_task_for_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    population = toolbox.population(n=150)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 50


    tasks = group(create_task_for_individual(ind) for ind in population)
    results = tasks.apply_async()  # اجرای همزمان تمام تسک‌ها و دریافت یک GroupResult
    fitnesses = results.get()  # دریافت نتایج اجرای تسک‌ها

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit


    print(fitnesses)
    # شروع الگوریتم ژنتیک
    for g in range(NGEN):
        # انتخاب و تولید نسل جدید
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Cross-over
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # ارزیابی افرادی که تغییر کرده‌اند
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        tasks = group(create_task_for_individual(ind) for ind in invalid_ind)
        results = tasks.apply_async()  # اجرای همزمان تمام تسک‌ها و دریافت یک GroupResult
        fitnesses = results.get()


        # به‌روزرسانی فیتنس برای افراد invalid_ind با استفاده از نتایج به دست آمده
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print(fitnesses)
        print(f"Generation : {g}")
        # جایگزینی جمعیت قدیمی با نسل جدید
        population[:] = offspring

    best_individuals = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)[:10]

    # چاپ و ذخیره 10 بهترین
    for i, ind in enumerate(best_individuals, 1):
        print(f"Top {i} Individual = ", ind, "Fitness = ", ind.fitness.values[0])
        if ind.fitness.values[0] > 0.7 :
            currency_data = read_data("EURCHF60.csv", 7000)
            if CV.Train(df_to_json(currency_data), *ind[:-1], ind[-1]) > 0.7 :
                if TR.Train(df_to_json(currency_data), *ind[:-1], ind[-1]) > 0.7 :
                    model = TRMM.Train(df_to_json(currency_data), *ind[:-1], ind[-1])
                    with open('EURCHF60.pkl', 'wb') as file:
                        pickle.dump(model, file)
                    with open('EURCHF60_parameters.pkl', 'wb') as file:
                        pickle.dump(ind, file)

                    # باید اینجا فایل دیتا را روی فایل قبلی دیتا اور رایت کنیم
                    break    

        save_to_file(f"Top {i} Individual = {ind}, Fitness = {ind.fitness.values[0]}")

    return best_individuals

if __name__ == "__main__":
    best_individuals = main()
