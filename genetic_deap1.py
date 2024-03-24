from deap import base, creator, tools
import random
import multiprocessing
import pandas as pd
from tasks import train_model_task  # فرض می‌کنیم این تابع در فایل tasks.py تعریف شده است

# فرض بر اینکه currency_files و param_space به درستی تعریف شده‌اند
currency_files = [
    "EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv",
    "AUDNZD60.csv", "AUDUSD60.csv", "EURAUD60.csv",
    "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv",
    "USDCAD60.csv", "USDCHF60.csv"
]
param_space = [(2, 10), (2, 20), (30, 500), (500, 6000), (100, 5000), (52, 75), (8, 18)]

def save_to_file(data, file_name="ga_results.txt"):
    with open(file_name, "a") as file:
        file.write(data + "\n")

def read_currency_data_to_json(currency_file_name):
    try:
        data = pd.read_csv(currency_file_name)
        return data.to_json(orient='split')
    except FileNotFoundError:
        print(f"File {currency_file_name} not found.")
        return None

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# تعریف یک تابع attribute generator
toolbox.register("attr_int", random.randint, 2, 20)

# اولیه سازی Individual و Population
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: [random.randint(*param) for param in param_space])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalModel(individual):
    # محاسبه فیتنس برای یک individual
    # در اینجا نحوه ارسال ورودی به train_model_task بررسی می‌شود
    results = []
    for currency_file in currency_files:
        data_json = read_currency_data_to_json(currency_file)
        if data_json:
            # ارسال درخواست آموزش و دریافت نتیجه
            result = train_model_task(data_json, *individual[:-1],individual[-1])  # فرض بر اینکه این تابع به صورت همگام اجرا شود و نتیجه را برگرداند
            results.append(result)

    # محاسبه میانگین نتایج به دست آمده از تمام فایل‌ها
    if results:
        average_result = sum(results) / len(results)
        return (average_result,)
    else:
        return (0,)

# تنظیمات GA
toolbox.register("evaluate", evalModel)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=min(param_space)[0], up=max(param_space)[1], indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_evolution():
    # تنظیم pool برای multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # ایجاد جمعیت اولیه
    population = toolbox.population(n=50)  # تعداد جمعیت را برای تست کمتر در نظر گرفته‌ام
    NGEN = 10  # تعداد نسل‌ها
    CXPB, MUTPB = 0.5, 0.2

    # اجرای الگوریتم GA
    for gen in range(NGEN):
        # انتخاب و تولید نسل جدید
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # ارزیابی فرزندان جدید که فیتنس آن‌ها به‌روزرسانی نشده است
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # جایگزینی جمعیت قدیمی با نسل جدید
        population[:] = offspring

    pool.close()
    pool.join()  # اطمینان از بسته شدن تمامی پروسس‌ها
    return population

if __name__ == "__main__":
    population = run_evolution()
    best_ind = tools.selBest(population, 1)[0]
    print(f"Best individual is {best_ind}, with fitness: {best_ind.fitness.values}")
