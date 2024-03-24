import random
import pandas as pd
from celery import group
from tasks import train_model_task

currency_files = [
    "EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv",
    "AUDNZD60.csv", "AUDUSD60.csv", "EURAUD60.csv",
    "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv",
    "USDCAD60.csv", "USDCHF60.csv"
]

def read_data(file_name, tail_size=7000):
    try:
        data = pd.read_csv(file_name)
        return data.tail(tail_size)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None



def generate_individual(param_space):
    # ایجاد قسمت‌های اصلی individual با استفاده از مقادیر تصادفی بر اساس param_space
    individual = [random.randint(param[0], param[1]) for param in param_space[:-1]]
    
    # ایجاد لیست allowed_hours به صورت تصادفی
    # توجه داشته باشید که این قسمت آخرین عنصر individual را تشکیل می‌دهد
    allowed_hours = random.sample(range(3, 24), random.randint(param_space[-1][0], param_space[-1][1]))
    
    # افزودن allowed_hours به عنوان آخرین عنصر individual
    individual.append(allowed_hours)
    
    return individual


def crossover(parent1, parent2):
    if len(parent1) < 3 or len(parent2) < 3:
        return random.choice([parent1, parent2])
    crossover_point = random.randint(1, len(parent1) - 2)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child



# def mutate(individual, mutation_rate, param_space):
#     for i in range(len(individual) - 1):
#         if random.random() < mutation_rate:
#             individual[i] = random.randint(param_space[i][0], param_space[i][1])
#     if random.random() < mutation_rate:
#         if isinstance(individual[-1], list) and individual[-1]:
#             new_hour = random.choice(list(set(range(3, 24)) - set(individual[-1])))
#             if len(individual[-1]) < param_space[-1][1]:
#                 individual[-1].append(new_hour)
#             else:
#                 individual[-1][random.randint(0, len(individual[-1]) - 1)] = new_hour
#         else:
#             print("Error: expected individual[-1] to be a list, got", type(individual[-1]))
#     return individual

def mutate(individual, mutation_rate, param_space):
    # جهش برای تمام آرگومان‌های به جز 'allowed_hours'
    for i in range(len(individual) - 1):
        if random.random() < mutation_rate:
            individual[i] = random.randint(param_space[i][0], param_space[i][1])
    
    # بررسی و جهش برای 'allowed_hours'
    if random.random() < mutation_rate:
        # اطمینان حاصل کنید که individual[-1] به درستی یک لیست است
        if isinstance(individual[-1], list):
            allowed_hours = individual[-1]
            
            # تصمیم‌گیری برای اضافه کردن یا حذف یک ساعت تصادفی
            if random.random() < 0.5:
                # تلاش برای اضافه کردن ساعت جدید، اگر حداکثر طول رعایت نشده باشد
                if len(allowed_hours) < param_space[-1][1]:
                    new_hour = random.choice(list(set(range(3, 24)) - set(allowed_hours)))
                    allowed_hours.append(new_hour)
            else:
                # تلاش برای حذف یک ساعت، اگر بیش از یک ساعت وجود داشته باشد
                if len(allowed_hours) > 1:
                    allowed_hours.remove(random.choice(allowed_hours))
            
            individual[-1] = allowed_hours
        else:
            print("Error: individual[-1] is not a list. It is:", type(individual[-1]))
            # در صورت خطا، بازگرداندن individual بدون تغییر
            return individual
    
    return individual



def save_to_file(data, file_name="ga_results.txt"):
    with open(file_name, "a") as file:
        file.write(data + "\n")



def evaluate_population(population, currency_files):
    results = []  # نتایج برای هر فرد
    tasks = []
    fitness_scores = []  # اینجا یک لیست برای نگهداری امتیازات ایجاد می‌کنیم

    # حلقه بر روی جمعیت
    for individual in population:
        # total_acc = 0
        # valid_models = 0
        # حلقه بر روی فایل‌های ارز
        for currency_file in currency_files:
            currency_data = read_data(currency_file, 7000)
            if currency_data is not None:
                temp = currency_data.to_json(orient='split')
                # اطمینان حاصل کنید که individual به صورت مناسبی آرگومان‌ها را برای train_model_task فراهم می‌کند
                # فرض بر این است که individual[-1] حاوی 'allowed_hours' است و بقیه عناصر آرگومان‌های دیگر را شامل می‌شوند
                task = train_model_task.s(temp, *individual[:-1], individual[-1])
                tasks.append(task)


    job = group(tasks)()
    results = job.get()  # results حاوی نتایج بازگشتی از هر کار است

    for result in results:
        try:
            # فرض می‌شود که هر نتیجه به صورت یک تاپل از (acc, wins, loses) بازگردانده شده است
            acc, wins, loses = result
            # این قسمت باید بر اساس ساختار دقیق نتایج بازگشتی از تابع train_model_task شما تنظیم شود
            if wins + loses >= 0.2 * (individual[3] * 0.033):  # این شرط باید بر اساس لاجیک مورد نظر شما تنظیم شود
                fitness_scores.append(acc) 
            else:
                 fitness_scores.append(0)    
        except Exception as e:
            print(f"Error processing task: {e}")


    print("stage finish")
    return fitness_scores  # برگرداندن لیست امتیازات دقت برای هر فرد در جمعیت



def genetic_algorithm(currency_files, population_size, generations, mutation_rate, param_space):
    print("Genetic algoritm is start ...")
    population = [generate_individual(param_space) for _ in range(population_size)]
    for generation in range(generations):
        fitness_scores = evaluate_population(population, currency_files)
        population_fitness = list(zip(population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Selection
        selected = population_fitness[:int(len(population) / 2)]
        population = [individual for individual, _ in selected]
        

        # Crossover and mutation
        while len(population) < population_size:
            if len(selected) >= 2:
                # انتخاب دو تاپل تصادفی از لیست selected و سپس استخراج فقط بخش پارامترها
                selected_individuals = random.sample(selected, 2)
                parent1, parent2 = selected_individuals[0][0], selected_individuals[1][0]
            else:
                # ایجاد دو والد جدید به صورت تصادفی اگر لیست selected کمتر از دو عنصر داشته باشد
                parent1 = generate_individual(param_space)
                parent2 = generate_individual(param_space)

            child = crossover(parent1, parent2)
            child = list(child)
            child = mutate(child, mutation_rate, param_space)
            population.append(child)
        try :
            max = 0
            for i in range (len(population_fitness)):
                if  population_fitness[0][-1] > max :
                     best_individual, best_fitness = population_fitness[i]
            if max == 0 :
                save_to_file(f"All fitness are zero ....")
            else :    
                save_to_file(f"Generation {generation}: Best Fitness = {best_fitness}, best individual = {best_individual}")
        except :
            save_to_file(f"population fitness in empty")


param_space = [(2, 8), (2, 20), (30, 500), (500, 6000), (100, 1000), (52, 70), (8, 18)]  # Example param_space including allowed_hours range
genetic_algorithm(currency_files, 10, 3, 0.2, param_space)
