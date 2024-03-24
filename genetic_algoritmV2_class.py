import random
import pandas as pd
from celery import group
from tasks import train_model_task

class GeneticAlgoritmV2 :

        
    def read_data(self, file_name, tail_size=7000):
        try:
            data = pd.read_csv(file_name)
            return data.tail(tail_size)
        except FileNotFoundError:
            print(f"File {file_name} not found.")
            return None



    def generate_individual(self, param_space):
        # ایجاد قسمت‌های اصلی individual با استفاده از مقادیر تصادفی بر اساس param_space
        individual = [random.randint(param[0], param[1]) for param in param_space[:-1]]
        
        # ایجاد لیست allowed_hours به صورت تصادفی
        # توجه داشته باشید که این قسمت آخرین عنصر individual را تشکیل می‌دهد
        allowed_hours = random.sample(range(3, 24), random.randint(param_space[-1][0], param_space[-1][1]))
        
        # افزودن allowed_hours به عنوان آخرین عنصر individual
        individual.append(allowed_hours)
        
        return individual


    def crossover(self, parent1, parent2):
        if len(parent1) < 3 or len(parent2) < 3:
            return random.choice([parent1, parent2])
        crossover_points = sorted(random.sample(range(1, len(parent1)-1), 2))
        child = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[crossover_points[1]:]
        return child




    def mutate(self, individual, mutation_rate, param_space):
        # تبدیل individual به لیست برای اجازه تغییرات
        individual = list(individual)  

        for i in range(len(individual) - 1):
            if random.random() < mutation_rate:
                individual[i] = random.randint(param_space[i][0], param_space[i][1])
        
        if random.random() < mutation_rate:
            # اطمینان از اینکه allowed_hours یک لیست است
            if not isinstance(individual[-1], list):
                print("Warning: 'allowed_hours' is not a list. Initializing to empty list.")
                allowed_hours = []
            else:
                allowed_hours = individual[-1]

            if random.random() < 0.5:
                if len(allowed_hours) < param_space[-1][1]:
                    new_hour = random.choice(list(set(range(3, 24)) - set(allowed_hours)))
                    allowed_hours.append(new_hour)
            else:
                if len(allowed_hours) > param_space[-1][0]:
                    allowed_hours.remove(random.choice(allowed_hours))
            individual[-1] = allowed_hours
        
        return tuple(individual)  # تبدیل دوباره individual به تاپل اگر نیاز به حفظ ساختار تاپل دارید






    def evaluate_population(self, population, currency_file):
        results = []  # نتایج برای هر فرد
        tasks = []
        fitness_scores = []  # اینجا یک لیست برای نگهداری امتیازات ایجاد می‌کنیم

        currency_data = self.read_data(currency_file, 7000)
        if currency_data is not None:

            for individual in population:
                task = train_model_task.s(currency_data.to_json(orient='split'), *individual[:-1], individual[-1])
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


        print(f"Stage finish, scors : {fitness_scores}")
        return fitness_scores  # برگرداندن لیست امتیازات دقت برای هر فرد در جمعیت





    def genetic_algorithm_for_all_currencies(self, currency_file, population_size, generations, mutation_rate, param_space):
        population = [self.generate_individual(param_space) for _ in range(population_size)]
        best_fitness = -1
        best_individual = None

        for generation in range(generations):
            fitness_scores = self.evaluate_population(population, currency_file)
            total_scores = sum(fitness_scores)
            print(f"Currency: {currency_file}, Generation {generation + 1}: Total Fitness = {total_scores}")

            if fitness_scores and max(fitness_scores) > best_fitness:
                best_fitness = max(fitness_scores)
                best_individual = population[fitness_scores.index(best_fitness)]

            print(f"Currency: {currency_file}, Generation {generation + 1}: Best Fitness = {best_fitness}")

            selected = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:population_size // 2]
            population = [ind for ind, _ in selected]

            while len(population) < population_size:
                if len(selected) >= 2:
                    parent1, parent2 = random.sample(selected, 2)
                else:
                    parent1 = self.generate_individual(param_space)
                    parent2 = self.generate_individual(param_space)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate, param_space)
                population.append(child)

        return f"Currency: {currency_file}, Best Individual: {best_individual}, Fitness: {best_fitness}"
