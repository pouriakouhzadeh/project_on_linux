from GeneticAlgoritmOptimization import GeneticAlgorithm
# Example of usage

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    result = ga.main("EURUSD60", 60, 20)
    print(f"Resurlt = {result}")