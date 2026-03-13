from src.selection.base import Selection
from src.selection.roullete import RoulleteSelection
from src.fitness.cross_entropy import CrossEntropyFitness
from src.models.individual import get_sample_individuals

def main():
    ind_mocks = get_sample_individuals()

    fitness = CrossEntropyFitness()
    
    for ind in ind_mocks:
        score = fitness.evaluate(ind)
        print(score)

if __name__ == "__main__":
    main()
