from dataclasses import dataclass
from pathlib import Path

MOCKS_FILE = "individual_mocks.txt"
MOCKS_PATH = Path('src/models') / MOCKS_FILE

@dataclass
class Individual():
    prompt: str
    fitness: float

def get_sample_individuals() -> list[Individual]:
    mocks = []

    with open(MOCKS_PATH) as file:
        for line in file:
            content = line.split(';')
            prompt = content[0]
            fitness = int(content[1])

            mocks.append(Individual(prompt, fitness))

    return mocks