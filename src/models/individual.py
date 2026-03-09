from dataclasses import dataclass

@dataclass
class Individual():
    prompt: str
    fitness: float

def mock_individuals() -> list[Individual]:
    mocks = []
    mocks.append(Individual("Galo", 9))
    mocks.append(Individual("Raposa", 7))
    mocks.append(Individual("Biriba", 5))
    mocks.append(Individual("Cartola", 3))
    mocks.append(Individual("Saci", 6))

    return mocks