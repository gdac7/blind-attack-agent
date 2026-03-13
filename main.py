from src.selection.base import Selection
from src.selection.roullete import RoulleteSelection
from src.models.individual import get_sample_individuals

def main():
    ind_mocks = get_sample_individuals()

    selection = RoulleteSelection()
    selected = selection.select(ind_mocks, len(ind_mocks), 5)
    print(selected)

if __name__ == "__main__":
    main()
