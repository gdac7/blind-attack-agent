from src.selection.base import Selection
from src.selection.roullete import RoulleteSelection
from src.models.individual import mock_individuals

def main():
    selection = RoulleteSelection()
    selection.select(mock_individuals(), 5, 5)

if __name__ == "__main__":
    main()
