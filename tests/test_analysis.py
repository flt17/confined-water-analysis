import pytest
import sys

sys.path.append("../")
from confined_water import analysis


class TestReadInSimulationData:
    def test_sets_up_position_universes_for_PIMD(self):
        path = "./files/bulk_water/quantum"
        PIMD_simulation = analysis.Simulation(path)

        PIMD_simulation.read_in_simulation_data(read_positions=True)

        assert len(PIMD_simulation.position_universes) == 5
