import pytest
import sys

sys.path.append("../")
from confined_water import analysis


class TestSimulationReadInSimulationData:
    def test_sets_up_position_universes_for_pimd(self):
        path = "./files/bulk_water/quantum"
        pimd_simulation = analysis.Simulation(path)

        pimd_simulation.read_in_simulation_data(read_positions=True)

        assert len(pimd_simulation.position_universes) == 5

    def test_sets_up_position_universes_with_multiple_pdbs_in_directory(self):
        path = "./files/bulk_water/classical"
        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        assert len(simulation.position_universes) == 1


class TestConfinedWaterSystemAddSimulation:
    def test_returns_added_simulation(self):
        path = "./files/bulk_water/quantum"

        bulk_water = analysis.ConfinedWaterSystem("Bulk Water")
        bulk_water.add_simulation("PIMD", path)

        assert isinstance(bulk_water.simulations.get("PIMD"), analysis.Simulation)


class TestSimulationComputeRDF:
    def test_raises_error_when_species_not_found(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        with pytest.raises(analysis.KeyNotFound):
            simulation.compute_rdf("O", "X")
