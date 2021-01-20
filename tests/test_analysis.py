import os
import pandas
import pytest
from sklearn.metrics import mean_squared_error
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

    def test_returns_same_result_as_vmd(self):

        path = "./files/bulk_water/classical"

        vmd_data = pandas.read_csv(
            os.path.join(path, "rdf_vmd.dat"), sep=r"\s", engine="python", header=None
        )

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )
        simulation.compute_rdf("O", "O", spatial_range=6.2, spatial_resolution=309)

        assert (
            mean_squared_error(
                vmd_data[1][1::], simulation.radial_distribution_functions["O-O"][1], squared=False
            )
            < 2e-2
        )


class TestSimulationComputeDensityProfile:
    def test_raises_error_when_species_not_found(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        with pytest.raises(analysis.KeyNotFound):
            simulation.compute_density_profile(["O", "H", "C"], direction="z")

    def test_raises_error_when_direction_not_found(self):
        path = "./files/water_on_graphene"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        with pytest.raises(analysis.KeyNotFound):
            simulation.compute_density_profile(["O", "H"], direction="w")


class TestSimulation_ComputeDensityProfileAlongCartesianAxis:
    def test_raises_error_when_no_solid_is_found(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        with pytest.raises(analysis.KeyNotFound):
            simulation.compute_density_profile(["O", "H"], direction="z")

    def test_returns_profile_in_z_direction(self):

        path = "./files/water_on_graphene"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )
        simulation.compute_density_profile(["O", "H"], direction="z")

        assert simulation.density_profiles.get("O H - z")
