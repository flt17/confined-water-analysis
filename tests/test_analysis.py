import numpy as np
import os
import pandas
import pytest
from sklearn.metrics import mean_squared_error
import sys


sys.path.append("../")
from confined_water import analysis


class TestSimulationReadInSimulationData:
    def test_sets_up_connectivity_correctly(self):
        path = "./files/water_in_carbon_nanotube/m30_n30"
        simulation = analysis.Simulation(path)

        simulation.set_pbc_dimensions(pbc_dimensions="z")
        simulation.read_in_simulation_data(read_positions=True)
        assert len(simulation.position_universes[0].residues) == 1094

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

    def test_sets_up_summed_forces_for_quantum_simulation(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/quantum"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=False, read_summed_forces=True)

        assert np.all(simulation.summed_forces)

    def test_sets_up_summed_forces_for_classical_simulation(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=False, read_summed_forces=True)

        assert np.all(simulation.summed_forces)

    def test_sets_up_velocity_universe_for_classical_simulation(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(
            read_positions=False, read_velocities=True, read_summed_forces=False
        )
        assert len(simulation.velocity_universes) == 1


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


class TestSimulation_ComputeWaterOrientationProfileAlongCartesianAxis:
    def test_returns_profile_in_z_direction(self):
        path = "./files/water_on_graphene"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )
        simulation.set_pbc_dimensions("xy")

        simulation.compute_water_orientation_profile(frame_frequency=10)

        assert (
            np.max(simulation.water_orientations[0]) <= 1
            and np.min(simulation.water_orientations[0]) >= -1
        )


class TestSimulation_ComputeWaterOrientationInRadialDirection:
    def test_returns_profile_in_radial_direction(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )
        simulation.set_pbc_dimensions("z")

        simulation.compute_water_orientation_profile(frame_frequency=10)

        assert (
            np.max(simulation.water_orientations[0]) <= 1
            and np.min(simulation.water_orientations[0]) >= -1
        )


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


class TestSimulation_ComputeDensityProfileInRadialDirection:
    def test_returns_profile_radial_direction_parallel_to_z_axis(self):

        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )
        simulation.compute_density_profile(["O", "H"], direction="radial z")

        assert simulation.density_profiles.get("O H - radial z")


class TestSimulation_SetUpHydrogenBondingAnalysis:
    def test_returns_hydrogen_bonding_objects_for_PIMD(self):
        path = "./files/bulk_water/quantum"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data(read_positions=True)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=10, time_between_frames=20
        )

        simulation.set_up_hydrogen_bonding_analysis()

        assert len(simulation.hydrogen_bonding) == 4


class TestSimulation_ComputeMeanSquaredDisplacement:
    def test_raises_error_when_species_not_found(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        with pytest.raises(analysis.KeyNotFound):
            simulation.compute_mean_squared_displacement(["O", "X"])

    def test_raises_error_when_time_between_frames_not_set(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        with pytest.raises(analysis.VariableNotSet):
            simulation.compute_mean_squared_displacement(["O", "H"])

    def test_raises_error_when_correlation_time_is_too_high(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        with pytest.raises(analysis.UnphysicalValue):
            simulation.compute_mean_squared_displacement(["O", "H"], correlation_time=100000)

    def test_raises_error_when_number_of_blocks_is_too_high(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        with pytest.raises(analysis.UnphysicalValue):
            simulation.compute_mean_squared_displacement(
                ["O", "H"], correlation_time=1000, number_of_blocks=20
            )

    def test_returns_MSD_for_bulk_water_only_oxygen(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )
        simulation.compute_mean_squared_displacement(
            ["O"], correlation_time=200, number_of_blocks=5
        )

        assert simulation.mean_squared_displacements.get("O - ct: 200")

    def test_returns_MSD_for_bulk_water_based_on_water_COM(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )
        simulation.compute_mean_squared_displacement(
            ["O", "H"], correlation_time=200, number_of_blocks=5
        )

        assert simulation.mean_squared_displacements.get("O H - ct: 200")


class TestSimulation_ComputeDiffusionCoefficientViaGreenKubo:
    def test_returns_diffusion_coefficient(self):

        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data(
            read_positions=True, read_velocities=True, read_summed_forces=False
        )

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        simulation.compute_diffusion_coefficient_via_green_kubo(
            species=["O"],
            correlation_time=1000,
            number_of_blocks=3,
            start_time=1000,
            end_time=20000,
        )
        assert len(simulation.diffusion_coefficients_via_GK["O - ct: 1000"]) == 3

    def test_raises_error_when_correlation_time_is_too_high(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/quantum"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data(read_positions=False, read_summed_forces=True)

        with pytest.raises(analysis.UnphysicalValue):
            simulation.compute_friction_coefficient_via_green_kubo(
                time_between_frames=1,
                correlation_time=100000,
                number_of_blocks=3000,
                start_time=1000,
                end_time=4500,
                frame_frequency=1,
            )

    def test_raises_error_when_number_of_blocks_is_too_high(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/quantum"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data(read_positions=False, read_summed_forces=True)

        with pytest.raises(analysis.UnphysicalValue):
            simulation.compute_friction_coefficient_via_green_kubo(
                time_between_frames=1,
                correlation_time=1000,
                number_of_blocks=3000,
                start_time=1000,
                end_time=4500,
                frame_frequency=1,
            )

    def test_returns_correct_autocorrelation_function(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data(read_positions=True, read_summed_forces=True)

        simulation.set_pbc_dimensions(pbc_dimensions="z")
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        simulation.compute_friction_coefficient_via_green_kubo(
            time_between_frames=1,
            correlation_time=1000,
            number_of_blocks=3,
            start_time=1000,
            end_time=4500,
            frame_frequency=1,
        )

        assert simulation.friction_coefficients.get("ct: 1000")


class TestSimulation_GetWaterContactLayerOnInterface:
    def test_returns_error_for_three_periodic_directions(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        with pytest.raises(analysis.UnphysicalValue):
            simulation.get_water_contact_layer_on_interface()

    def test_returns_contact_layer_for_two_dimensions(self):
        path = "./files/water_on_graphene"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)

        simulation.set_pbc_dimensions("xy")

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        simulation.compute_density_profile(["O", "H"], direction="z")

        spatial_expansion_contact_layer = simulation.get_water_contact_layer_on_interface()

        assert spatial_expansion_contact_layer > 0

    def test_returns_contact_layer_for_one_dimensions(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)

        simulation.set_pbc_dimensions("z")

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        simulation.compute_density_profile(["O", "H"], direction="radial z")

        spatial_expansion_contact_layer = simulation.get_water_contact_layer_on_interface()

        assert spatial_expansion_contact_layer > 0
