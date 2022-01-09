import sys

sys.path.append("../")
from confined_water import analysis
from confined_water import global_variables
from confined_water import free_energy


class TestComputeAtomicProbabilities:
    def test_returns_probabilities_for_tube(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        simulation.set_pbc_dimensions(pbc_dimensions="z")
        pbc_indices = global_variables.DIMENSION_DICTIONARY.get(simulation.pbc_dimensions)

        simulation.compute_density_profile(["O", "H"], direction="radial z")

        spatial_expansion_contact_layer = simulation.get_water_contact_layer_on_interface()

        pbc_dimensions_indices = global_variables.DIMENSION_DICTIONARY.get(
            simulation.pbc_dimensions
        )
        tube_radius = simulation.compute_tube_radius(pbc_dimensions_indices)
        free_energy.compute_spatial_distribution_of_atoms_on_interface(
            simulation.position_universes[0],
            simulation.topology,
            spatial_expansion_contact_layer,
            pbc_indices,
            species="O",
            start_frame=0,
            end_frame=100,
            frame_frequency=1,
            tube_radius=tube_radius,
            tube_length_in_unit_cells=6,
        )

    def test_returns_probabilities_for_tube_in_parallel(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        simulation.set_pbc_dimensions(pbc_dimensions="z")
        pbc_indices = global_variables.DIMENSION_DICTIONARY.get(simulation.pbc_dimensions)

        simulation.compute_density_profile(["O", "H"], direction="radial z")

        spatial_expansion_contact_layer = simulation.get_water_contact_layer_on_interface()

        pbc_dimensions_indices = global_variables.DIMENSION_DICTIONARY.get(
            simulation.pbc_dimensions
        )
        tube_radius = simulation.compute_tube_radius(pbc_dimensions_indices)
        free_energy.compute_spatial_distribution_of_atoms_on_interface(
            simulation.position_universes[0],
            simulation.topology,
            spatial_expansion_contact_layer,
            pbc_indices,
            species="O",
            start_frame=80,
            end_frame=100,
            frame_frequency=1,
            tube_radius=tube_radius,
            tube_length_in_unit_cells=6,
            parallel=True,
            number_of_cores=2
        )
    def test_returns_probabilities_for_tube_in_parallel_serial(self):
        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        simulation.set_pbc_dimensions(pbc_dimensions="z")
        pbc_indices = global_variables.DIMENSION_DICTIONARY.get(simulation.pbc_dimensions)

        simulation.compute_density_profile(["O", "H"], direction="radial z")

        spatial_expansion_contact_layer = simulation.get_water_contact_layer_on_interface()

        pbc_dimensions_indices = global_variables.DIMENSION_DICTIONARY.get(
            simulation.pbc_dimensions
        )
        tube_radius = simulation.compute_tube_radius(pbc_dimensions_indices)
        free_energy.compute_spatial_distribution_of_atoms_on_interface(
            simulation.position_universes[0],
            simulation.topology,
            spatial_expansion_contact_layer,
            pbc_indices,
            species="O",
            start_frame=80,
            end_frame=100,
            frame_frequency=1,
            tube_radius=tube_radius,
            tube_length_in_unit_cells=6,
            parallel=True,
            number_of_cores=1
        )

    def test_returns_probabilities_for_sheet(self):
        path = "./files/water_on_graphene"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data(read_positions=True)

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=20
        )

        simulation.set_pbc_dimensions(pbc_dimensions="xy")
        pbc_indices = global_variables.DIMENSION_DICTIONARY.get(simulation.pbc_dimensions)

        simulation.compute_density_profile(["O", "H"], direction="z")

        spatial_expansion_contact_layer = simulation.get_water_contact_layer_on_interface()

        free_energy.compute_spatial_distribution_of_atoms_on_interface(
            simulation.position_universes[0],
            simulation.topology,
            spatial_expansion_contact_layer,
            pbc_indices,
            species="O",
            start_frame=0,
            end_frame=100,
            frame_frequency=1,
        )
