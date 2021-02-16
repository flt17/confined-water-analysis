import sys

sys.path.append("../")
from confined_water import analysis
from confined_water import global_variables
from confined_water import free_energy


class TestComputeAtomicProbabilities:
    # def test_returns_probabilities_for_tube(self):
    #     path = "./files/water_in_carbon_nanotube/classical"

    #     simulation = analysis.Simulation(path)
    #     simulation.read_in_simulation_data(read_positions=True, read_summed_forces=True)

    #     simulation.set_pbc_dimensions(pbc_dimensions="z")
    #     pbc_indices = global_variables.DIMENSION_DICTIONARY.get(simulation.pbc_dimensions)

    #     free_energy.compute_atomic_probabilities(
    #         simulation.position_universes,
    #         simulation.topology,
    #         pbc_indices,
    #         start_frame=0,
    #         end_frame=100,
    #         frame_frequency=1,
    #     )

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

        free_energy.compute_atomic_probabilities(
            simulation.position_universes,
            simulation.topology,
            spatial_expansion_contact_layer,
            pbc_indices,
            start_frame=0,
            end_frame=100,
            frame_frequency=1,
        )
