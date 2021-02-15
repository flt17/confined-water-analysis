import sys

sys.path.append("../")
from confined_water import analysis
from confined_water import global_variables
from confined_water import free_energy


class TestComputeAtomicProbabilities:
    def test_returns_invariant_probabilities(self):
        path = "./files/water_in_carbon_nanotube/classical"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data(read_positions=True, read_summed_forces=True)

        simulation.set_pbc_dimensions(pbc_dimensions="z")
        pbc_indices = global_variables.DIMENSION_DICTIONARY.get(simulation.pbc_dimensions)

        free_energy.compute_atomic_probabilities(
            simulation.position_universes,
            simulation.topology,
            pbc_indices,
            start_frame=0,
            end_frame=100,
            frame_frequency=1,
        )
