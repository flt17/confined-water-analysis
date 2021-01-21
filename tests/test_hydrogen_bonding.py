import numpy as np
import pytest
import sys

sys.path.append("../")
from confined_water import analysis
from confined_water import hydrogen_bonding


class TestHydrogenBondingFindAcceptorDonorPairs:
    def test_returns_pandas_dataframe_for_bulk_water(self):
        path = "./files/bulk_water/classical"

        topology_name = "revPBE0-D3-w64-T300K-1bar"
        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True, topology_file_name=topology_name)

        hydrogen_bonding_analysis = hydrogen_bonding.HydrogenBonding(
            simulation.position_universes[0], simulation.topology
        )

        hydrogen_bonding_analysis.find_acceptor_donor_pairs(
            start_frame=0,
            end_frame=-1,
            frame_frequency=10,
            time_between_frames=20,
            pbc_dimensions="xyz",
        )

        assert np.max(hydrogen_bonding_analysis.dataframe["Time"]) == 2000
