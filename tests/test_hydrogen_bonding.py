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

    def test_returns_pandas_dataframe_for_carbon_nanotube(self):

        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)

        simulation.set_pbc_dimensions("z")

        hydrogen_bonding_analysis = hydrogen_bonding.HydrogenBonding(
            simulation.position_universes[0], simulation.topology
        )

        hydrogen_bonding_analysis.find_acceptor_donor_pairs(
            start_frame=0,
            end_frame=-1,
            frame_frequency=10,
            time_between_frames=20,
            pbc_dimensions="z",
        )

        assert np.max(hydrogen_bonding_analysis.dataframe["Time"]) == 2000


class TestHydrogenBondingHeavyAtomsAnalysis:
    def test_returns_pandas_dataframe_for_carbon_nanotube(self):

        path = "./files/water_in_carbon_nanotube/m12_n12/classical"

        simulation = analysis.Simulation(path)

        simulation.read_in_simulation_data(read_positions=True)

        simulation.set_pbc_dimensions("z")

        hydrogen_bonding_analysis = hydrogen_bonding.HydrogenBonding(
            simulation.position_universes[0], simulation.topology
        )

        hydrogen_bonding_analysis.heavy_atoms_analysis(
            start_frame=0,
            end_frame=-1,
            frame_frequency=10,
            time_between_frames=20,
            pbc_dimensions="z",
            spatial_extent_contact_layer=3.65
        )

        assert len(np.unique(hydrogen_bonding_analysis.heavy_atoms_dataframe["Species"])) == 2

