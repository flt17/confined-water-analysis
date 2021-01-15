import os
import pytest
import sys
from unittest import mock

sys.path.append("../")
from confined_water import utils


class TestGetPathToFile:
    def test_raises_error_when_path_not_found(self):
        path = "./files/bulk_water/classical"
        suffix = "random_suffix"

        with pytest.raises(utils.UnableToFindFile):
            utils.get_path_to_file(path, suffix)

    def test_returns_all_files_with_same_suffix(self):
        path = "./files/bulk_water/classical"
        suffix = "pdb"

        file_names = utils.get_path_to_file(path, suffix)

        assert file_names == [
            os.path.join(path, "PBE-D3-bnnt-w68-T330K-1bar.pdb"),
            os.path.join(path, "PBE-D3-cnt-w65-T330K-1bar.pdb"),
            os.path.join(path, "revPBE0-D3-w64-T300K-1bar.pdb"),
        ]

    def test_returns_all_files_containing_same_string_in_prefix(self):
        path = "./files/bulk_water/classical"
        suffix = "pdb"
        prefix = "T330K"

        file_names = utils.get_path_to_file(path, suffix, prefix, exact_match=False)

        assert file_names == [
            os.path.join(path, "PBE-D3-bnnt-w68-T330K-1bar.pdb"),
            os.path.join(path, "PBE-D3-cnt-w65-T330K-1bar.pdb"),
        ]

    def test_returns_requested_file(self):
        path = "./files/bulk_water/classical"
        suffix = "pdb"
        prefix = "revPBE0-D3-w64-T300K-1bar"

        file_name = utils.get_path_to_file(path, suffix, prefix)

        assert file_name == os.path.join(path, "revPBE0-D3-w64-T300K-1bar.pdb")


class TestGetAseAtomsObject:
    @mock.patch.object(utils, "get_path_to_file")
    def test_returns_ase_AO_from_pdb(self, get_path_to_file_mock):
        path_to_pdb = "./files/bulk_water/classical/revPBE0-D3-w64-T300K-1bar.pdb"

        ase_AO = utils.get_ase_atoms_object(path_to_pdb)

        get_path_to_file_mock.assert_not_called()
        assert ase_AO.get_global_number_of_atoms() == 192


class TestGetMdanalysisUniverse:
    def test_raises_error_when_unknown_universe_type(self):
        path = "./files/bulk_water/classical"
        universe_type = "banana"

        with pytest.raises(utils.UndefinedOption):
            utils.get_mdanalysis_universe(path, universe_type)

    def test_returns_universe_with_single_trajectory(self):
        path = "./files/bulk_water/classical"
        universe_type = "positions"
        pdb_prefix = "revPBE0-D3-w64-T300K-1bar"

        mdanalysis_universe = utils.get_mdanalysis_universe(path, universe_type, pdb_prefix)

        assert mdanalysis_universe.trajectory.n_frames == 106

    def test_returns_universe_with_multiple_trajectory(self):
        path = "./files/bulk_water/quantum"
        universe_type = "positions"
        pdb_prefix = "revPBE0-D3-w64-T300K-1bar"

        mdanalysis_universes = utils.get_mdanalysis_universe(path, universe_type, pdb_prefix)

        assert len(mdanalysis_universes) == 5
