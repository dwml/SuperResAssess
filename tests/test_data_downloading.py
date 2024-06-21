from superresassess.data import HCP

def test_hcp_setup(tmpdir):
    hcp = HCP(tmpdir, download=True)
    assert hcp.data_dir == tmpdir
    assert hcp.download == True
