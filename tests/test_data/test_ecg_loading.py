"""
Verify lead order consistency between MIT-BIH and PTB-XL datasets.

Tests:
1. MIT-BIH lead order via scipy.io.loadmat.
2. PTB-XL lead order via wfdb.rdsamp.
3. PTB-XL leads match the standard 12-lead sequence.
4. Synthetic ECG consistency between .mat and WFDB formats.
"""

import pytest
import scipy.io
import wfdb
import numpy as np
from pathlib import Path


@pytest.fixture
def mitbih_mat_path():
    """Path to MIT-BIH Arrhythmia .mat file."""
    return Path("tests/test_data/data/JS01053.mat").absolute()


@pytest.fixture
def ptbxl_wfdb_path():
    """Path to PTB-XL WFDB record."""
    return Path("tests/test_data/data/07004_hr").absolute()


def test_mitbih_lead_order_via_scipy(mitbih_mat_path):
    """Verify scipy correctly loads 12-lead data from MIT-BIH."""
    assert mitbih_mat_path.exists(), f"File missing: {mitbih_mat_path}"

    mat_data = scipy.io.loadmat(mitbih_mat_path)
    variables = [k for k in mat_data if not k.startswith("__")]

    assert len(variables) == 1, f"Unexpected variables: {variables}"

    leads = mat_data[variables[0]]
    assert leads.shape[0] == 12, f"Expected 12 leads, got {leads.shape[0]}"


def test_ptbxl_lead_order_via_wfdb(ptbxl_wfdb_path):
    """Verify WFDB reads PTB-XL with correct lead names and order."""
    header = ptbxl_wfdb_path.with_suffix(".hea")
    assert header.exists(), f"Missing header: {header}"

    _, meta = wfdb.rdsamp(str(ptbxl_wfdb_path))
    lead_names = meta.get("sig_name", [])

    assert lead_names[:3] == ["I", "II", "III"], "Incorrect limb leads"
    assert len(lead_names) == 12, "Expected 12 leads"


def test_ptbxl_leads_match_standard_order(ptbxl_wfdb_path):
    """Confirm PTB-XL leads follow the standard 12-lead sequence."""
    _, meta = wfdb.rdsamp(str(ptbxl_wfdb_path))
    lead_names = [lead.upper() for lead in meta["sig_name"]]

    STANDARD_LEADS = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    assert [lead.upper() for lead in lead_names] == [
        lead.upper() for lead in STANDARD_LEADS
    ], f"Unexpected lead order: {lead_names}"


# Synthetic ECG test helpers
NUM_LEADS = 12
SAMPLING_RATE = 500  # Hz


@pytest.fixture
def synthetic_ecg():
    """Generate a 12-lead synthetic ECG signal."""
    np.random.seed(42)
    time = np.linspace(0, 1, SAMPLING_RATE)
    return np.array([np.sin(2 * np.pi * (i + 1) * time) for i in range(NUM_LEADS)])


@pytest.fixture
def synthetic_mat_path(tmp_path, synthetic_ecg):
    """Create a temporary .mat file with synthetic ECG data."""
    path = tmp_path / "synthetic.mat"
    scipy.io.savemat(path, {"ecg_data": synthetic_ecg})
    return path


@pytest.fixture
def synthetic_wfdb_path(tmp_path, synthetic_ecg):
    """Create a temporary WFDB record with synthetic data."""
    path = tmp_path / "synthetic"
    wfdb.wrsamp(
        path.name,
        write_dir=path.parent,
        fs=SAMPLING_RATE,
        units=["mV"] * NUM_LEADS,
        sig_name=[
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        p_signal=synthetic_ecg.T,
    )
    return path


def test_synthetic_data_consistency(synthetic_mat_path, synthetic_wfdb_path):
    """Ensure synthetic ECG matches when saved as .mat vs WFDB."""
    # Load .mat format
    mat_leads = scipy.io.loadmat(synthetic_mat_path)["ecg_data"]

    # Load WFDB format and transpose to (leads, samples)
    wfdb_leads = wfdb.rdsamp(str(synthetic_wfdb_path))[0].T

    assert (
        mat_leads.shape == wfdb_leads.shape
    ), f"Shape mismatch: .mat {mat_leads.shape}, WFDB {wfdb_leads.shape}"

    assert np.allclose(
        mat_leads, wfdb_leads, atol=1e-6
    ), "Data mismatch between .mat and WFDB"
