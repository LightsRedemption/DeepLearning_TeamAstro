import numpy as np
import pytest

from src.preprocessing import N_FEATURES, preprocess_many, preprocess_one


def valid_pair():
    return np.arange(1, 56, dtype=float), np.arange(2, 57, dtype=float)


def test_valid_pair_has_108_features():
    bp, rp = valid_pair()
    result = preprocess_one(bp, rp)
    assert result.shape == (N_FEATURES,)
    assert np.isfinite(result).all()
    assert np.allclose(result[:3], [2.0, 3.0, 4.0])
    assert np.allclose(result[54:57], [1.5, 2.0, 2.5])


@pytest.mark.parametrize("band,size", [("bp", 54), ("bp", 56), ("rp", 54), ("rp", 56)])
def test_wrong_length_rejected(band, size):
    bp, rp = valid_pair()
    if band == "bp":
        bp = np.ones(size)
    else:
        rp = np.ones(size)
    with pytest.raises(ValueError, match="55 coefficients"):
        preprocess_one(bp, rp)


@pytest.mark.parametrize("band", ["bp", "rp"])
def test_zero_denominator_rejected(band):
    bp, rp = valid_pair()
    if band == "bp":
        bp[0] = 0
    else:
        rp[0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        preprocess_one(bp, rp)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(bad_value):
    bp, rp = valid_pair()
    bp[10] = bad_value
    with pytest.raises(ValueError, match="finite"):
        preprocess_one(bp, rp)


def test_many_returns_alignment_mask():
    bp1, rp1 = valid_pair()
    bp2, rp2 = valid_pair()
    bp2[0] = 0
    result = preprocess_many([bp1, bp2], [rp1, rp2])
    assert result.features.shape == (1, 108)
    assert result.valid_mask.tolist() == [True, False]
    assert result.rejection_reasons[0] == ""
    assert "nonzero" in result.rejection_reasons[1]
