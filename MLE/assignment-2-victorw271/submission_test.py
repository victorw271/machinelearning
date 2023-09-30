import os, pathlib
import pytest

def test_submission():
    try:
        s = os.system(r"jupyter nbconvert --to notebook --execute Assignment_2_2023.ipynb")
        assert s == 0
    except:
        pytest.fail("Error while running notebook.")
