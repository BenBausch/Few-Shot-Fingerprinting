import pytest
import os

@pytest.fixture(scope='session', autouse=True)
def set_test_directory():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))