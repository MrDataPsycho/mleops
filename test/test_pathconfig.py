from src.pathconfig import PathConfig
import os


def test_init():
    path_repo = PathConfig()
    path_dict = path_repo.to_dict()
    for _, path in path_dict.items():
        print(path)
        assert os.path.exists(path)