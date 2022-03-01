from omegaconf import OmegaConf


def test_load_config():
    expected = {'preferences': {'user': 'raviraja', 'trait': 'i_like_my_sleeping'}}
    config = OmegaConf.load("config.yaml")
    assert expected == config

