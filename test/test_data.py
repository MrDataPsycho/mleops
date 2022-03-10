import pytest
from src.factory.data import DataModule


@pytest.mark.functional
def test_value_matched():
    expected = {
        'idx': 0,
        'label': 1,
        'sentence': "Our friends won't buy this analysis, let alone the next one we propose."
    }
    data_model = DataModule()
    data_model.prepare_data()
    # cola_dataset = load_dataset("glue", "cola")
    train_sample = data_model.train_data[0]
    assert train_sample == expected

