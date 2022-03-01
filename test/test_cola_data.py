from datasets import load_dataset
import pytest


@pytest.mark.functional
def test_value_matched():
    expected = {
        'idx': 0,
        'label': 1,
        'sentence': "Our friends won't buy this analysis, let alone the next one we propose."
    }
    cola_dataset = load_dataset("glue", "cola")
    train_dataset = cola_dataset['train']
    assert train_dataset[0] == expected

