import pytest
from train import create_model, train_model, compile_model
from tensorflow.keras.models import Sequential
import pandas as pd


@pytest.fixture
def sample_data():
    data = pd.DataFrame(
        {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1, 2, 3, 4, 5],
            'fetal_health': [0, 0, 1, 2, 2]
        }
    )
    return data


def test_get_data(sample_data):
    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health']
    assert not X.empty
    assert not y.empty


def test_model_type():
    model = create_model(4)
    assert isinstance(model, Sequential)


def test_model_layers():
    model = create_model(4)
    assert len(model.layers) > 0


def test_train_model(sample_data):
    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health']
    model = create_model(2)
    compile_model(model)
    train_model(model, X, y)
    assert model.history.history['loss'][-1] > 0.0
