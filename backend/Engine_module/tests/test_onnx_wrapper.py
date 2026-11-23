import sys
from pathlib import Path

import numpy as np
import pytest

ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_DIR))

from src.gis_engine.inference.onnx import ONNXModelWrapper


class _Session:
    def __init__(self, output):
        self.output = output

    def run(self, output_names, inputs):
        return [self.output]


def _wrapper(output):
    wrapper = ONNXModelWrapper.__new__(ONNXModelWrapper)
    wrapper.session = _Session(output)
    wrapper.input_name = "input"
    wrapper.output_name = "output"
    return wrapper


def test_predict_accepts_expected_output_shape_and_finite_values():
    tensor = np.zeros((1, 5, 2, 3), dtype=np.float32)
    logits = np.zeros((1, 4, 2, 3), dtype=np.float32)

    raw_logits, class_map = _wrapper(logits).predict(tensor)

    assert raw_logits.shape == (1, 4, 2, 3)
    assert class_map.shape == (2, 3)


@pytest.mark.parametrize(
    "output, message",
    [
        (np.zeros((4, 2, 3), dtype=np.float32), "4 dimensions"),
        (np.zeros((1, 3, 2, 3), dtype=np.float32), "expected shape"),
        (np.full((1, 4, 2, 3), np.nan, dtype=np.float32), "NaN"),
        (np.full((1, 4, 2, 3), np.inf, dtype=np.float32), "NaN"),
    ],
)
def test_predict_rejects_malformed_model_outputs(output, message):
    tensor = np.zeros((1, 5, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        _wrapper(output).predict(tensor)


def test_predict_rejects_empty_model_outputs():
    class _EmptySession(_Session):
        def run(self, output_names, inputs):
            return []

    wrapper = ONNXModelWrapper.__new__(ONNXModelWrapper)
    wrapper.session = _EmptySession(None)
    wrapper.input_name = "input"
    wrapper.output_name = "output"
    tensor = np.zeros((1, 5, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="no outputs"):
        wrapper.predict(tensor)
