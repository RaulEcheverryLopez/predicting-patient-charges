import numpy as np
import pandas as pd
import pytest
from src.pipeline import MLPipeline


def test_pipeline_initialization():
    pipeline = MLPipeline()
    assert pipeline.model is None


def test_load_data():
    pipeline = MLPipeline()
    data = pipeline.load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
