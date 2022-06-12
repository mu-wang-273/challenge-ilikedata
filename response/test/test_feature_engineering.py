"""
Unit test for feature_engineering.py
"""
from src.feature_engineering import fe_customer
import pandas as pd
from pandas.testing import assert_frame_equal

def test_fe_customer():
    input_data = pd.read_parquet("test/fe_test_input.parquet")
    test_output_data = pd.read_parquet("test/fe_test_output.parquet")
    output_data = fe_customer(input_data)
    assert_frame_equal(test_output_data, output_data)
