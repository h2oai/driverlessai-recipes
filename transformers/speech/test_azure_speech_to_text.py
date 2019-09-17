"""An example of integration with """

import datatable as dt
import pandas as pd
from azure_speech_to_text import AzureSpeechToText
import pytest


@pytest.fixture
def in_out_dt() -> dt.Frame:
    input_dt = dt.Frame(A=[
        'data/sample01.wav',
        'data/sample02.wav',
        'data/sample03.wav',
        'data/sample04.wav',
        ], stype=str)

    expected_dt = pd.Series([
        "This is custom recipe.",
        "This is example of integration with azure speech recognition.",
        "This example is provided by age to dot AI.",
        "0.",
        ])

    return (input_dt, expected_dt)


def test_input(in_out_dt):
    t = AzureSpeechToText()
    input_dt, expected_dt = in_out_dt
    output_dt = t.transform(input_dt)
    print(output_dt[1])
    assert output_dt.equals(expected_dt)
