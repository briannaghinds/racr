"""
@author: Brianna Hinds
Description: helper funcitons for the F1 application
"""
from constants import INPUT_COLS, DEFAULT_VALS
import pandas as pd


def data_cleaning(user_choices, pull_default=True):
    cols_missing = [i for i in INPUT_COLS if i not in user_choices]

    for cols in cols_missing:
        user_choices[cols] = DEFAULT_VALS.get(cols, 0)

    # make sure input is in order the model expects
    user_choices = user_choices[INPUT_COLS]

    # SANITY PRINTS
    # print(user_choices)
    # st.dataframe(user_choices)

    return user_choices