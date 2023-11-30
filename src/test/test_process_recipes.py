import pytest
from src import process_recipes


class TestRecipeProcessor:
    def test_load_jsonlines(self):
        # Should take a path to JSONLines files and parse into a Python object
        # Should parse according to jq syntax
        pass

    def test_to_pandas(self):
        # Should convert either an array of lists or dicts into a pandas Dataframe
        # Input should be able to vary between array of list or dicts
        pass

    def test_df_to_jsonlines(self):
        # Should take path to output file and write contents of pandas Dataframe
        # in JSONLines format
        pass

    def test_condense_lists(self):
        # Should take a list of list of strings and return a list of strings
        # Each nested list should be concatenated into a single string, separated by input `sep`
        # Error in concatenating should return empty string and warning
        pass

    def test_replace_iteratively(self):
        # Should take a base string, a list of strings, replacement pattern, and a bool
        # Base string should be appropriately split
        # Strings from list should be inserted at split locations depending on bool
        pass

    def test_condense_and_pad(self):
        # Should take two lists of strings and two padding strings and return a single string
        # Lists should be of same length, raises error if lists are not of equal size
        # Padding should be applied to all concatenation operations
        pass
