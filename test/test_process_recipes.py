import pytest
import json
import pandas as pd
from src.data import process_recipes


@pytest.fixture(scope="session")
def jsonl_length():
    return 4


@pytest.fixture(scope="session")
def jsonl_file(tmp_path_factory, jsonl_length):
    content = {"test": ["a", "b", "c"]}
    fn = tmp_path_factory.mktemp("dataset") / "test.jsonl"
    with open(fn, "w", encoding="utf-8") as f:
        for i in range(jsonl_length):
            f.write(json.dumps(content) + "\n")

    return fn


@pytest.fixture()
def nested_list_of_strings():
    return [["abc", "b", "c"], ["e", "f", "g"]]


@pytest.fixture(
    scope="module",
    params=[
        ([["abc", "b", "c"], ["e", "f", "g"]], ["test", "test2", "test3"]),
        ([{"test": ["a", "b", "c"]}], ["test"]),
        ([], []),
    ],
)
def data_formats(request):
    return request.param[0], request.param[1]


class TestRecipeProcessor:
    test_rp = process_recipes.RecipeProcessor(".test")

    def test_load_jsonlines(self, jsonl_file, jsonl_length):
        res = self.test_rp.load_jsonlines(jsonl_file)
        print(res)

        # Should read all lines in file
        assert len(res) == jsonl_length

        # Should return list
        assert isinstance(res, list)

    def test_to_pandas(self, data_formats):
        content, columns = data_formats
        df = self.test_rp.to_pandas(content, columns)
        # Should convert either an array of lists or dicts into a pandas Dataframe
        # Input should be able to vary between array of list or dicts
        assert list(df.columns) == columns
        assert df.shape[0] == len(content)

    def test_to_pandas_empty_column_name_list(self):
        content = [["abc", "b", "c"], ["e", "f", "g"]]
        columns = []

        df = self.test_rp.to_pandas(content, columns)  # type: ignore

        assert df.shape[0] == len(content)

    @pytest.mark.parametrize("content", [set(), set("test")])
    def test_to_pandas_raises_on_invalid_type(self, content):
        with pytest.raises(TypeError):
            df = self.test_rp.to_pandas(content=content)  # type: ignore

    def test_df_to_jsonlines(self, tmp_path_factory, data_formats):
        # Should take path to output file and write contents of pandas Dataframe
        # in JSONLines format
        content, columns = data_formats
        df = pd.DataFrame(content, columns=columns)
        fn = tmp_path_factory.mktemp("df_out") / "test.jsonl"

        # Write to file
        self.test_rp.df_to_jsonlines(fn, df)

        # Load freshly written data for assertions
        loaded_df = pd.read_json(fn, lines=True)
        assert df.equals(loaded_df)

    def test_condense_lists(self, nested_list_of_strings):
        sep = "#a8"
        condensed = self.test_rp.condense_lists(nested_list_of_strings, sep=sep)

        # Should take a list of list of strings and return a list of strings
        # Each nested list should be concatenated into a single string, separated by input `sep`

        # Check final length
        assert len(nested_list_of_strings) == len(condensed)

        # Check separators
        first_list = nested_list_of_strings[0]
        first_condensed = condensed[0]
        assert len(first_list) == len(first_condensed.split(sep))

    def test_condense_lists_excepts_TypeError_to_None(self):
        condensed = self.test_rp.condense_lists(
            [["test", None], None], sep="test"  # type:ignore
        )
        assert condensed[0] is None
        assert condensed[1] is None

    def test_replace_iteratively(self, nested_list_of_strings):
        sep = "<SEP>"
        expected_length = len(nested_list_of_strings[0])
        single_string = self.test_rp.condense_lists(nested_list_of_strings, sep=sep)[0]

        repl_str = "<REPLACED>"
        replacers = [repl_str] * expected_length
        # Should take a base string, a list of strings, replacement pattern, and a bool
        # Base string should be appropriately split
        # Strings from list should be inserted at split locations depending on bool
        replaced_append = self.test_rp.replace_iteratively(
            single_string, replacers, sep, append=True
        )

        # Check number of replacements
        assert expected_length == len(replaced_append)
        # Check append
        for repl in replaced_append:
            assert repl.endswith(repl_str)

        # Check prepend
        replaced_prepend = self.test_rp.replace_iteratively(
            single_string, replacers, sep, append=False
        )
        for repl in replaced_prepend:
            assert repl.startswith(repl_str)

    def test_condense_and_pad(self, nested_list_of_strings):
        sep = "<SEP>"
        pad_sep: str = "<PAD_SEP>"
        expected_length = len(nested_list_of_strings[0]) + 1
        result = self.test_rp.condense_and_pad(
            nested_list_of_strings[0],
            nested_list_of_strings[1],
            sep=sep,
            pad_sep=pad_sep,
        )
        # Should take two lists of strings and two padding strings and return a single string
        # Lists should be of same length, raises error if lists are not of equal size
        # Padding should be applied to all concatenation operations
        assert len(result.split(pad_sep)) == expected_length
        assert len(result.split(sep)) == expected_length

        # Check for handling of string concatenation with None
        expected_result = ""
        none_result = self.test_rp.condense_and_pad(
            nested_list_of_strings[0],
            [None] * len(nested_list_of_strings[0]),  # type: ignore
            sep=sep,
            pad_sep=pad_sep,
        )
        assert none_result == expected_result

        # Check for error raised on length mismatch
        with pytest.raises(RuntimeError):
            result = self.test_rp.condense_and_pad(
                nested_list_of_strings[0],
                nested_list_of_strings[1][:2],
                sep=sep,
                pad_sep=pad_sep,
            )
