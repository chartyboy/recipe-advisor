"""
Implements data cleaning process for scraped recipe text data.

Classes
-------
RecipeProcessor
    Class with packaged methods to load and clean text data.
"""

import jq
import json
import re
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Collection, Iterable, Generator, Optional, Sequence


@dataclass
class RecipeProcessor:
    """
    Interface to handle processing scraped recipe data in JSONLines files.

    Attributes
    ----------
    schema : str
        Schema to process JSON in jq_ syntax.

    .. _jq: https://jqlang.github.io/jq/

    Methods
    -------
    load_jsonlines(self, fpath: str)
        Loads JSONLines files with a jq schema.
    process_recipes(self, path_dict: dict[str, str])
        Main function to clean and output data.
    """

    schema: str

    def __post_init__(self):
        self.jq_filter = jq.compile(self.schema)

    def load_jsonlines(self, fpath: str) -> List[List | Dict]:
        """
        Loads a JSONLines file with a jq schema.

        Parameters
        ----------
        fpath : str
            Path to location of JSONLines file.

        Returns
        -------
        content : List[List | Dict]
            Result of processing with jq schema.

        See Also
        --------
        to_pandas : Convert loaded content into pandas Dataframes
        """

        content = list()
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = self.jq_filter.input(json.loads(line))
                    content.append(data.all())
        return content

    def to_pandas(
        self, content: List[List | Dict], columns: List[str] = list()
    ) -> pd.DataFrame:
        """
        Creates a pandas Dataframe from a list of lists or dicts.

        Parameters
        ----------
        content : List[List | Dict]

        columns : List[str], default = list()
            Column names for the resulting Dataframe.

        Returns
        -------
        df : pd.DataFrame
            pandas Dataframe created from content lists and column names.
        """
        if content:
            if isinstance(content[0], dict):
                df = pd.DataFrame.from_records(content)
            elif isinstance(content[0], list):
                if columns:
                    df = pd.DataFrame(content, columns=columns)
                else:
                    df = pd.DataFrame(content)
            else:
                raise TypeError(f"Content is of invalid type, got {type(content[0])}")
        else:  # empty object
            if isinstance(content, dict) or isinstance(content, list):
                df = pd.DataFrame(content)
            else:  # not a dict or list
                raise TypeError(f"Content is of invalid type, got {type(content[0])}")
        return df

    def df_to_jsonlines(self, out_path: str, df: pd.DataFrame) -> None:
        """
        Writes content of a Dataframe to a JSONLines file.

        Parameters
        ----------
        out_path : str
            Path of file (including the filename) to write to.

        df : pd.Dataframe
            Dataframe to read data from.
        """
        jsonl_out = df.to_json(orient="records", lines=True, force_ascii=False)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(jsonl_out)

    def condense_lists(self, content: List[List[str]], sep: str = ", ") -> List[str]:
        """
        Concatenate multiple lists of strings into individual strings.

        Parameters
        ----------
        content : List[List[str]]
            Array of lists of strings.

        sep : str, default = ", "
            Separator between concatenated strings.

        Returns
        -------
        condensed : List[str]
            Array of concatenated strings.
        """
        condensed = list()
        for text in content:
            try:
                combined = sep.join(text)
                condensed.append(combined)
            except TypeError:
                condensed.append(None)
        return condensed

    def replace_iteratively(
        self,
        base: str,
        iterative_replacer: List[str],
        indicator: str,
        append: bool = True,
    ) -> List[str]:
        """
        Split string along a character sequence, then concatenate with replacements.

        Parameters
        ----------
        base : str
            String where replacements will be inserted.

        iterative_replacer : List[str]
            Iterable of strings to use as replacements.

        indicator : str
            String sequence to be split along in `base`.

        append : bool, default = True
            Append replacements to split strings if True. If False, prepend to
            split strings instead.

        Returns
        -------
        res : List[str]
            Array of split strings with concatenated replacements.
        """

        splitted = base.split(indicator)
        res = list()
        replaces = iterative_replacer.copy()
        for base_part in splitted:
            if append:
                replaced = base_part + replaces.pop(0)
            else:  # prepend instead
                replaced = replaces.pop(0) + base_part
            res.append(replaced)
        return res

    def condense_and_pad(
        self,
        content: List[str],
        pad: List[str],
        sep: str = ", ",
        pad_sep: str = " ",
    ) -> str:
        """
        Concatenate two lists of strings with varying padding and separators.

        Parameters
        ----------
        content : List[str]
            Values to join with strings in `pad`. Strings will be concatenated
            after a string in `pad`,

        pad : List[str]
            Values to join with strings in `content`. Strings will be concatenated
            before a string in `content`,

        sep : str
            String sequence to separate successive strings.

        pad_sep : str
            String sequence to separate strings from `content` and `pad`.

        Returns
        -------
        concat_result : str
            Result of combining two lists of strings.
        """
        if len(pad) != len(content):
            raise RuntimeError("Lists are not of equal length.")
        try:
            concat_result = ""
            for padding, text in zip(pad, content):
                to_concat = padding + pad_sep + text + sep
                concat_result += to_concat

        # Escape on attempting to concat str and None together
        except TypeError:
            return ""

        return concat_result

    def process_recipes(
        self,
        path_dict: dict[str, str],
        columns=["recipe_name", "ingredients", "instructions"],
    ):
        """
        Main loop to load and process recipe data.

        Parameters
        ----------
        path_dict : dict[str, str]
            Map of input paths to output paths.

        Notes
        -------
        This method will read each file in path_dict.keys() and output the results
        to the corresponding file in path_dict.values().
        """

        for inpath, outpath in path_dict.items():
            print(f"Now processing {inpath}...")
            content = self.load_jsonlines(inpath)
            recipe_df = self.to_pandas(content, columns)

            to_replace = r"#@^"

            # Yields strings such as '1.', '2.'
            def num_labeler(
                start: int = 1, stop: int = 1, sep: str = "."
            ) -> Generator[str, Any, Any]:
                for i in range(start, stop + 1):
                    yield str(i) + sep

            def join_strings(strings: List[str], to_replace: str = ""):
                try:
                    return to_replace.join(strings)
                except:
                    return None

            def round_ingredients(string):
                # Match numbers with decimal point and more numbers after
                matches = list(re.finditer(r"\d[\d]*\.\d+", string))
                head = 0
                string_fragments = list()

                if matches:
                    for amount in matches:
                        string_fragments.append(string[head : amount.start()])
                        head = amount.end()

                        ingredient_amount = float(amount.group(0))
                        rounded_amount = str(round(ingredient_amount, 2))
                        string_fragments.append(rounded_amount)
                    string_fragments.append(string[head:])
                    res = "".join(string_fragments)
                else:
                    res = string
                return res

            recipe_df["step_instructions"] = recipe_df["instructions"].apply(
                join_strings, to_replace=to_replace
            )
            # Drop rows with empty values
            recipe_df = recipe_df.replace(to_replace={"": None})
            original_size = recipe_df.shape[0]
            recipe_df = recipe_df.dropna()
            num_dropped = original_size - recipe_df.shape[0]
            print(f"Rows dropped: {num_dropped}")

            recipe_df["step_instructions"] = recipe_df["step_instructions"].apply(
                self.replace_iteratively,
                args=(
                    [
                        x
                        for x in num_labeler(
                            start=1, stop=len(recipe_df["instructions"]), sep=". "
                        )
                    ],
                    to_replace,
                ),
                append=False,
            )
            recipe_df[["ingredients", "instructions"]] = recipe_df[
                ["ingredients", "instructions"]
            ].map(
                lambda x: "\n".join(x)
            )  # type: ignore
            recipe_df["step_instructions"] = recipe_df["step_instructions"].apply(
                lambda x: "\n".join(x)
            )

            # Replace decimal codepoints with unicode characters
            recipe_df = recipe_df.apply(lambda x: x.str.replace("&#39;", "'"))
            recipe_df = recipe_df.apply(lambda x: x.str.replace("&#34;", '"'))
            recipe_df["ingredients"] = recipe_df["ingredients"].apply(round_ingredients)
            # Pad strings so that each entry has a header
            pad = ["Recipe Name", "\nIngredients", "\nCooking Instructions"]
            recipe_df["whole_recipe"] = recipe_df[
                ["recipe_name", "ingredients", "step_instructions"]
            ].apply(self.condense_and_pad, axis=1, pad=pad, pad_sep=": ")

            self.df_to_jsonlines(outpath, recipe_df)
            print(f"Finished processing {inpath}, exported to {outpath}.")


# if __name__ == "__main__":
#     schema = ".recipe_name, .ingredients, [.instructions[].text]"
#     columns = ["recipe_name", "ingredients", "instructions"]
#     fpath = [
#         "./datasets/raw/epicurious.jl",
#         "./datasets/raw/foodnetwork.jl",
#         "./datasets/raw/allrecipes.jl",
#         "./datasets/raw/tasty.jl",
#     ]
#     outpath = [
#         "./datasets/interim/epicurious_cleaned.jsonl",
#         "./datasets/interim/foodnetwork_cleaned.jsonl",
#         "./datasets/interim/allrecipes_cleaned.jsonl",
#         "./datasets/interim/tasty_cleaned.jsonl",
#     ]

#     rp = RecipeProcessor(schema)
#     rp.process_recipes(dict(zip(fpath, outpath)))
