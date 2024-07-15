import os
import pandas as pd
import numpy as np
import spacy


# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
def remove_invalid_keystrokes(df):
    """
    A helper function that takes as input a dataframe, and returns a new dataframe
    no longer containing rows with the string "<0>".

    Parameters:
    - df: a pandas DataFrame.

    Returns:
    - DataFrame without rows containing the string "<0>".
    """
    # A helper function that takes as input a dataframe, and return a new
    # dataframe no longer containing rows with the string "<0>"
    return df.loc[df["key"] != "<0>"]


def clean_letters(letters):
    """
    Removes single quotation marks from each item in a list of letters/strings.

    Parameters:
    - letters (list[str]): A list of strings or letters, each potentially wrapped with single quotation marks.

    Returns:
    - list[str]: A list of strings or letters without the surrounding single quotation marks.

    Note:
    This function is designed to clean up lists where each item might be wrapped with single quotation marks
    (e.g., ["'a'", "'b'", "'c'"] to ["a", "b", "c"]).

    Example:
    >>> clean_letters(["'a'", "'b'", "'hello'"])
    ['a', 'b', 'hello']

    """
    return [item.strip("'") for item in letters]


class SentenceParser:
    """Parses our dataset into best-effort sentences by trying to account for punctuation.

    We them use the spacy tokenizer to tokenize the best-effort sentences into words for word level features
    """

    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def path(self):
        return self.csv_file_path

    def as_df(self):
        return pd.read_csv(
            os.path.join(os.getcwd(), "cleaned.csv"),
            dtype={
                "key": str,
                "press_time": np.float64,
                "release_time": np.float64,
                "platform_id": np.uint8,
                "session_id": np.uint8,
                "user_ids": np.uint8,
            },
        )

    def letters(self, as_list: bool = False):
        if as_list is True:
            return list(remove_invalid_keystrokes(self.as_df()).iloc[:, 1])
        elif as_list is False:
            return remove_invalid_keystrokes(self.as_df())

    def get_words(self, data_df):
        tokenized_words = []
        sentences = self.make_sentences(data_df)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentences)
        for token in doc:
            tokenized_words.append(token.text)
        return tokenized_words


def reconstruct_text(key_presses):
    text = []
    skip_next = 0
    shift_active = False

    for key in key_presses:
        if skip_next > 0:
            skip_next -= 1
            continue
        if key == "comma":
            text.append(",")
            continue
        if key.startswith("Key."):
            if key == "Key.space":
                text.append(" ")
            elif key == "Key.backspace":
                if text:
                    text.pop()
            elif key == "Key.shift_r":
                # Do nothing for shift keys
                shift_active = True
                continue
            else:
                # Handle other special keys if necessary
                continue
        else:
            if key == ' ""\' ""':
                text.append("'")
                continue
            else:
                # Remove single quotes around the characters
                char = key.strip("'")
                if shift_active:
                    char = char.upper()
                    shift_active = False
                text.append(char)
    return "".join(text)
