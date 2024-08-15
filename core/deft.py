import statistics
from core.utils import create_kit_data_from_df


def capitalize_string(s):
    if len(s) == 1:
        return s.upper()
    elif len(s) > 1:
        return s[0].upper() + s[1:]


def deft(key1, key2):
    key1 = key1.replace("'", "")
    key2 = key2.replace("'", "")
    weird_cases = {
        "appreciated",
        "had",
        "realized",
        "committed",
        "committing",
        "commitment",
        "could",
        "assault",
        "saying",
        "can",
        "with",
        "one-off",
        "methods",
        "teacher",
        "to",
        "against",
        "sport",
        "and",
        "that",
        "did",
        "realizes",
        "remain",
        "an",
        "storyline",
        "was",
        '""its ""',
        "behavior",
    }
    unsupported_key = {
        "Key.esc",
        "Key.down",
        "Key.up",
        "Key.left",
        "Key.right",
        "Ç",
        "ç",
        "Key.media_volume_up",
        "Key.media_volume_down",
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "_",
        "+",
        "~",
        "{",
        "}",
        "|" ":",
        '"',
        "<",
        ">",
        "?",
        "Key.enter",
        "`",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
        "-",
        "=",
        "Key.backspace",
        ".",
        "Key.shift_r",
        "Key.space",
        '"',
    }
    if key1 in unsupported_key:
        # print(f"Key 1 unsupported: {key1.upper()} ")
        return -1
    elif key2 in unsupported_key:
        # print(f"Key 2 unsupported: {key1.upper()} ")
        return -1
    # Check for weird cases
    if key1 in weird_cases:
        print(f"Key 1 weird {key1.upper()} case")
        return -1
    elif key2 in weird_cases:
        print(f"Key 2 weird {key2.upper()} case")
        return -1

    if key1 == "To  " or key2 == "To  " or key1 == "to  " or key2 == "to  ":
        print("weird To   case")
        return -1
    simple_keyboard_layout = [
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["Z", "X", "C", "V", "B", "N", "M"],
    ]

    keyboard_layout = [
        [
            "`",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
            "-",
            "=",
            "Key.backspace",
        ],
        ["Key.tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"],
        [
            "Key.caps_lock",
            "A",
            "S",
            "D",
            "F",
            "G",
            "H",
            "J",
            "K",
            "L",
            ";",
            "'",
            "Key.enter",
        ],
        [
            "Key.shift",
            "Z",
            "X",
            "C",
            "V",
            "B",
            "N",
            "M",
            "Comma",
            ".",
            "/",
            "Key.shift_r",
        ],
        [
            "Key.fn",
            "Key.ctrl",
            "Key.alt",
            "Key.cmd",
            "Key.space",
            "Key.cmd_r",
            "Key.alt_r",
        ],
    ]

    shifted_keys = {
        "1": "!",
        "2": "@",
        "3": "#",
        "4": "$",
        "5": "%",
        "6": "^",
        "7": "&",
        "8": "*",
        "9": "(",
        "0": ")",
        "-": "_",
        "=": "+",
        "`": "~",
        "[": "{",
        "]": "}",
        "\\": "|",
        ";": ":",
        "'": '"',
        ",": "<",
        ".": ">",
        "/": "?",
    }

    key_coordinates = {}
    for row_idx, row in enumerate(simple_keyboard_layout):
        for col_idx, key in enumerate(row):
            key_coordinates[key] = (col_idx, row_idx)
            if key in shifted_keys:
                continue
                # key_coordinates[shifted_keys[key]] = (col_idx, row_idx)
    if key1 == ' ""':
        key1 = '"'
    elif key2 == ' ""':
        key2 = '"'

    if key1 == ' "':
        key1 = '"'
    elif key2 == ' "':
        key2 = '"'

    if key1 == ' " "':
        key1 = '"'
    elif key2 == ' " "':
        key2 = '"'

    if key1 == "'''":
        key1 = "'"
    elif key2 == "'''":
        key2 = "'"

    if key1 == ' "" ""':
        # print("Weird quote case")
        key1 = '"'
    elif key2 == ' "" ""':
        # print("Weird quote case")
        key2 = '"'
    if key1 == '"" ""':
        print("Other weird quote case")
        key1 = '"'
    elif key2 == '"" ""':
        print("Other weird quote case")
        key2 = '"'
    # for row_idx, row in enumerate(keyboard_layout):
    #     for col_idx, key in enumerate(row):
    #         print(
    #             capitalize_string(key).replace("'", ""),
    #             capitalize_string(key2).replace("'", ""),
    #             capitalize_string(key2).replace("'", "") == key,
    #         )
    if not key1 == "'" and not key2 == "'":
        key1_coords = key_coordinates.get(capitalize_string(key1).replace("'", ""))
        key2_coords = key_coordinates.get(capitalize_string(key2).replace("'", ""))
    else:
        # Special case for single quote (apostrophe)
        key1_coords = key_coordinates.get(capitalize_string(key1))
        key2_coords = key_coordinates.get(capitalize_string(key2))

    if key1_coords is None:
        return -1
        raise ValueError(f"Key 1: '{(key1)}' is not found in the keyboard layout.")
    if key2_coords is None:
        # print(f"Unknown string {key2}")
        return -1
        raise ValueError(
            f"Key 2: '{capitalize_string(key2)}, {len(key2)}' is not found in the keyboard layout."
        )

    return max(
        abs((key2_coords[0] - key1_coords[0])), abs((key1_coords[1] - key2_coords[1]))
    )


def flatten_list(nested_list):
    """
    Flattens a nested list into a single-level list.

    Args:
    nested_list (list): A list that may contain nested lists.

    Returns:
    list: A single-level list with all the elements from the nested lists.
    """

    return [
        item
        for sublist in nested_list
        for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])
    ]


def find_avg_deft_for_deft_distance_and_kit_feature(df, deft_distance, feature_type):
    kit_data = create_kit_data_from_df(df, feature_type)
    kit_keys = list(kit_data.keys())
    matches = []
    for kit_key in kit_keys:
        key1, key2 = kit_key.split("|*")
        if deft(key1, key2) == deft_distance:
            matches.append(kit_data[kit_key])
    # print("RAW:", matches)
    # print()
    flat_matches = flatten_list(matches)
    # print("FLAT:", flat_matches)
    if len(matches) == 0:
        print("****** NO MATCHES FOUND - investigate further **********")
        print(df)
        return 0
    return statistics.mean(flat_matches)
