import csv


def extract_mutation_positions(mutation_file: str) -> set[int]:
    """
    Extracts all mutation positions from the second column of a tab-separated file.
    """
    positions = set()
    try:
        with open(mutation_file, "r", newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    try:
                        positions.add(int(row[1]))
                    except ValueError:
                        continue
    except FileNotFoundError:
        raise FileNotFoundError(f"Mutation file '{mutation_file}' not found.")
    return positions


def label_mutation_overlap(muts_of_interest, df, start_col="Start", end_col="End"):
    """
    Adds a 'label' column to the DataFrame indicating whether any mutation position
    from muts_of_interest falls within the [Start, End] range of each row.

    Parameters:
    - muts_of_interest: list of integers (mutation positions)
    - df: pandas DataFrame with at least `start_col` and `end_col`
    - start_col: name of the start column in df
    - end_col: name of the end column in df

    Returns:
    - DataFrame with a new column 'label' (1 if any mutation is in range, else 0)
    """
    muts_set = set(muts_of_interest)

    def is_mut_in_window(start, end):
        return any(pos in muts_set for pos in range(start, end + 1))

    df = df.copy()
    df["Label"] = df.apply(
        lambda row: int(is_mut_in_window(row[start_col], row[end_col])), axis=1
    )
    return df


def label_mutation_overlap_dict(
    muts_of_interest, record, start_key="Start", end_key="End"
):
    """
    Adds a 'Label' key to the record dictionary indicating whether any mutation position
    from muts_of_interest falls within the [Start, End] range.

    Parameters:
    - muts_of_interest: list of integers (mutation positions)
    - record: dict with at least start_key and end_key
    - start_key: name of the key for start position
    - end_key: name of the key for end position

    Returns:
    - The same record dict with a new key 'Label' (1 if any mutation is in range, else 0)
    """
    muts_set = set(muts_of_interest)
    start = record[start_key]
    end = record[end_key]

    label = int(any(pos in muts_set for pos in range(start, end + 1)))

    return label
