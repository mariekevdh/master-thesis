import pandas as pd
import argparse
import os


def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-df",
        "--data_folder",
        type=str,
        help="Folder that contains the Webis-WikiDiscussions-18 data files.",
        default="data/",
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Name of output file, without extension.",
        default="grouped_data",
    )
    parser.add_argument(
        "-ts",
        "--test_size",
        type=int,
        help="Number of samples to save as separate test set.",
        default=1000,
    )
    args = parser.parse_args()
    return args


def group_comments(data_path, return_stats=True):
    stats = {}

    # Read in comments file
    df_comments = pd.read_csv(os.path.join(data_path, "COMMENTS.tsv"), delimiter="\t")
    # Drop raw text column to save memory
    df_comments.drop("TEXT-RAW", axis=1, inplace=True)
    stats["nr_comments_og"] = len(df_comments)

    # Read in discussions file and rename TITLE column to DISCUSSION-TITLE
    df_discussions = pd.read_csv(
        os.path.join(data_path, "DISCUSSIONS.tsv"), delimiter="\t"
    )
    df_discussions.rename(columns={"TITLE": "DISCUSSION-TITLE"}, inplace=True)
    stats["nr_discussions_og"] = len(df_discussions)

    # Read in pages file and rename TITLE column to PAGE-TITLE
    df_pages = pd.read_csv(os.path.join(data_path, "PAGES.tsv"), delimiter="\t")
    df_pages.rename(columns={"TITLE": "PAGE-TITLE"}, inplace=True)

    print("Finished reading in data files. Now merging...")

    # Left merge DISCUSSION-TITLE and PAGE-ID onto comments dataframe
    # Discussions without comments will be left out
    df_merged = pd.merge(
        df_comments,
        df_discussions[["DISCUSSION-ID", "PAGE-ID", "DISCUSSION-TITLE"]],
        on="DISCUSSION-ID",
        how="left",
    )
    # Delete original dfs to save memory
    del df_comments
    del df_discussions

    # Merge PAGE-TITLE onto comments dataframe
    df_merged = pd.merge(
        df_merged, df_pages[["PAGE-ID", "PAGE-TITLE"]], on="PAGE-ID", how="left"
    )
    # Delete original df to save memory
    del df_pages

    # Add comment number as separate column
    df_merged["COMMENT-NR"] = (
        df_merged["COMMENT-ID"].apply(lambda x: x.split("-")[-1]).astype(int)
    )

    # Sort df based on discussion id and comment number
    df_merged = df_merged.sort_values(by=["DISCUSSION-ID", "COMMENT-NR"])

    print("Finished merging data files. Now grouping comments...")

    # Group by discussion and aggregate comments into lists
    df_grouped = (
        df_merged.groupby(
            ["DISCUSSION-ID", "PAGE-TITLE", "DISCUSSION-TITLE", "PAGE-ID"]
        )
        .agg(
            COMMENTS_ID=("COMMENT-ID", list),
            TEXT_CLEAN=("TEXT-CLEAN", list),
            COMMENT_NR=("COMMENT-NR", list),
            USER=("USER", list),
            TIMESTAMP=("TIMESTAMP", list),
            LEVEL=("LEVEL", list),
        )
        .reset_index()
    )

    del df_merged
    print("Finished grouping comments. Now reshaping comments column.")

    # Construct the COMMENTS column by combining all lists into dictionaries
    df_grouped["COMMENTS"] = df_grouped.apply(
        lambda x: [
            {
                "COMMENT-ID": cid,
                "TEXT-CLEAN": text,
                "COMMENT-NR": cnr,
                "USER": user,
                "TIMESTAMP": ts,
                "LEVEL": lvl,
            }
            for cid, text, cnr, user, ts, lvl in zip(
                x["COMMENTS_ID"],
                x["TEXT_CLEAN"],
                x["COMMENT_NR"],
                x["USER"],
                x["TIMESTAMP"],
                x["LEVEL"],
            )
        ],
        axis=1,
    )

    # Drop the intermediate list columns as they are no longer needed
    df_grouped.drop(
        columns=[
            "COMMENTS_ID",
            "TEXT_CLEAN",
            "COMMENT_NR",
            "USER",
            "TIMESTAMP",
            "LEVEL",
        ],
        inplace=True,
    )

    print("Finished reshaping comments column. ")

    stats["nr_discussions_no_comments"] = stats["nr_discussions_og"] - len(df_grouped)

    if return_stats:
        return df_grouped, stats
    return df_grouped


if __name__ == "__main__":
    args = create_arg_parser()
    data_path = args.data_folder
    output_file = args.output_file

    df_grouped, stats = group_comments(data_path)

    # Create separate test df
    df_test = df_grouped.sample(n=args.test_size, random_state=42)
    df_grouped.drop(df_test.index, inplace=True)

    # Write both dataframes to jsonl files
    output_path = os.path.join(data_path, output_file + ".jsonl")
    output_path_test = os.path.join(data_path, output_file + "_test.jsonl")

    df_test.to_json(output_path_test, orient="records", lines=True)
    print(f"Test file saved as {output_path_test}")

    # df_grouped.to_json(output_path, orient="records", lines=True)

    # Write main data to file per row to save memory
    with open(output_path, "w") as file:
        # Iterate over each row in the DataFrame
        for index, row in df_grouped.iterrows():
            # Convert the row to a JSON string
            json_string = row.to_json()
            # Write the JSON string as a new line in the file
            file.write(json_string + "\n")
    print(f"Main file saved as {output_path}")

    print('\nStatistics:')
    print(f'Original nr of comments: {stats['nr_comments_og']}')
    print(f'Original nr of discussions: {stats['nr_discussions_og']}')
    print(f'Nr of discussions without comments (left out): {stats['nr_discussions_no_comments']}')
