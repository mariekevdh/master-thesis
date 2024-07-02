import argparse
from datasets import load_dataset, Dataset
import pandas as pd
import re

def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-in",
        "--input_file",
        type=str,
        help="Name of input file, without extension.",
        default="grouped_data",
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


def get_username(example):
    # Regular expressions to find patterns including an IP address or a username between '--' and '(talk)'
    # and the specific 'Preceding unsigned comment added by' pattern with IP and '(talk)'
    pattern1 = r'--\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[\w\s-]+?)\s*\(talk\)$'
    pattern2 = r'Preceding unsigned comment added by (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[\w\s-]+?)\s*$'
    pattern3 = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s*\(talk\)$'

    for i, comment in enumerate(example['COMMENTS']):
        text = str(comment['TEXT-CLEAN'])

        # Try the first pattern
        match = re.search(pattern1, text)
        if not match:
            # If no match, try the second pattern
            match = re.search(pattern2, text, flags=re.IGNORECASE)
        if not match:
            # If no match, try the second pattern
            match = re.search(pattern3, text)

        if match:
            # Extract the IP address or username from the matched group
            username = match.group(1).strip() 
            # If the USER field is empty, fill it with the extracted username
            if not example['COMMENTS'][i]['USER']:
                example['COMMENTS'][i]['USER'] = username

        # Remove the entire pattern 1 and 3 from the text and replace newlines with a space
        text = re.sub(pattern1, '', text)
        text = re.sub(pattern3, '', text)

        # Remove the 'unsigned' signature
        text = re.sub(r'—\s?(the\s+)?Preceding.*$', '', text, flags=re.IGNORECASE)

        # Remove any trailing dashes with optional surrounding whitespace
        text = re.sub(r'\s*--\s*$', '', text)

        # Replace newlines with spaces and reduce multiple spaces to a single space
        text = re.sub('\s+', ' ', text).strip()

        # Update the cleaned text back into the dataset
        example['COMMENTS'][i]['TEXT-CLEAN'] = text

    return example

def filter_discussions(example):
    # Extract the text from the first comment's 'TEXT-CLEAN' field
    text = str(example['COMMENTS'][0]['TEXT-CLEAN']).lower()
    first_user = example['COMMENTS'][0]['USER']

    # Check if first user is not null/None
    if not first_user:
        return False

    # Check if the text starts with 'Hello fellow Wikipedians'
    if text.startswith('hello fellow wikipedians'):
        return False

    # Check if the text contains both 'fair use' and 'image'
    if 'fair use' in text and 'image' in text:
        return False

    # Check if the text has less than 10 words
    if len(text.split()) < 10:
        return False

    # Check if the text starts with 'This article was automatically assessed'
    if text.startswith('this article was automatically assessed'):
        return False

    # Check if discussion is merge request
    if 'merge' in text:
        return False

    # Check if first comment is not a first level comment
    if text.startswith(':'):
        return False
    
    # Check for bot messages (as first comment)
    if 'automated bot run' in text:
        return False

    # If none of the conditions are met, return True
    return True


def find_interaction(example):
    """Find discussions with interaction. Interaction means that user 1 has at least one comment after another user in the discussion."""
    users = [comment['USER'] for comment in example['COMMENTS']]
    if len(set(users)) > 1:
        user_two = None
        for i, user in enumerate(users):
            if i > 0:
                if user != users[0]:
                    user_two = i
                    break
        if user_two:
            if users[0] in users[user_two:]:
                return True
    return False


if __name__ == "__main__":
    args = create_arg_parser()

    # TBA because code was run in a notebook