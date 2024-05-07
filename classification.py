from datasets import load_dataset
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        type=str,
        help="LLM name from Hugginface or local path",
    ),
    parser.add_argument(
        "-t",
        "--temperature",
        default=0.7,
        type=float,
        help="Temperature value between 0.0 and 1.0 for the model.",
    ),
    parser.add_argument(
        "-in",
        "--input_filename",
        required=True,
        type=str,
        help="Path do the input data (jsonl file)",
    ),
    parser.add_argument(
        "-out",
        "--output_filename",
        default="labeled_output.tsv",
        type=str,
        help="Path do the output file (tsv file)",
    ),
    parser.add_argument(
        "-p",
        "--prompt_type",
        default="simple_oneshot",
        type=str,
        help="Prompt type. Possible options: TBA",
    ),

    args = parser.parse_args()
    return args


def format_discussion(comments):
    pass


def create_prompt(example, prompt_type):
    # Prompt templates
    page_title = example["PAGE-TITLE"]  # (Wikipedia page title)
    discussion_title = example["DISCUSSION-TITLE"]  # (Talk page discussion title)
    discussion_text_first_comment = str(example["COMMENTS"][0][
        "TEXT-CLEAN"
    ]).replace("\n", " ")  # (first comment of discussion)
    # discussion_text_full = format_discussion(
    #     example["COMMENTS"]
    # )  # (full discussion with all comments)

    if prompt_type == "simple_oneshot":
        return f"""Determine if the following Wikipedia Talk Page discussion is about fact-checking. Answer YES or NO and then explain your answer. Discussion: {discussion_text_first_comment}"""


def generate_output(example, tokenizer, model, temperature, prompt_type, device):
    prompt = create_prompt(example, prompt_type)

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate attention mask
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)

    # Generate text
    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        num_return_sequences=1,
        max_length=1000,
        # do_sample=True,
        # temperature=temperature
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": generated_text}


if __name__ == "__main__":
    args = create_arg_parser()

    # Load in test set
    dataset = load_dataset("json", data_files=args.input_filename)["train"]
    dataset = dataset.select(range(5))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using GPU')
    else:
        device = torch.device("cpu")
        print('Using CPU')

    torch.set_default_dtype(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    # Generate ouput
    processed_data = dataset.map(
        generate_output,
        fn_kwargs={
            "prompt_type": args.prompt_type,
            "tokenizer": tokenizer,
            "model": model,
            "temperature": args.temperature,
            "device": device
        },
    )

    # Create DataFrame from processed_data and save to a file
    df = pd.DataFrame(processed_data)
    df[['DISCUSSION-ID', 'generated_text']].to_csv(args.output_filename, sep='\t', index=False)
