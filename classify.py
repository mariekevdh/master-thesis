from datasets import load_dataset
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import re


def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-in", "--input_file", type=str, help="Input file", required=True
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Name of output file",
        default="output.jsonl",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="Batch size",
        default=4,
    )
    parser.add_argument(
        "-s",
        "--selection",
        type=int,
        help="Chunk that will be processed. 1 means first 1000, 2 means second 1000, etc. If 0, all data will be processed.",
        default=0,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="Model name",
        default="microsoft/Phi-3-mini-4k-instruct",
    )
    parser.add_argument(
        "-p",
        "--prompt_type",
        type=str,
        help="Prompt type to use for generating labels",
        default="label",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=bool,
        help="If set to true, classification report will be printed. A column 'TRUE' is expected in the input data with the true labels",
        default=False,
    )
    args = parser.parse_args()
    return args


def normalize_text(text):
    # Characters to normalize if they appear consecutively
    characters = r'[():=?!\'"\/&*-]+'
    # Replace consecutive occurrences of any characters in the set with a single instance
    normalized_text = re.sub(r"([" + re.escape(characters) + r"])\1+", r"\1", text)

    filter_pattern = r"[^a-zA-Z0-9()?=+!:\'\"/\-&*\s]"
    cleaned_text = re.sub(filter_pattern, "", normalized_text)

    return cleaned_text


def generate_prompt(example, prompt_type):
    title = normalize_text(str(example["DISCUSSION-TITLE"]))
    first_comment = normalize_text(example["COMMENTS"][0]["TEXT-CLEAN"])
    if prompt_type == "label":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy, or not. "
                "You will be provided with the title and first comment of the discussion. Based on this excerpt, answer 'yes' if it is, answer 'no' if it is not. "
                "Response Template:\n"
                "Yes/No\n\n"
                "Discussion Excerpt:\n"
                f"{title}\n"
                f"{first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "label_explanation":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy, or not. "
                "You will be provided with the title and first comment of the discussion. Based on this excerpt, answer 'yes' if it is, answer 'no' if it is not and briefly explain your answer.\n\n"
                "Response Template:\n"
                "Yes/No. Explanation: [Explain your answer]\n\n"
                "Discussion Excerpt:\n"
                f"{title}\n"
                f"{first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "label_explanation_examples_yes":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy, or not. "
                "You will be provided with the title and first comment of the discussion. Based on this excerpt, answer 'yes' if it is, answer 'no' if it is not and briefly explain your answer. "
                "Examples of when your answer should be 'yes' are when the primary focus is on the verification of facts, the accuracy of the content from the associated Wikipedia article, the correction or identification of incorrect content, the validity or absence of sources or similar concepts.\n\n"
                "Response Template:\n"
                "Yes/No. Explanation: [Explain your answer]\n\n"
                "Discussion Excerpt:\n"
                f"{title}\n"
                f"{first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "label_explanation_examples_both":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy, or not. "
                "You will be provided with the title and first comment of the discussion. Based on this excerpt, answer 'yes' if it is, answer 'no' if it is not and briefly explain your answer. "
                "Examples of when your answer should be 'yes' are when the primary focus is on the verification of facts, the accuracy of the content from the associated Wikipedia article, the correction or identification of incorrect content, the validity or absence of sources or similar concepts. "
                "Examples of when your answer should be 'no' are when the discussion does not focus on these aspects, but for example on writing style, adding content, potential bias, content organization, general commentary, technical issues such as broken links, etc. Also classify as 'No' if the discussion lacks sufficient context for a clear classification or if the content does not make sense.\n\n"
                "Response Template:\n"
                "Yes/No. Explanation: [Explain your answer]\n\n"
                "Discussion Excerpt:\n"
                f"{title}\n"
                f"{first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "label_examples_both":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy, or not. "
                "You will be provided with the title and first comment of the discussion. Based on this excerpt, answer 'yes' if it is, answer 'no' if it is not. "
                "Examples of when your answer should be 'yes' are when the primary focus is on the verification of facts, the accuracy of the content from the associated Wikipedia article, the correction or identification of incorrect content, the validity or absence of sources or similar concepts. "
                "Examples of when your answer should be 'no' are when the discussion does not focus on these aspects, but for example on writing style, adding content, potential bias, content organization, general commentary, technical issues such as broken links, etc. Also classify as 'No' if the discussion lacks sufficient context for a clear classification or if the content does not make sense.\n\n"
                "Response Template:\n"
                "Yes/No.\n\n"
                "Discussion Excerpt:\n"
                f"{title}\n"
                f"{first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "highlight":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy, or not. "
                "You will be provided with the title and first comment of the discussion. Based on this excerpt, answer 'yes' if it is, answer 'no' if it is not and highlight the phrase or phrases that most influenced your answer. "
                "Examples of when your answer should be 'yes' are when the primary focus is on the verification of facts, the accuracy of the content from the associated Wikipedia article, the correction or identification of incorrect content, the validity or absence of sources or similar concepts. "
                "Examples of when your answer should be 'no' are when the discussion does not focus on these aspects, but for example on writing style, adding content, potential bias, content organization, general commentary, technical issues such as broken links, etc. Also classify as 'No' if the discussion lacks sufficient context for a clear classification or if the content does not make sense.\n\n"
                "Response Template:\n"
                "Yes/No. Phrase(s): [Highlight specific parts of the discussion that influenced your answer.]\n\n"
                "Discussion Excerpt:\n"
                f"{title}\n"
                f"{first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "summary":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy, or not. "
                "You will be provided with the title and first comment of the discussion. Based on this excerpt, concisely summarize the text and answer 'yes' if it is, answer 'no' if it is not. "
                "Examples of when your answer should be 'yes' are when the primary focus is on the verification of facts, the accuracy of the content from the associated Wikipedia article, the correction or identification of incorrect content, the validity or absence of sources or similar concepts. "
                "Examples of when your answer should be 'no' are when the discussion does not focus on these aspects, but for example on writing style, adding content, potential bias, content organization, general commentary, technical issues such as broken links, etc. Also classify as 'No' if the discussion lacks sufficient context for a clear classification or if the content does not make sense.\n\n"
                "Response Template:\n"
                "Yes/No. Summary: [Concise summary of the discussion excerpt]\n\n"
                "Discussion Excerpt:\n"
                f"{title}\n"
                f"{first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "highlight_explanation":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy, or not. "
                "You will be provided with the title and first comment of the discussion. Based on this excerpt, answer 'yes' if it is, answer 'no' if it is not and highlight the phrase or phrases that most influenced your answer. Briefly explain how the selected excerpts directly relate to the classification of the discussion regarding factual accuracy and verifiability. "
                "Examples of when your answer should be 'yes' are when the primary focus is on the verification of facts, the accuracy of the content from the associated Wikipedia article, the correction or identification of incorrect content, the validity or absence of sources or similar concepts. "
                "Examples of when your answer should be 'no' are when the discussion does not focus on these aspects, but for example on writing style, adding content, potential bias, content organization, general commentary, technical issues such as broken links, etc. Also classify as 'No' if the discussion lacks sufficient context for a clear classification or if the content does not make sense.\n\n"
                "Response Template:\n"
                "Yes/No. Explanation: [Cite specific parts of the discussion that influenced your decision and explain their relevance to your classification.]\n\n"
                "Discussion Excerpt:\n"
                f"{title}\n"
                f"{first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "instructions_explanation":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy.\n\n"
                "Instructions:\n"
                "1. You will be provided with the title and first comment of the discussion. Carefully read the provided title and discussion excerpt.\n"
                "2. Analyze whether the main theme of the discussion is primarily concerned with the verifiability or factual accuracy of the content of the associated Wikipedia article.\n"
                "3. Classify the discussion as:\n"
                "   - 'Yes' if the primary focus is on verifying facts, ensuring the accuracy of content, removing, correcting or identifying incorrect content or mistakes, assessing the validity of references, the absence of references, providing references, or similar aspects.\n"
                "   - 'No' if the discussion does not focus on these aspects, but for example on writing style, adding content, potential bias, content organization, general commentary, technical issues such as broken links, etc. Also classify as 'No' if the discussion lacks sufficient context for a clear classification or if the content does not make sense.\n"
                "4. Support your classification with specific citations from the discussion. Briefly explain how the selected excerpts directly relate to the classification of the discussion regarding factual accuracy and verifiability.\n\n"
                "Response Template:\n",
                "Yes/No. Explanation: [Cite specific parts of the discussion that influenced your decision and explain their relevance to your classification.]\n\n"
                "Discussion Excerpt:\n"
                f"Title: {title}\n"
                f"First comment: {first_comment}"
                "<|end|><|assistant|>"
            )
        }
    elif prompt_type == "instructions_label":
        return {
            "PROMPT": (
                "<|user|>Objective:\n"
                "Determine whether the primary focus of the Wikipedia Talk Page discussion below is on verifiability or factual accuracy.\n\n"
                "Instructions:\n"
                "1. You will be provided with the title and first comment of the discussion. Carefully read the provided title and discussion excerpt.\n"
                "2. Analyze whether the main theme of the discussion is primarily concerned with the verifiability or factual accuracy of the content of the associated Wikipedia article.\n"
                "3. Classify the discussion as:\n"
                "   - 'Yes' if the primary focus is on verifying facts, ensuring the accuracy of content, removing, correcting or identifying incorrect content or mistakes, assessing the validity of references, the absence of references, providing references, or similar aspects.\n"
                "   - 'No' if the discussion does not focus on these aspects, but for example on writing style, adding content, potential bias, content organization, general commentary, technical issues such as broken links, etc. Also classify as 'No' if the discussion lacks sufficient context for a clear classification or if the content does not make sense.\n"
                "Response Template:\n"
                "Yes/No.\n\n"
                "Discussion Excerpt:\n"
                f"Title: {title}\n"
                f"First comment: {first_comment}"
                "<|end|><|assistant|>"
            )
        }



# Function to convert examples to model inputs with padding
def convert_to_tensors(example):
    tokens = tokenizer(
        example["PROMPT"],
        padding="max_length",
        truncation=True,
        max_length=1000,
        return_tensors="pt",
    )
    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask}


def generate_text(input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=500,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_length = input_ids.shape[1]
    return [
        tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
        for output in outputs
    ]


def get_labels(output_list):
    labels = []
    for output in output_list:
        if output.lower().strip().startswith("yes"):
            labels.append("yes")
        else:
            labels.append("no")
    return labels


if __name__ == "__main__":
    args = create_arg_parser()
    input_file = args.input_file
    output_file = args.output_file
    batch_size = args.batch_size
    model_name = args.model_name
    selection = args.selection

    # Load dataset and generate prompts
    dataset = load_dataset("json", data_files=input_file)["train"]
    dataset = dataset.map(lambda example: generate_prompt(example, args.prompt_type))

    if selection > 0:
        # Select chunk of data
        start_index = (selection - 1) * 20000
        end_index = start_index + 20000
        indices = list(range(start_index, end_index))
        dataset = dataset.select(indices)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = dataset.map(convert_to_tensors, batched=True)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model, bfloat16 to save time and memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    output_texts = []

    for batch in tqdm(dataloader, desc="Generating Texts"):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        generated_texts = generate_text(input_ids, attention_mask)
        output_texts.extend(generated_texts)

    if args.test:
        df = dataset.to_pandas()[["DISCUSSION-ID", "PROMPT", "TRUE"]]
    else:
        df = dataset.to_pandas()[["DISCUSSION-ID", "PROMPT"]]
    df["output"] = output_texts
    pred_labels = get_labels(output_texts)
    df["pred_labels"] = pred_labels

    df.to_json(output_file, orient="records", lines=True)
    print(f"Results saved as {output_file}")

    print("Classified as yes: ", pred_labels.count('yes'))

    if args.test:
        print("\n")
        true_labels = dataset["TRUE"]
        cr = classification_report(true_labels, pred_labels)
        print(cr)
        cr_output_path = output_file[:-6] + "_cr.txt"
        xl_output_path = output_file[:-6] + "_mismatch.xlsx"
        mismatch_df = df[df["TRUE"] != df["pred_labels"]]
        mismatch_df.to_excel(xl_output_path)
        print(f"Mismatch overview saved as {xl_output_path}")
        with open(cr_output_path, "w") as f:
            f.write(cr)
        print(f"Classification report saved as {cr_output_path}")
