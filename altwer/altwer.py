import jiwer
import json
import re
import string
from itertools import product


def preprocess_eval(reference):
    """
    Preprocess reference string to extract acceptable alternatives.
    Converts the reference string into all possible sentence combinations.

    Args:
        reference (str): A reference string containing alternatives in brackets.

    Returns:
        list: A list of all possible sentence combinations.
    """
    # Extract alternatives from brackets
    parts = re.findall(r'\[.*?\]', reference)

    # Create a list of alternatives for each bracket
    eval_combinations = []
    for part in parts:
        options = json.loads(part)  # Parse the JSON array directly
        eval_combinations.append(options)

    # Create all combinations of sentences
    base_sentence = re.sub(r'\[.*?\]', '{}', reference)
    combinations = [base_sentence.format(*combo) for combo in product(*eval_combinations)]
    return combinations


def wer(references, hypotheses, verbose=False, empty_text="<|nospeech|>", lowercase=False, remove_punctuation=False):
    """
    Calculate WER for a list of references and hypotheses, supporting multiple reference options.

    Args:
        references (list): A list of reference strings, where each string may
                           contain alternatives in brackets (e.g., ["[\"a\",\"b\"]"]).
        hypotheses (list): A list of hypothesis strings.
        verbose (bool): If True, print detailed output. Defaults to False.
        empty_text (str): Placeholder for empty hypotheses. Defaults to "<|nospeech|>".
                          Whisper uses "<|nocaptions|>" before version 3.
        lowercase (bool): If True, converts all input to lowercase. Defaults to False.
        remove_punctuation (bool): If True, removes punctuation from all input. Defaults to False.

    Returns:
        float: The word error rate (WER).
    """
    if len(references) != len(hypotheses):
        raise ValueError("Length of references and hypotheses must be the same.")

    total_errors = 0
    total_words = 0

    for idx, (reference, hypothesis) in enumerate(zip(references, hypotheses), start=1):
        hypothesis = hypothesis.strip() or empty_text
        reference = reference.strip()

        # Apply preprocessing options
        if lowercase:
            hypothesis = hypothesis.lower()
            reference = reference.lower()

        if remove_punctuation:
            punctuation = str.maketrans('', '', string.punctuation)
            hypothesis = hypothesis.translate(punctuation)
            reference = reference.translate(punctuation)

        # Generate all possible combinations for the reference field
        try:
            reference_combinations = preprocess_eval(reference)
        except Exception as e:
            raise ValueError(f"Error processing reference: {reference}") from e

        # Calculate WER for all combinations, find the closest match
        best_wer = float("inf")
        best_match = None

        if verbose:
            print(f"\nEntry {idx}:")
            print(f"Hypothesis: '{hypothesis}'")
            print("Reference combinations and their WER:")

        for ref_option in reference_combinations:
            try:
                wer_result = jiwer.compute_measures(ref_option, hypothesis)
                wer = wer_result["wer"]

                if verbose:
                    print(f"  - '{ref_option}': WER = {wer:.4f}")

                if wer < best_wer:
                    best_wer = wer
                    best_match = ref_option
            except Exception as e:
                raise RuntimeError(f"Error calculating WER for reference option: {ref_option}") from e

        if verbose and best_match is not None:
            print(f"Best match: '{best_match}' with WER = {best_wer:.4f}")

        total_errors += int(best_wer * len(best_match.split()))
        total_words += len(best_match.split())

    if total_words == 0:
        return 0.0

    return total_errors / total_words
