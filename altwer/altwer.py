import json
import re
import string
from itertools import product

import jiwer


def _compute_wer(reference, hypothesis):
    """
    Compute the word error rate for a reference/hypothesis pair while staying
    compatible with multiple jiwer versions.
    """
    if hasattr(jiwer, "compute_measures"):
        return jiwer.compute_measures(reference, hypothesis)["wer"]
    if hasattr(jiwer, "process_words"):
        return jiwer.process_words(reference, hypothesis).wer
    return jiwer.wer(reference, hypothesis)


def _extract_options(part):
    """
    Return the possible alternatives represented within a single bracket.
    Supports both the modern pipe-separated format and the legacy JSON array format.
    """
    if "|" in part:
        return [option.strip() for option in part.split("|")]

    try:
        parsed = json.loads(f"[{part}]")
        return [str(option) for option in parsed]
    except json.JSONDecodeError:
        return [part.strip()]


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
    parts = re.findall(r'\[(.*?)\]', reference)

    # Create a list of alternatives for each bracket
    eval_combinations = []
    for part in parts:
        options = _extract_options(part)
        eval_combinations.append(options)

    # Create all combinations of sentences
    base_sentence = re.sub(r'\[.*?\]', '{}', reference)
    combinations = [base_sentence.format(*combo) for combo in product(*eval_combinations)]
    return combinations


def wer(
    references,
    hypotheses,
    verbose=False,
    empty_text="<|nospeech|>",
    lowercase=False,
    remove_punctuation=False
):
    """
    Calculate WER for a list of references and hypotheses, supporting multiple reference options.

    Args:
        references (list): A list of reference strings, where each string may
                           contain alternatives in brackets (e.g., ["[jenta|jenten]"]).
        hypotheses (list): A list of hypothesis strings.
        verbose (bool): If True, print detailed output. Defaults to False.
        empty_text (str): Placeholder for empty hypotheses. Defaults to "<|nospeech|>".
                          Whisper uses "<|nocaptions|>" before version 3.
        lowercase (bool): If True, converts all input to lowercase. Defaults to False.
        remove_punctuation (bool): If True, removes punctuation from all input. Defaults to False.

    Returns:
        float: The word error rate (WER), calculated as the average of WERs for all sentences.
    """
    if len(references) != len(hypotheses):
        raise ValueError("Length of references and hypotheses must be the same.")

    total_wer = 0.0
    count = 0

    for idx, (reference, hypothesis) in enumerate(zip(references, hypotheses), start=1):
        hypothesis = hypothesis.strip() or empty_text
        reference = reference.strip()

        # Generate all possible combinations for the reference field
        try:
            reference_combinations = preprocess_eval(reference)
        except Exception as e:
            raise ValueError(f"Error processing reference: {reference}") from e

        # Apply preprocessing options to expanded references and hypotheses
        if lowercase:
            hypothesis = hypothesis.lower()
            reference_combinations = [ref.lower() for ref in reference_combinations]

        if remove_punctuation:
            punctuation = str.maketrans('', '', string.punctuation)
            hypothesis = hypothesis.translate(punctuation)
            reference_combinations = [ref.translate(punctuation) for ref in reference_combinations]

        # Calculate WER for all combinations, find the closest match
        best_wer = float("inf")
        best_match = None

        if verbose:
            print(f"\nEntry {idx}:")
            print(f"Hypothesis: '{hypothesis}'")
            print("Reference combinations and their WER:")

        for ref_option in reference_combinations:
            try:
                wer = _compute_wer(ref_option, hypothesis)

                if verbose:
                    print(f"  - '{ref_option}': WER = {wer:.4f}")

                if wer < best_wer:
                    best_wer = wer
                    best_match = ref_option
            except Exception as e:
                raise RuntimeError(f"Error calculating WER for reference option: {ref_option}") from e

        if verbose and best_match is not None:
            print(f"Best match: '{best_match}' with WER = {best_wer:.4f}")

        total_wer += best_wer
        count += 1

    # Return the average WER across sentences
    return total_wer / count if count > 0 else 0.0
