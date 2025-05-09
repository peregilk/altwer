# altwer

`altwer` is a Python package designed for evaluating automatic speech recognition (ASR) results, particularly for languages where multiple correct spellings or expressions of the same word are acceptable. This phenomenon, known as **orthographic variation** or **linguistic variability**, is common in languages like Norwegian, Danish, Dutch, and some dialect-rich languages, where regional or formal differences influence spelling and vocabulary. Other languages might allow spellings like both "e-mail" and "email".

For instance, in Norwegian, both "matta" and "matten" might be correct spelling of "the mat" depending on regional usage or style, just as both "broa" and "broen" are valid spellings for "the bridge." Such variability poses challenges for ASR evaluation, as standard WER metrics treat these as errors.

---

## The Problem

In traditional WER evaluation:
1. **Normalization**: Techniques like lowercasing or removing punctuation are often used to reduce variability. However, these methods are insufficient when evaluating languages with valid orthographic or lexical differences, as they cannot account for meaning-preserving variations.
2. **Ambiguity in Transcriptions**: Some transcription guidelines may allow optional elements like fillers (e.g., "uh", "eh", "ah") or minor regional variations, but standard WER evaluation penalizes all deviations from a single reference transcription.

For example:
- Hypothesis: "katten ligger på matta"
- Reference: "katten ligger på matten"

This would result in an error in standard WER, despite both being correct.

---

## The Suggested Solution

**altwer** addresses this issue by allowing references to specify multiple valid alternatives. 

1. The **altwer**-package computes the WER by considering all alternatives in the reference and selecting the one that minimizes the error. If alternate spellings are specified, the best match is computed automatically. In cases where alternate spellings are not specified, **altwer** should give the same result as **jiwer**. This approach makes **altwer** ideal for:
- **Handling Orthographic Variation**: Allows multiple correct spellings. For example: `[matta|matten]` or `[organization|organisation]` or `[email|e-mail]`.
- **Optional Fillers**: Handles other cases where you do not want variations to be counted as errors. For example: `[eh|ah|uhh|...|]` or `[WHO|World Health Organization]`.


2. Use LLMs to automate the creation of this format. Experiments indicates that prompted correctly a Reasoning-model is able to create at least a first draft of this format. We provide some example tempates for doing the conversion.
- [Norwegian Bokmål example template](norwegian_template.txt)
- [English example template](english_template.txt)
  
---

## Installation

Install with pip:

```bash
pip install altwer
```

## Usage

```python
from altwer import wer

references = [
    "[jenta|jenten] [jogga|jogget] på [broa|broen|brua|bruen]",
    "[katten|katta] ligger på [matta|matten]",
    "Det var en fin dag."
]
hypotheses = [
    "jenta jogga på broa",
    "katten ligger på matta",
    "Det var en fin dag."
]

# Calculate WER
wer_score = wer(references, hypotheses, verbose=True, lowercase=True, remove_punctuation=True)
print(f"WER: {wer_score:.4f}")
```

## License

MIT
