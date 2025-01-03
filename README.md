
# altwer

`altwer` is a Python package for calculating Word Error Rate (WER) with support for multiple reference options. It mimics the behavior of `jiwer` but allows more flexibility with optional parameters.

## Features

- Support for references with multiple alternatives.
- Handles empty predictions with a customizable placeholder (`<|nospeech|>` by default).
- Optional preprocessing:
  - Convert text to lowercase.
  - Remove punctuation.
- Verbose mode for detailed debugging.

## Installation

Install with pip:

```bash
pip install altwer
```

## Usage

```python
from altwer import wer

references = [
    '["jenta","jenten"] ["jogga","jogget"] på ["broa","broen","brua","bruen"]',
    '["katten","katta"] ligger på ["matta","matten"]',
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
