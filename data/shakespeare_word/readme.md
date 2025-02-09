# Shakespeare Word-Level Dataset

This directory contains the Shakespeare dataset processed at the word level for language modeling tasks.

## Files
- `input.txt`: Raw Shakespeare text
- `train.bin`: Training data (90% of dataset)
- `val.bin`: Validation data (10% of dataset)
- `meta.pkl`: Metadata containing vocabulary and mappings
- `prepare.py`: Script to process the raw text into word-level dataset

## Processing Details
- Words are tokenized with basic preprocessing
- Contractions and possessives are handled separately (e.g., "it's" → ["it", "'s"])
- Punctuation is preserved as separate tokens
- Special tokens: 
  - `<UNK>`: Unknown/rare words (frequency < 2)
  - `<EOL>`: End of line marker
- Words appearing less than twice are mapped to `<UNK>`

## Usage
Run `prepare.py` to download and process the dataset:
```bash
python prepare.py
``` 