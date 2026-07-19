# Runtime Bundle Notice

This bundle intentionally excludes `.env`, CT-RATE volumes, radiology report CSV files,
case indexes, and CT preview images. CT-RATE prohibits redistribution of dataset content;
each recipient must obtain access and download the data with their own Hugging Face token.

Included runtime assets:

- ChestCT-Agent source code and tests
- Official CT-CLIP source code
- `CT-CLIP_v2.pt` model checkpoint
- Text-classifier model and aggregate evaluation metrics
- A pinned `requirements-lock.txt` for recreating the main Python environment

Create a local `.env` from `.env.example`. Do not commit or share API tokens.
