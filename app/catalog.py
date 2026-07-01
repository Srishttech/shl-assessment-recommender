import json

from app.config import CATALOG_PATH


def load_catalog(path: str = CATALOG_PATH):
    """Load the repaired SHL catalog JSON. Preserved exactly as in the notebook."""
    with open(path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return catalog


def build_texts(catalog):
    """Builds the text blob for each catalog item, used for embeddings.
    Logic unchanged from the notebook."""
    texts = []
    for item in catalog:
        text = f"""
Name: {item.get('name', '')}
Description:
{item.get('description', '')}
Job Levels:
{', '.join(item.get('job_levels', []))}
Categories:
{', '.join(item.get('keys', []))}
Languages:
{', '.join(item.get('languages', []))}
Duration:
{item.get('duration', '')}
Remote:
{item.get('remote', '')}
Adaptive:
{item.get('adaptive', '')}
"""
        texts.append(text)
    return texts
