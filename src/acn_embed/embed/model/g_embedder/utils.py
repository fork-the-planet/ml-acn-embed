import json
from pathlib import Path


def get_subword_info(subword_json_path: Path):
    with subword_json_path.open("r", encoding="utf8") as fobj:
        subword_idx_to_str = json.load(fobj)
    assert isinstance(subword_idx_to_str, list)
    subword_str_to_idx = {subword_idx_to_str[idx]: idx for idx in range(len(subword_idx_to_str))}
    return subword_idx_to_str, subword_str_to_idx
