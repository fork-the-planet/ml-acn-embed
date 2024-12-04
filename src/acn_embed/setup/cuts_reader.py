import gzip
import json
from pathlib import Path


def read_cuts_utts(path: Path):
    with (
        gzip.open(path, "rt", encoding="utf8")
        if path.name.endswith(".gz")
        else open(path, "r", encoding="utf8")
    ) as read_file:
        for read_line in read_file:
            jsonl = json.loads(read_line)
            utt_id = jsonl["id"]

            assert len(jsonl["supervisions"]) == 1
            supervision = jsonl["supervisions"][0]
            assert len(jsonl["recording"]["sources"]) == 1

            utt = {
                "type": "utt",
                "utt_id": utt_id,
                "orig_text": supervision["custom"]["texts"][0],
                "length_s": jsonl["duration"],
            }
            yield utt
