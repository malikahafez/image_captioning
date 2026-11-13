"""
evaluate_metrics.py

Scores a trained image-captioning model with the metrics from our slides:
BLEU-4, METEOR, ROUGE-L and CIDEr-D (CIDEr-D is the COCO-server version of CIDEr).

It does NOT run the model. It only reads the captions the model already
generated (the json written by test.py) and compares them with the human
captions from COCO.

Usage (from the project root, venv active):
    python evaluate_metrics.py --preds results/prelstm.json
    python evaluate_metrics.py --preds results/a.json results/b.json   # side-by-side table

Requirements: pycocoevalcap, Java on PATH, and the project path must contain
no spaces (METEOR's jar fails on paths with spaces).
"""
import argparse
import json
from collections import defaultdict

# Official COCO caption-evaluation scorers (installed with: pip install pycocoevalcap)
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # lowercases + strips punctuation
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor                  # runs a Java jar under the hood
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider                     # this implementation is CIDEr-D


def load_predictions(path):
    """Read the model's captions json -> {image_id: [{'caption': text}]}."""
    with open(path) as f:
        preds = json.load(f)

    res = {}
    for p in preds:
        # The model's vocabulary adds <start>/<end> markers. They are not real
        # words, so remove them or they would be counted as words and lower the scores.
        text = p["caption"].replace("<start>", "").replace("<end>", "").strip()
        # The scorers expect a list with one caption per image, each wrapped in a dict.
        res[p["image_id"]] = [{"caption": text}]
    return res


def load_references(path, image_ids):
    """Read the human captions json -> {image_id: [{'caption': text}, ...]},
    keeping only the images we have predictions for."""
    with open(path) as f:
        data = json.load(f)

    # captions_val2014.json covers ~40k images but the model only captioned our
    # test split, so keep only those images' references (about 5 per image).
    wanted = set(image_ids)
    gts = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] in wanted:
            gts[ann["image_id"]].append({"caption": ann["caption"]})
    return dict(gts)


def evaluate(pred_path, ref_path):
    """Return a dict of metric name -> score (0 to ~1) for one predictions file."""
    res = load_predictions(pred_path)
    gts = load_references(ref_path, res.keys())

    # Stop early if any captioned image has no human captions to compare against.
    missing = set(res.keys()) - set(gts.keys())
    assert not missing, f"{len(missing)} predicted images have no references"

    # Tokenize both sides the same way (lowercase, punctuation removed) so that
    # "a river ." and "A river" count as the same words.
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # (scorer, names of the scores it returns). Bleu(4) returns four scores
    # at once (cumulative BLEU-1 to BLEU-4); the others return a single score.
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE-L"]),
        (Cider(), ["CIDEr-D"]),
    ]

    results = {}
    for scorer, names in scorers:
        # compute_score returns (overall_score, per_image_scores); we only need the overall one.
        score, _ = scorer.compute_score(gts, res)
        if isinstance(score, list):          # BLEU: one value per n-gram size
            for name, s in zip(names, score):
                results[name] = s
        else:                                # METEOR / ROUGE-L / CIDEr-D: a single value
            results[names[0]] = score
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", default="data/annotations/captions_val2014.json",
                        help="COCO human captions json")
    parser.add_argument("--preds", nargs="+", required=True,
                        help="one or more prediction json files (one column each in the table)")
    parser.add_argument("--out", default=None,
                        help="optional json file to save all scores (includes BLEU-1 to BLEU-3)")
    args = parser.parse_args()

    # Score every predictions file against the same references.
    all_results = {p: evaluate(p, args.refs) for p in args.preds}

    # Print only the four metrics from the slides, multiplied by 100 (usual reporting style).
    keep = ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"]
    print(f"{'Metric':<10}" + "".join(f"{p.split('/')[-1]:>20}" for p in args.preds))
    for m in keep:
        print(f"{m:<10}" + "".join(f"{100 * r[m]:>20.2f}" for r in all_results.values()))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()