import random
from copy import deepcopy
from src.utils.io import read_jsonl, write_jsonl

def add_none(options):
    opts = deepcopy(options)
    opts["E"] = "None of the above"
    return opts, "E"

def original_valid(row):
    r = deepcopy(row)
    r["space_label"] = "VALID"
    r["transform"] = "original"
    return r

def remove_correct(row):
    r = deepcopy(row)
    correct = r["answer"]
    if correct in r["options"]:
        del r["options"][correct]
    r["options"], lab = add_none(r["options"])
    r["answer"] = lab
    r["space_label"] = "INCOMPLETE"
    r["transform"] = "missing_correct"
    return r

def adversarial_swap(row):
    r = deepcopy(row)
    correct = r["answer"]
    wrong_values = [v for k, v in r["options"].items() if k != correct]
    if wrong_values and correct in r["options"]:
        r["options"][correct] = random.choice(wrong_values)
    r["options"], lab = add_none(r["options"])
    r["answer"] = lab
    r["space_label"] = "CONTRADICTED"
    r["transform"] = "adversarial_wrong_space"
    return r

def main():
    rows = read_jsonl("datasets/processed/original.jsonl")
    out = []
    for row in rows:
        out.append(original_valid(row))
        out.append(remove_correct(row))
        out.append(adversarial_swap(row))
    write_jsonl("datasets/transformed/hypothesismed_eval.jsonl", out)
    print(f"Saved {len(out)} transformed examples.")

if __name__ == "__main__":
    main()
