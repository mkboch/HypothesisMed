import pandas as pd

def compute_metrics(path):
    df = pd.read_json(path, lines=True)

    result = {"n": int(len(df))}
    if len(df) == 0:
        return result

    result["answer_accuracy"] = float((df["pred_answer"] == df["gold_answer"]).mean())

    if "pred_space_label" in df.columns:
        mask = df["pred_space_label"].notna()
        result["space_label_coverage"] = float(mask.mean())
        if mask.sum() > 0:
            result["space_label_accuracy"] = float(
                (df.loc[mask, "pred_space_label"] == df.loc[mask, "gold_space_label"]).mean()
            )

    invalid = df[df["gold_space_label"].isin(["INCOMPLETE", "CONTRADICTED"])]
    if len(invalid) > 0:
        result["false_commitment_rate"] = float(
            (invalid["pred_answer"] != invalid["gold_answer"]).mean()
        )

    return result
