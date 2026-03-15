import pandas as pd


def compute_metrics(predictions, targets):
    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(predictions, targets):
        pred_set = set(tuple(x) for x in pred)
        target_set = set(tuple(x) for x in target)

        tp += len(pred_set & target_set)
        fp += len(pred_set - target_set)
        fn += len(target_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
    }


def save_metrics(rows, output_path: str):
    df = pd.DataFrame(rows, columns=["model", "dataset", "precision", "recall", "micro_f1"])
    df.to_csv(output_path, index=False)
