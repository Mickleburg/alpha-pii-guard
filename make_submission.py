import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Convert prediction CSV to submission CSV with columns: id,Prediction"
    )
    parser.add_argument("input_csv", help="Path to source CSV")
    parser.add_argument("output_csv", help="Path to output CSV")
    parser.add_argument(
        "--id-column",
        default="id",
        help="Source id column name, default: id",
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Source prediction column name, default: prediction",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.id_column not in df.columns:
        raise ValueError(
            f"Column '{args.id_column}' not found. Available columns: {list(df.columns)}"
        )

    if args.prediction_column not in df.columns:
        raise ValueError(
            f"Column '{args.prediction_column}' not found. Available columns: {list(df.columns)}"
        )

    submission = pd.DataFrame({
        "id": df[args.id_column],
        "Prediction": df[args.prediction_column],
    })

    submission.to_csv(args.output_csv, index=False)
    print(f"Saved {len(submission)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()