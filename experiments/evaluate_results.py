import argparse
import json
from src.evaluation.metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    print(json.dumps(compute_metrics(args.path), indent=2))

if __name__ == "__main__":
    main()
