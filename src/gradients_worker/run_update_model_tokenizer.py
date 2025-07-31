#!/usr/bin/env python3
import argparse
import asyncio
import sys

from gradients_worker.utils import update_model_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Update a model with a new tokenizer and upload to HF hub"
    )
    parser.add_argument(
        "--model-id", required=True, help="The HF ID of the model to update"
    )
    parser.add_argument(
        "--tokenizer-id", required=True, help="The HF ID of the tokenizer to use"
    )

    args = parser.parse_args()

    try:
        result = asyncio.run(
            update_model_tokenizer(
                model_id=args.model_id, tokenizer_id=args.tokenizer_id
            )
        )
        print(result)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
