#!/usr/bin/env python3
import argparse
import asyncio
import sys

from gradients_worker.utils import merge_and_upload_model


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA with base model and upload to HF hub"
    )
    parser.add_argument(
        "--lora-model-id", required=True, help="The HF ID of the LoRA adapter"
    )
    parser.add_argument(
        "--base-model-id", required=True, help="The HF ID of the base model"
    )
    parser.add_argument(
        "--anonimize",
        type=bool,
        default=True,
        help="Whether to anonymize the model name",
    )

    args = parser.parse_args()

    try:
        result = asyncio.run(
            merge_and_upload_model(
                lora_model_id=args.lora_model_id,
                base_model_id=args.base_model_id,
                anonimize=args.anonimize,
            )
        )
        print(result)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
