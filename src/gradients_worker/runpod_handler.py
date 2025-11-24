"""
RunPod Serverless Handler for Model Merging

This handler runs in a RunPod serverless container with GPUs.
It receives merge requests, executes them, and returns the merged model repo ID.

Note: No manual cleanup (gc.collect, cache clearing) is needed in serverless -
the container is destroyed after the handler returns, freeing all memory automatically.
"""

import os
import tempfile
import uuid
from logging import getLogger

import runpod
import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = getLogger(__name__)


def merge_and_upload_model(
    lora_model_id: str,
    base_model_id: str,
    anonimize: bool,
    hf_token: str,
    hf_username: str,
) -> str:
    """Merge LoRA with base model and upload to HF hub.

    Uses device_map="auto" to automatically distribute across all available GPUs.

    Args:
        lora_model_id: The HF ID of the LoRA adapter
        base_model_id: The HF ID of the base model
        anonimize: Whether to anonymize the model name
        hf_token: HuggingFace API token
        hf_username: HuggingFace username for uploads

    Returns:
        str: The new model repo ID
    """
    logger.info(
        f"Merging {base_model_id} + {lora_model_id} on {torch.cuda.device_count()} GPUs"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        new_repo_name = (
            str(uuid.uuid4()) if anonimize else f"merged-{lora_model_id.split('/')[-1]}"
        )
        new_repo_id = f"{hf_username}/{new_repo_name}"

        api = HfApi(token=hf_token)
        if api.repo_exists(new_repo_id):
            logger.info(f"Model {new_repo_id} already exists, skipping merge")
            return new_repo_id

        merged_path = os.path.join(tmp_dir, "merged_model")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)

        # Load base model - device_map="auto" distributes across all GPUs
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=hf_token,
        )

        # Load and merge LoRA
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_model_id, token=hf_token)

        logger.info("Merging LoRA with base model...")
        merged_model = model.merge_and_unload()

        # Save merged model
        logger.info("Saving merged model...")
        merged_model.save_pretrained(merged_path, save_peft_format=False)
        tokenizer.save_pretrained(merged_path)

        # Upload to HuggingFace Hub
        logger.info("Uploading to HuggingFace Hub...")
        api.create_repo(new_repo_id, private=False, exist_ok=True)
        api.upload_folder(
            folder_path=merged_path,
            repo_id=new_repo_id,
            commit_message="Upload merged model via RunPod",
        )

        logger.info(f"Successfully uploaded to {new_repo_id}")
        return new_repo_id


def handler(job):
    """RunPod serverless handler function.

    When this function returns, the worker stops and billing ends.

    Args:
        job: RunPod job object containing input parameters

    Returns:
        dict: Result containing model_repo_id or error information
    """
    try:
        logger.info(f"Received job: {job}")
        job_input = job.get("input", job)
        logger.info(f"Job input: {job_input}")

        logger.info(
            f"Job started: {job_input['base_model_id']} + {job_input['lora_model_id']}"
        )

        model_repo_id = merge_and_upload_model(
            lora_model_id=job_input["lora_model_id"],
            base_model_id=job_input["base_model_id"],
            anonimize=job_input.get("anonimize", True),
            hf_token=job_input["hf_token"],
            hf_username=job_input["hf_username"],
        )

        return {"model_repo_id": model_repo_id, "status": "success"}

    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker (queue-based, scales to zero)...")
    runpod.serverless.start({"handler": handler})
