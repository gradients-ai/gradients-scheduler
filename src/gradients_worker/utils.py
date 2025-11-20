import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from logging import getLogger
from typing import Optional, Tuple

import datasets
import runpod
import torch
import yaml
from huggingface_hub import HfApi
from minio import Minio
from peft import PeftModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from gradients_worker import constants as cst
from gradients_worker.config import settings

logger = getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def log_memory_stats():
    """Log detailed memory statistics for debugging."""
    logger.info("===== MEMORY STATS =====")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2
            logger.info(
                f"GPU {i} Memory: Allocated: {allocated:.2f} MB, "
                f"Reserved: {reserved:.2f} MB, "
                f"Max Allocated: {max_allocated:.2f} MB"
            )
    else:
        logger.info("No CUDA devices available")


async def merge_and_upload_model(
    lora_model_id: str, base_model_id: str, anonimize: bool = True
) -> str:
    """Merge LoRA with base model and upload to HF hub.

    Args:
        lora_model_id: The HF ID of the LoRA adapter
        base_model_id: The HF ID of the base model

    Returns:
        str: The new model repo ID
    """
    logger.info(f"Merging LoRA {lora_model_id} with base model {base_model_id}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        if anonimize:
            new_repo_name = str(uuid.uuid4())
        else:
            new_repo_name = f"merged-{lora_model_id.split('/')[-1]}"

        new_repo_id = f"{settings.HF_USERNAME}/{new_repo_name}"

        api = HfApi()
        if api.repo_exists(new_repo_id):
            logger.info(f"Model {new_repo_id} already exists, skipping merge")
            return new_repo_id

        merged_path = os.path.join(tmp_dir, "merged_model")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        device_map = "cpu" if settings.USE_CPU_FOR_MODELS else "auto"
        logger.info(f"Loading models with device_map: {device_map}")

        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, device_map=device_map
            )
            model = PeftModel.from_pretrained(base_model, lora_model_id)
            merged_model = model.merge_and_unload()
        except Exception as e:
            logger.warning(
                f"Failed to merge LoRA: {e}. Downloading models directly instead."
            )
            merged_model = AutoModelForCausalLM.from_pretrained(
                lora_model_id, device_map=device_map
            )

        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        api.create_repo(new_repo_id, private=False)
        api.upload_folder(
            folder_path=merged_path,
            repo_id=new_repo_id,
            commit_message="Upload merged model",
        )

        logger.info(f"Successfully merged and uploaded model to {new_repo_id}")
        return new_repo_id


async def update_model_tokenizer(model_id: str, tokenizer_id: str) -> str:
    """Update a model with a new tokenizer and upload to HF hub.

    Args:
        model_id: The HF ID of the model to update
        tokenizer_id: The HF ID of the tokenizer to use

    Returns:
        str: The new model repo ID
    """
    model_parts = model_id.split("/")
    new_repo_name = f"{model_parts[0]}__{model_parts[1]}-with-tokenizer"
    new_repo_id = f"{settings.HF_USERNAME}/{new_repo_name}"

    api = HfApi()
    if api.repo_exists(new_repo_id):
        logger.info(f"Model {new_repo_id} already exists, skipping update")
        return new_repo_id

    logger.info(f"Updating model {model_id} with tokenizer from {tokenizer_id}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        device_map = "cpu" if settings.USE_CPU_FOR_MODELS else "auto"
        logger.info(f"Loading models with device_map: {device_map}")

        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        merged_path = os.path.join(tmp_dir, "updated_model")
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        api.create_repo(new_repo_id, private=False)
        api.upload_folder(
            folder_path=merged_path,
            repo_id=new_repo_id,
            commit_message=f"Update model with {tokenizer_id} tokenizer",
        )

        logger.info(f"Successfully updated model and uploaded to {new_repo_id}")
        return new_repo_id


def save_dataset_to_temp(
    dataset: datasets.Dataset, prefix: str = "dataset_"
) -> Tuple[str, int]:
    """Save a dataset to a temporary JSON file.

    Args:
        dataset: The dataset to save
        prefix: Prefix for the temporary file name

    Returns:
        Tuple[str, int]: Path to the temporary file and its size
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix=prefix)

    data = dataset.to_list()

    with open(temp_file.name, "w") as f:
        json.dump(data, f, cls=DateTimeEncoder)

    file_size = os.path.getsize(temp_file.name)
    return temp_file.name, file_size


def upload_to_minio(
    file_path: str,
    object_name: str,
    expires: int = cst.MINIO_EXPIRATION_DAYS * cst.SECONDS_PER_DAY,
) -> Optional[str]:
    """Upload a file to Minio and return a presigned URL.

    Args:
        file_path: Path to the file to upload
        object_name: Name to give the object in Minio
        expires: Expiration time in seconds for presigned URL

    Returns:
        str | None: Presigned URL of the uploaded file if successful, None otherwise
    """
    endpoint = os.environ["S3_COMPATIBLE_ENDPOINT"]
    access_key = os.environ["S3_COMPATIBLE_ACCESS_KEY"]
    secret_key = os.environ["S3_COMPATIBLE_SECRET_KEY"]
    bucket_name = os.environ["S3_BUCKET_NAME"]

    try:
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=True,
        )

        result = client.fput_object(bucket_name, object_name, file_path)

        if result:
            url = client.presigned_get_object(
                bucket_name, object_name, expires=timedelta(seconds=expires)
            )
            return url
        return None

    except Exception as e:
        logger.error(f"Error uploading to Minio: {e}")
        return None


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: The loaded configuration
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, config_path)

    logger.info(f"Loading configuration from {config_file_path}")
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    return config


async def merge_and_upload_model_subprocess(
    lora_model_id: str, base_model_id: str, anonimize: bool = True
) -> str:
    """Run merge_and_upload_model in a subprocess to avoid GPU memory leaks.

    Args:
        lora_model_id: The HF ID of the LoRA adapter
        base_model_id: The HF ID of the base model
        anonimize: Whether to anonymize the model name

    Returns:
        str: The new model repo ID
    """
    cmd = [
        "python",
        "src/gradients_worker/run_merge_and_upload_model.py",
        "--lora-model-id",
        lora_model_id,
        "--base-model-id",
        base_model_id,
        "--anonimize",
        str(anonimize),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        # Get the last non-empty line (our result)
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        return lines[-1] if lines else ""
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess failed: {e.stderr}")
        raise Exception(f"Failed to merge and upload model: {e.stderr}")


async def update_model_tokenizer_subprocess(model_id: str, tokenizer_id: str) -> str:
    """Run update_model_tokenizer in a subprocess to avoid GPU memory leaks.

    Args:
        model_id: The HF ID of the model to update
        tokenizer_id: The HF ID of the tokenizer to use

    Returns:
        str: The new model repo ID
    """
    cmd = [
        "python",
        "src/gradients_worker/run_update_model_tokenizer.py",
        "--model-id",
        model_id,
        "--tokenizer-id",
        tokenizer_id,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        # Get the last non-empty line (our result)
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        return lines[-1] if lines else ""
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess failed: {e.stderr}")
        raise Exception(f"Failed to update model tokenizer: {e.stderr}")


@retry(
    stop=stop_after_attempt(lambda: settings.RUNPOD_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def merge_and_upload_model_runpod(
    lora_model_id: str, base_model_id: str, anonimize: bool = True
) -> str:
    """Run merge_and_upload_model on RunPod serverless with automatic retry.

    Args:
        lora_model_id: The HF ID of the LoRA adapter
        base_model_id: The HF ID of the base model
        anonimize: Whether to anonymize the model name

    Returns:
        str: The new model repo ID

    Raises:
        Exception: If all retry attempts fail
    """
    if not settings.RUNPOD_API_KEY or not settings.RUNPOD_ENDPOINT_ID:
        raise Exception(
            "RunPod is not configured. Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID"
        )

    logger.info(f"Starting RunPod merge: {base_model_id} + {lora_model_id}")

    # Set RunPod API key and initialize endpoint
    runpod.api_key = settings.RUNPOD_API_KEY
    endpoint = runpod.Endpoint(settings.RUNPOD_ENDPOINT_ID)
    job = endpoint.run(
        {
            "input": {
                "lora_model_id": lora_model_id,
                "base_model_id": base_model_id,
                "anonimize": anonimize,
                "hf_token": settings.HF_TOKEN,
                "hf_username": settings.HF_USERNAME,
            }
        }
    )

    logger.info(f"RunPod job submitted: {job.job_id}")

    # Wait for completion with timeout
    start_time = time.time()
    while job.status() not in [cst.RUNPOD_STATUS_COMPLETED, cst.RUNPOD_STATUS_FAILED]:
        if time.time() - start_time > settings.RUNPOD_TIMEOUT:
            raise TimeoutError(f"RunPod job timed out after {settings.RUNPOD_TIMEOUT}s")

        logger.info(
            f"RunPod job status: {job.status()} (elapsed: {time.time() - start_time:.0f}s)"
        )
        time.sleep(settings.RUNPOD_POLL_INTERVAL)

    # Get result
    output = job.output()
    if job.status() == cst.RUNPOD_STATUS_COMPLETED and output.get("status") == "success":
        model_repo_id = output["model_repo_id"]
        logger.info(f"RunPod merge successful: {model_repo_id}")
        return model_repo_id

    # Job failed
    error = output.get("error", "Unknown error")
    raise Exception(f"RunPod job failed: {error}")
