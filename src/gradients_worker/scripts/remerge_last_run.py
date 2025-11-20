#!/usr/bin/env python3
"""
Script to re-merge the last trained model for a task.

This script:
1. Finds the wandb run with the highest training_number for a given task
2. Submits a RunPod job to merge model_repo (LoRA) with previously_finetuned_model_repo (base)
3. Logs a new wandb run with updated merged_model_repo field

Usage:
    python -m gradients_worker.scripts.remerge_last_run
"""

from logging import getLogger

import runpod

import wandb
from gradients_worker import constants as cst
from gradients_worker.config import settings
from gradients_worker.utils import load_config

logger = getLogger(__name__)


def remerge_task(task_name: str):
    """Re-merge the last trained model for a given task."""

    # 1. Load task configuration
    config = load_config(cst.CONFIG_FILENAME)
    task_config = config.get(task_name)
    wandb_project = task_config.get(cst.KEY_WANDB_PROJECT)

    logger.info(f"Task: {task_name}")
    logger.info(f"W&B Project: {wandb_project}")

    # 2. Find the run with highest training_number
    api = wandb.Api()
    runs = api.runs(f"{settings.WANDB_ENTITY}/{wandb_project}")

    # Find run with max training_number
    max_training_number = -1
    target_run = None

    for run in runs:
        if not hasattr(run, "config") or cst.KEY_TRAINING_NUMBER not in run.config:
            continue

        training_number = run.config.get(cst.KEY_TRAINING_NUMBER, 0)
        if training_number > max_training_number:
            max_training_number = training_number
            target_run = run

    logger.info(f"Found run with highest training_number: {max_training_number}")
    logger.info(f"Run name: {target_run.name}")
    logger.info(f"Run ID: {target_run.id}")

    # 3. Extract model repositories
    model_repo = target_run.config.get(cst.KEY_MODEL_REPO)  # LoRA adapter
    previously_finetuned_model_repo = target_run.config.get(
        cst.KEY_PREVIOUSLY_FINETUNED_MODEL_REPO
    )

    logger.info(f"LoRA model (model_repo): {model_repo}")
    logger.info(
        f"Base model (previously_finetuned_model_repo): {previously_finetuned_model_repo}"
    )

    # 4. Submit RunPod merge job (without waiting for completion)
    logger.info("Submitting RunPod merge job...")

    # Submit job using RunPod API
    runpod.api_key = settings.RUNPOD_API_KEY
    endpoint = runpod.Endpoint(settings.RUNPOD_ENDPOINT_ID)

    job = endpoint.run(
        {
            "input": {
                "lora_model_id": model_repo,
                "base_model_id": previously_finetuned_model_repo,
                "anonimize": False,
                "hf_token": settings.HF_TOKEN,
                "hf_username": settings.HF_USERNAME,
            }
        }
    )
    logger.info(f"RunPod job submitted successfully: {job.job_id}")

    # 5. Construct expected merged model repo name
    # Based on merge_and_upload_model logic when anonimize=False:
    # new_repo_name = f"merged-{lora_model_id.split('/')[-1]}"
    lora_model_name = model_repo.split("/")[-1]
    expected_merged_repo = f"{settings.HF_USERNAME}/merged-{lora_model_name}"

    logger.info(f"Expected merged model repo: {expected_merged_repo}")

    # 6. Create new W&B run with updated merged_model_repo
    # Copy all config from original run
    new_config = dict(target_run.config)
    new_config[cst.KEY_MERGED_MODEL_REPO] = expected_merged_repo

    logger.info("Creating new W&B run with updated merged_model_repo...")

    new_run = wandb.init(
        entity=settings.WANDB_ENTITY,
        project=wandb_project,
        name=f"{target_run.name}_remerged",
        config=new_config,
    )

    # Copy metrics from original run if available
    if hasattr(target_run, "summary"):
        metrics = {}
        for key in [cst.KEY_TEST_LOSS, cst.KEY_SYNTH_LOSS, cst.KEY_WEIGHTED_LOSS, cst.KEY_SCORE]:
            if key in target_run.summary:
                metrics[key] = target_run.summary[key]

        if metrics:
            wandb.log(metrics)
            logger.info(f"Logged metrics: {metrics}")

    new_run_id = new_run.id
    wandb.finish()

    logger.info(f"New W&B run created: {new_run_id}")
    logger.info(f"Successfully triggered re-merge for task '{task_name}'")
    logger.info(f"RunPod job ID: {job.job_id}")
    logger.info(f"Expected merged repo: {expected_merged_repo}")

    print(f"RunPod Job ID: {job.job_id}")
    print(f"Expected Merged Repo: {expected_merged_repo}")
    print(f"W&B Run ID: {new_run_id}")


if __name__ == "__main__":
    # task_name = "templar-chat"
    task_name = "qwen-32b-chat"
    remerge_task(task_name)
