import asyncio
import logging
import random
from datetime import datetime, timedelta

import wandb

from . import constants as cst
from .api import GradientsAPI
from .config import settings
from .dataset_scheduler import DatasetsScheduler
from .models import (
    HotkeyDetails,
    TaskRequest,
    TaskRequestChat,
    TaskRequestChatWithCustomDataset,
    TaskStatus,
    TaskStatusResponse,
    TaskType,
)
from .utils import (
    load_config,
    merge_and_upload_model_runpod,
    merge_and_upload_model_subprocess,
    update_model_tokenizer_subprocess,
)

logger = logging.getLogger(__name__)


class GradientsTrainingScheduler:
    def __init__(self, task_name: str):
        """Initialize training scheduler for a specific task name.

        Args:
            task_name: The task name from config.yaml
        """
        self.task_name = task_name
        self.config = load_config(cst.CONFIG_FILENAME)
        self.task_config = self.config.get(task_name, {})
        self.api = GradientsAPI()
        self.wandb_project = self.task_config.get(cst.KEY_WANDB_PROJECT)
        self.tokenizer_merged = False
        self.training_number = 0
        self.last_merged_model = self.task_config[cst.KEY_MODEL_REPO]
        self.last_successful_run = None
        task_type_str = self.task_config.get(cst.KEY_TASK_TYPE, "InstructText")
        self.task_type = TaskType(task_type_str)

        self.task_request = None

        # Initialize the dataset scheduler
        self.dataset_scheduler = DatasetsScheduler(self.task_config)

        logger.info(
            f"Initialized GradientsTrainingScheduler for {task_name} with task_type: {self.task_type.value}"
        )

    def generate_task_request(
        self,
    ) -> TaskRequest | TaskRequestChat | TaskRequestChatWithCustomDataset:
        """Generate task request from task_config.

        Sets self.task_request and self.training_number.

        Returns:
            TaskRequest | TaskRequestChat | TaskRequestChatWithCustomDataset: The generated task request
        """
        # Will prepare if not already done
        if not self.dataset_scheduler.prepare_datasets():
            raise RuntimeError("Failed to prepare datasets")

        # Determine which chunk to use based on training number
        chunk_index = self.training_number % self.dataset_scheduler.num_chunks
        logger.info(
            f"Using chunk idx {chunk_index} ({self.dataset_scheduler.num_chunks} chunks) for training {self.training_number}"
        )

        if self.task_type == TaskType.INSTRUCTTEXT:
            dataset_url = self.dataset_scheduler.prepare_and_upload_whole_chunk(
                chunk_index
            )

            task_request = TaskRequest(
                model_repo=self.last_merged_model,
                ds_repo=dataset_url,
                field_instruction=cst.DEFAULT_INSTRUCTION_FIELD,
                field_input=cst.DEFAULT_INPUT_FIELD,
                field_output=cst.DEFAULT_OUTPUT_FIELD,
                hours_to_complete=self.task_config.get(
                    cst.KEY_HOURS_TO_COMPLETE, cst.DEFAULT_HOURS_TO_COMPLETE
                ),
                file_format=cst.FILE_FORMAT_S3,
            )
        elif self.task_type == TaskType.CHAT:
            dataset_url = self.dataset_scheduler.prepare_and_upload_whole_chunk(
                chunk_index
            )

            task_request = TaskRequestChat(
                model_repo=self.last_merged_model,
                ds_repo=dataset_url,
                chat_template=self.task_config.get(
                    cst.KEY_CHAT_TEMPLATE, cst.DEFAULT_CHAT_TEMPLATE
                ),
                chat_column=cst.DEFAULT_CHAT_COLUMN,
                chat_role_field=cst.DEFAULT_CHAT_ROLE_FIELD,
                chat_content_field=cst.DEFAULT_CHAT_CONTENT_FIELD,
                chat_user_reference=cst.DEFAULT_CHAT_USER_REFERENCE,
                chat_assistant_reference=cst.DEFAULT_CHAT_ASSISTANT_REFERENCE,
                hours_to_complete=self.task_config.get(
                    cst.KEY_HOURS_TO_COMPLETE, cst.DEFAULT_HOURS_TO_COMPLETE
                ),
                file_format=cst.FILE_FORMAT_S3,
            )

        elif self.task_type == TaskType.CUSTOMDATASETCHAT:
            train_url, test_url = (
                self.dataset_scheduler.prepare_and_upload_chunk_train_test(chunk_index)
            )

            task_request = TaskRequestChatWithCustomDataset(
                model_repo=self.last_merged_model,
                chat_template=self.task_config.get(
                    cst.KEY_CHAT_TEMPLATE, cst.DEFAULT_CHAT_TEMPLATE
                ),
                chat_column=cst.DEFAULT_CHAT_COLUMN,
                chat_role_field=cst.DEFAULT_CHAT_ROLE_FIELD,
                chat_content_field=cst.DEFAULT_CHAT_CONTENT_FIELD,
                chat_user_reference=cst.DEFAULT_CHAT_USER_REFERENCE,
                chat_assistant_reference=cst.DEFAULT_CHAT_ASSISTANT_REFERENCE,
                hours_to_complete=self.task_config.get(
                    cst.KEY_HOURS_TO_COMPLETE, cst.DEFAULT_HOURS_TO_COMPLETE
                ),
                file_format=cst.FILE_FORMAT_S3,
                training_data=train_url,
                test_data=test_url,
            )

        return task_request

    def update_training_state_from_wandb(self) -> None:
        """Update training state (training number and merged model) from wandb.

        Updates self.training_number and self.last_merged_model based on the latest
        successful run found in wandb.
        """
        api = wandb.Api()

        try:
            runs = api.runs(f"{settings.WANDB_ENTITY}/{self.wandb_project}")
            logger.info(f"Found {len(runs)} runs in W&B")
        except Exception as e:
            error_message = str(e)
            if "Could not find project" in error_message:
                logger.info(f"No project found: {error_message}")
                return
            # Re-raise other errors
            logger.error(f"Error getting runs from W&B: {e}")
            raise

        last_number = -1
        last_merged_model = None

        for run in runs:
            if not hasattr(run, "config") or cst.KEY_TRAINING_NUMBER not in run.config:
                continue

            training_number = run.config.get(cst.KEY_TRAINING_NUMBER, 0)
            if training_number > last_number:
                last_number = training_number
                last_merged_model = run.config.get(cst.KEY_MERGED_MODEL_REPO)

        logger.info(f"Last training number found: {last_number}")
        if last_merged_model:
            logger.info(f"Last merged model: {last_merged_model}")
        else:
            logger.info("No merged model found for last training")

        logger.info(
            f"Updating training number from {self.training_number} to {last_number + 1}"
        )
        self.training_number = last_number + 1

        logger.info(
            f"Updating last merged model from {self.last_merged_model} to {last_merged_model}"
        )
        self.last_merged_model = last_merged_model

    def calculate_next_run_time(self) -> datetime:
        """Calculate the next run time based on last successful run and config.

        Returns:
            datetime: The next time to run training
        """
        if not self.last_successful_run:
            logger.info("No previous successful run, scheduling immediate run")
            return datetime.now()

        run_intervals = self.task_config.get(cst.KEY_RUN_INTERVALS, {})

        min_days = run_intervals.get(cst.KEY_MIN_DAYS, cst.DEFAULT_RUN_INTERVAL)
        max_days = run_intervals.get(cst.KEY_MAX_DAYS, cst.DEFAULT_RUN_INTERVAL)
        min_hours = run_intervals.get(cst.KEY_MIN_HOURS, cst.DEFAULT_RUN_INTERVAL)
        max_hours = run_intervals.get(cst.KEY_MAX_HOURS, cst.DEFAULT_RUN_INTERVAL)

        # Calculate random interval between min and max
        days_delta = random.randint(min_days, max(min_days, max_days))
        hours_delta = random.randint(min_hours, max(min_hours, max_hours))

        total_seconds = (
            days_delta * cst.SECONDS_PER_DAY + hours_delta * cst.SECONDS_PER_HOUR
        )
        next_run = self.last_successful_run + timedelta(seconds=total_seconds)

        logger.info(f"Next run scheduled for: {next_run}")
        return next_run

    async def update_model_tokenizer(self) -> None:
        """Update model tokenizer if specified in config."""
        if cst.KEY_TOKENIZER_ID in self.task_config and not self.tokenizer_merged:
            logger.info(f"Updating tokenizer for model {self.last_merged_model}")

            updated_model_repo = await update_model_tokenizer_subprocess(
                model_id=self.last_merged_model,
                tokenizer_id=self.task_config[cst.KEY_TOKENIZER_ID],
            )

            self.last_merged_model = updated_model_repo
            self.tokenizer_merged = True
            logger.info(f"Updated model repo to {updated_model_repo}")

    def _get_winning_miner_result(self, results: list[HotkeyDetails]) -> HotkeyDetails:
        """Select best hotkey detail based on lowest test loss and valid repo."""
        valid = [
            r
            for r in results
            if (
                r.quality_score is not None
                and r.test_loss is not None
                and r.test_loss != 0
            )
        ]
        if not valid:
            raise ValueError("No hotkey details with valid test loss and repo found")
        return min(valid, key=lambda x: x.test_loss)

    def _log_training_metrics(
        self,
        miner_result: HotkeyDetails,
        response: TaskStatusResponse,
        merged_model: str,
    ) -> str:
        """Log training results as separate runs."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        config = {
            cst.KEY_MERGED_MODEL_REPO: merged_model,
            cst.KEY_MODEL_REPO: miner_result.repo,
            cst.KEY_PREVIOUSLY_FINETUNED_MODEL_REPO: response.base_model_repository,
            cst.KEY_ORIGINAL_MODEL_REPO: self.task_config[cst.KEY_MODEL_REPO],
            "timestamp": timestamp,
            "task_id": response.id,
            cst.KEY_WINNING_MINER: miner_result.hotkey,
            cst.KEY_TRAINING_NUMBER: self.training_number,
        }

        run = wandb.init(
            entity=settings.WANDB_ENTITY,
            project=self.wandb_project,
            name=f"{miner_result.repo.split('/')[-1]}_{response.id}",
            config=config,
        )

        wandb.log(
            {
                cst.KEY_TEST_LOSS: miner_result.test_loss,
                cst.KEY_SYNTH_LOSS: miner_result.synth_loss,
                "weighted_loss": (miner_result.test_loss + miner_result.synth_loss) / 2,
                "score": miner_result.quality_score,
            }
        )

        run_id = run.id
        wandb.finish()

        return run_id

    async def pre_training_setup(self) -> bool:
        """Perform all setup needed before training.

        This includes:
        - Checking if enough time has passed since last run
        - Preparing model repository (finding and merging best model)
        - Updating model tokenizer if needed
        - Getting last training number
        - Generating task request

        Returns:
            bool: True if setup was successful and training should proceed
        """
        logger.info("Starting pre-training setup")

        try:
            # Check if it's time to run
            next_run_time = self.calculate_next_run_time()

            if next_run_time > datetime.now():
                sleep_seconds = (next_run_time - datetime.now()).total_seconds()
                logger.info(
                    f"Not time to run yet. Next run at {next_run_time} (in {sleep_seconds / 3600:.2f} hours)"
                )
                return False

            await self.update_model_tokenizer()
            self.update_training_state_from_wandb()

            logger.info(f"Setting training number to {self.training_number}")

            # Generate task request
            task_request = self.generate_task_request()
            logger.info(f"Changed task request to {task_request}")
            self.task_request = task_request

            logger.info("Pre-training setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during pre-training setup: {e}", exc_info=True)
            return False

    async def train(self) -> str | None:
        """Run the training process.

        Returns:
            str | None: Trained model repository URL if successful, None otherwise
        """
        logger.info("Starting training process")

        setup_success = await self.pre_training_setup()
        if not setup_success:
            logger.info(
                "Pre-training setup did not complete successfully, skipping training"
            )
            return None

        logger.info("Creating training task")

        try:
            task = await self.api.create_training_task_by_type(
                self.task_type, self.task_request
            )

            task_id = task.task_id
            logger.info(f"Training task created with ID: {task_id}")

            while True:
                response = await self.api.get_task_status(task_id)
                logger.info(f"Task status: {response.status}")

                if response.status == TaskStatus.SUCCESS:
                    logger.info("Training completed successfully!")
                    try:
                        details = await self.api.get_task_hotkey_details(task_id)
                        if not details.hotkey_details:
                            raise ValueError("No hotkey details found")

                        best_hotkey = self._get_winning_miner_result(
                            details.hotkey_details
                        )
                        if not best_hotkey or not best_hotkey.repo:
                            raise ValueError("No valid repo found in hotkey details")
                        trained_model_repository = best_hotkey.repo
                        logger.info(f"Chosen model repo: {trained_model_repository}")

                        try:
                            if settings.USE_RUNPOD_FOR_MERGE:
                                logger.info("Using RunPod for merge")
                                merged_model = await merge_and_upload_model_runpod(
                                    lora_model_id=trained_model_repository,
                                    base_model_id=self.task_request.model_repo,
                                )
                            else:
                                logger.info("Using local subprocess for merge")
                                merged_model = await merge_and_upload_model_subprocess(
                                    lora_model_id=trained_model_repository,
                                    base_model_id=self.task_request.model_repo,
                                )
                        except Exception as e:
                            logger.error(f"Error merging and uploading model: {e}")
                            merged_model = trained_model_repository

                        self._log_training_metrics(
                            miner_result=best_hotkey,
                            response=response,
                            merged_model=merged_model,
                        )

                        logger.info(
                            f"Logged metrics from best miner: {best_hotkey.hotkey}"
                        )
                        self.last_successful_run = datetime.now()
                        return trained_model_repository

                    except Exception as e:
                        logger.error(f"Failed to process hotkey details: {str(e)}")
                        return None

                elif response.status.is_failure():
                    error_msg = f"Training task failed: {response.status}"
                    logger.error(error_msg)
                    return None

                logger.info(
                    f"Waiting for {settings.CHECK_INTERVAL} seconds before checking again"
                )
                await asyncio.sleep(settings.CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            return None

    async def run_forever(self):
        """Run training in an infinite loop with appropriate scheduling."""
        logger.info(f"Starting training scheduler for {self.task_name}")

        while True:
            try:
                result = await self.train()
                if result:
                    logger.info(
                        f"Training completed successfully. Model repo: {result}"
                    )
                else:
                    logger.warning("Training did not produce results")
                await asyncio.sleep(cst.SLEEP_INTERVAL_SECONDS)

            except Exception as e:
                logger.error(f"Error in training loop: {e}", exc_info=True)
                await asyncio.sleep(cst.SLEEP_INTERVAL_SECONDS)
