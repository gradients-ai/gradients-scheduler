import logging

from aiohttp_retry import ExponentialRetry, RetryClient

from gradients_worker import constants as cst
from gradients_worker.config import settings
from gradients_worker.models import (
    MinimalTaskWithHotkeyDetails,
    NewTaskResponse,
    TaskRequest,
    TaskRequestChatWithCustomDataset,
    TaskResultResponse,
    TaskStatusResponse,
    TaskType,
    TaskWithFixedDatasetsRequest,
)

logger = logging.getLogger(__name__)


class GradientsAPI:
    def __init__(self):
        self.base_url = settings.GRADIENTS_API_URL
        self.headers = {"Authorization": f"Bearer {settings.GRADIENTS_API_KEY}"}

        self.post_retry_options = ExponentialRetry(
            attempts=cst.RETRY_ATTEMPTS_POST,
            start_timeout=cst.RETRY_START_TIMEOUT_POST,
            max_timeout=cst.RETRY_MAX_TIMEOUT_POST,
            factor=2,
            statuses={status for status in range(400, 600)},
            exceptions={Exception},
        )
        self.get_retry_options = ExponentialRetry(
            attempts=cst.RETRY_ATTEMPTS_GET,
            start_timeout=cst.RETRY_START_TIMEOUT_GET,
            max_timeout=cst.RETRY_MAX_TIMEOUT_GET,
            factor=2,
            statuses={status for status in range(500, 600)},
            exceptions={Exception},
        )

        logger.info(
            f"Gradients API initialized with base URL: {self.base_url} and headers: {self.headers}"
        )

    async def create_training_task(self, task_request: TaskRequest) -> NewTaskResponse:
        logger.info(
            f"Sending create training task request: {task_request.model_dump_json(indent=2)} \n headers: {self.headers}"
        )

        async with RetryClient(retry_options=self.post_retry_options) as session:
            async with session.post(
                f"{self.base_url}{cst.CREATE_TASK_ENDPOINT}",
                headers=self.headers,
                json=task_request.model_dump(),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to create training task: {response.status} {error_text}"
                    )
                    response.raise_for_status()
                return NewTaskResponse.model_validate(await response.json())

    async def create_chat_training_task(
        self, task_request: TaskRequest
    ) -> NewTaskResponse:
        logger.info(
            f"Sending create chat training task request: {task_request.model_dump_json(indent=2)} \n headers: {self.headers}"
        )

        async with RetryClient(retry_options=self.post_retry_options) as session:
            async with session.post(
                f"{self.base_url}{cst.TASKS_CREATE_ENDPOINT_CHAT}",
                headers=self.headers,
                json=task_request.model_dump(),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to create training task: {response.status} {error_text}"
                    )
                    response.raise_for_status()
                return NewTaskResponse.model_validate(await response.json())

    async def create_training_task_with_fixed_datasets(
        self, task_request: TaskWithFixedDatasetsRequest
    ) -> NewTaskResponse:
        logger.info(
            f"Sending create training task with fixed datasets request: {task_request.model_dump_json(indent=2)} \n headers: {self.headers}"
        )

        async with RetryClient(retry_options=self.post_retry_options) as session:
            async with session.post(
                f"{self.base_url}{cst.TASKS_CREATE_WITH_FIXED_DATASETS_ENDPOINT}",
                headers=self.headers,
                json=task_request.model_dump(),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to create training task with fixed datasets: {response.status} {error_text}"
                    )
                    response.raise_for_status()
                return NewTaskResponse.model_validate(await response.json())

    async def create_chat_training_task_with_custom_dataset(
        self, task_request: TaskRequestChatWithCustomDataset
    ) -> NewTaskResponse:
        """Create a chat training task with pre-split train/test datasets."""
        logger.info(
            f"Sending create custom dataset chat request: {task_request.model_dump_json(indent=2)} \n headers: {self.headers}"
        )

        async with RetryClient(retry_options=self.post_retry_options) as session:
            async with session.post(
                f"{self.base_url}{cst.TASKS_CREATE_CUSTOM_DATASET_CHAT_ENDPOINT}",
                headers=self.headers,
                json=task_request.model_dump(),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to create custom dataset chat task: {response.status} {error_text}"
                    )
                    response.raise_for_status()
                return NewTaskResponse.model_validate(await response.json())

    async def get_task_status(self, task_id: str) -> TaskStatusResponse:
        logger.debug(f"Checking status for task: {task_id}")
        async with RetryClient(retry_options=self.get_retry_options) as session:
            async with session.get(
                f"{self.base_url}{cst.GET_TASK_STATUS_ENDPOINT.format(task_id=task_id)}",
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return TaskStatusResponse.model_validate(await response.json())

    async def get_task_results(self, task_id: str):
        async with RetryClient(retry_options=self.get_retry_options) as session:
            async with session.get(
                f"{self.base_url}{cst.GET_TASK_RESULTS_ENDPOINT.format(task_id=task_id)}",
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def get_miner_breakdown(self, task_id: str) -> TaskResultResponse:
        """Get the breakdown of miner results for a specific task."""
        logger.debug(f"Getting miner breakdown for task: {task_id}")
        async with RetryClient(retry_options=self.get_retry_options) as session:
            async with session.get(
                f"{self.base_url}{cst.GET_TASKS_RESULTS_ENDPOINT.format(task_id=task_id)}",
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return TaskResultResponse.model_validate(await response.json())

    async def get_task_hotkey_details(
        self, task_id: str
    ) -> MinimalTaskWithHotkeyDetails:
        """Get the hotkey details for a specific task using the auditing endpoint."""
        logger.debug(f"Getting task hotkey details for task: {task_id}")
        async with RetryClient(retry_options=self.get_retry_options) as session:
            async with session.get(
                f"{self.base_url}{cst.AUDITING_TASKS_BY_ID_ENDPOINT.format(task_id=task_id)}",
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return MinimalTaskWithHotkeyDetails.model_validate(
                    await response.json()
                )

    async def create_training_task_by_type(
        self,
        task_type: TaskType,
        task_request: TaskRequest
        | TaskWithFixedDatasetsRequest
        | TaskRequestChatWithCustomDataset,
    ) -> NewTaskResponse:
        """Create a training task by sending a request to the appropriate API endpoint based on task type."""

        if task_type == TaskType.INSTRUCTTEXTWITHFIXEDDATASETS:
            return await self.create_training_task_with_fixed_datasets(task_request)
        elif task_type == TaskType.INSTRUCTTEXT:
            return await self.create_training_task(task_request)
        elif task_type == TaskType.CHAT:
            return await self.create_chat_training_task(task_request)
        elif task_type == TaskType.CUSTOMDATASETCHAT:
            return await self.create_chat_training_task_with_custom_dataset(
                task_request
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
