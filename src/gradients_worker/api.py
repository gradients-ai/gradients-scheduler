import logging

from aiohttp_retry import ExponentialRetry, RetryClient

from gradients_worker.config import settings
from gradients_worker.models import (
    MinimalTaskWithHotkeyDetails,
    NewTaskResponse,
    TaskRequest,
    TaskResultResponse,
    TaskStatusResponse,
    TaskWithFixedDatasetsRequest,
)

logger = logging.getLogger(__name__)

CREATE_TASK_ENDPOINT = "/v1/tasks/create"
TASKS_CREATE_ENDPOINT_CHAT = "/v1/tasks/create_chat"
TASKS_CREATE_WITH_FIXED_DATASETS_ENDPOINT = "/v1/tasks/create_with_fixed_datasets"
GET_TASK_STATUS_ENDPOINT = "/v1/tasks/{task_id}"
GET_TASK_RESULTS_ENDPOINT = "/v1/tasks/task_results/{task_id}"
GET_TASKS_RESULTS_ENDPOINT = "/v1/tasks/breakdown/{task_id}"
AUDITING_TASKS_BY_ID_ENDPOINT = "/auditing/tasks/{task_id}"


class GradientsAPI:
    def __init__(self):
        self.base_url = settings.GRADIENTS_API_URL
        self.headers = {"Authorization": f"Bearer {settings.GRADIENTS_API_KEY}"}

        self.post_retry_options = ExponentialRetry(
            attempts=3,
            start_timeout=2,
            max_timeout=10,
            factor=2,
            statuses={status for status in range(400, 600)},
            exceptions={Exception},
        )
        self.get_retry_options = ExponentialRetry(
            attempts=10,
            start_timeout=60,
            max_timeout=1800,
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
                f"{self.base_url}{CREATE_TASK_ENDPOINT}",
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


    async def create_chat_training_task(self, task_request: TaskRequest) -> NewTaskResponse:
        logger.info(
            f"Sending create chat training task request: {task_request.model_dump_json(indent=2)} \n headers: {self.headers}"
        )

        async with RetryClient(retry_options=self.post_retry_options) as session:
            async with session.post(
                f"{self.base_url}{TASKS_CREATE_ENDPOINT_CHAT}",
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
                f"{self.base_url}{TASKS_CREATE_WITH_FIXED_DATASETS_ENDPOINT}",
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

    async def get_task_status(self, task_id: str) -> TaskStatusResponse:
        logger.debug(f"Checking status for task: {task_id}")
        async with RetryClient(retry_options=self.get_retry_options) as session:
            async with session.get(
                f"{self.base_url}{GET_TASK_STATUS_ENDPOINT.format(task_id=task_id)}",
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return TaskStatusResponse.model_validate(await response.json())

    async def get_task_results(self, task_id: str):
        async with RetryClient(retry_options=self.get_retry_options) as session:
            async with session.get(
                f"{self.base_url}{GET_TASK_RESULTS_ENDPOINT.format(task_id=task_id)}",
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def get_miner_breakdown(self, task_id: str) -> TaskResultResponse:
        """Get the breakdown of miner results for a specific task."""
        logger.debug(f"Getting miner breakdown for task: {task_id}")
        async with RetryClient(retry_options=self.get_retry_options) as session:
            async with session.get(
                f"{self.base_url}{GET_TASKS_RESULTS_ENDPOINT.format(task_id=task_id)}",
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
                f"{self.base_url}{AUDITING_TASKS_BY_ID_ENDPOINT.format(task_id=task_id)}",
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return MinimalTaskWithHotkeyDetails.model_validate(
                    await response.json()
                )
