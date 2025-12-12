from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    PREPARING_DATA = "preparing_data"
    IDLE = "idle"
    READY = "ready"
    SUCCESS = "success"
    LOOKING_FOR_NODES = "looking_for_nodes"
    DELAYED = "delayed"
    EVALUATING = "evaluating"
    PREEVALUATION = "preevaluation"
    TRAINING = "training"
    FAILURE = "failure"
    FAILURE_FINDING_NODES = "failure_finding_nodes"
    PREP_TASK_FAILURE = "prep_task_failure"
    NODE_TRAINING_FAILURE = "node_training_failure"

    def is_failure(self):
        return self in [
            TaskStatus.FAILURE,
            TaskStatus.FAILURE_FINDING_NODES,
            TaskStatus.NODE_TRAINING_FAILURE,
            TaskStatus.PREP_TASK_FAILURE,
        ]


class WinningSubmission(BaseModel):
    hotkey: str
    score: float
    model_repo: str


class TaskStatusResponse(BaseModel):
    id: UUID
    account_id: UUID
    status: TaskStatus
    base_model_repository: str | None = None
    trained_model_repository: str | None = None
    ds_repo: str | None = None
    field_input: str | None = None
    field_system: str | None = None
    field_output: str | None = None
    field_instruction: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    system_format: str | None = None
    chat_template: str | None = None
    chat_column: str | None = None
    chat_role_field: str | None = None
    chat_content_field: str | None = None
    chat_user_reference: str | None = None
    chat_assistant_reference: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str
    hours_to_complete: int
    task_type: str | None = None
    result_model_name: str | None = None


class EvaluationConfig(BaseModel):
    enabled: bool = False
    tasks: list[str] = []
    batch_size: int = 20
    device: str = "cuda:0"
    output_path: str = "evaluation_results"


class TaskRequest(BaseModel):
    model_repo: str
    ds_repo: str
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None
    field_system: str | None = None
    format: str | None = None
    hours_to_complete: int
    no_input_format: str | None = None
    file_format: str | None = "hf"


class TaskRequestChat(BaseModel):
    model_repo: str = Field(
        ...,
        description="The repository for the model",
        examples=["Qwen/Qwen2.5-Coder-32B-Instruct"],
    )
    hours_to_complete: int
    chat_template: str = Field(
        ..., description="The chat template of the dataset", examples=["chatml"]
    )
    chat_column: str | None = Field(
        None,
        description="The column name containing the conversations",
        examples=["conversations"],
    )
    chat_role_field: str | None = Field(
        None, description="The column name for the role", examples=["role"]
    )
    chat_content_field: str | None = Field(
        None, description="The column name for the content", examples=["content"]
    )
    chat_user_reference: str | None = Field(
        None, description="The user reference", examples=["user"]
    )
    chat_assistant_reference: str | None = Field(
        None, description="The assistant reference", examples=["assistant"]
    )

    ds_repo: str = Field(
        ...,
        description="The repository for the dataset",
        examples=["Magpie-Align/Magpie-Pro-300K-Filtered"],
    )
    file_format: str | None = "hf"


class TaskRequestChatWithCustomDataset(TaskRequestChat):
    """Request model for create_custom_dataset_chat endpoint."""

    ds_repo: str | None = Field(
        None, description="Optional: The original repository of the dataset"
    )
    file_format: str | None = "s3"
    training_data: str = Field(..., description="URL to the prepared training dataset")
    test_data: str = Field(..., description="URL to the prepared test dataset")


class NewTaskResponse(BaseModel):
    success: bool
    task_id: UUID | None
    created_at: datetime | None
    account_id: UUID | None


class MinerTaskResult(BaseModel):
    hotkey: str
    quality_score: float
    test_loss: float | None
    synth_loss: float | None
    score_reason: str | None = ""


class TaskResultResponse(BaseModel):
    id: UUID
    miner_results: list[MinerTaskResult] | None


class HotkeyDetails(BaseModel):
    hotkey: str
    submission_id: UUID | None = None
    quality_score: float | None = None
    test_loss: float | None = None
    synth_loss: float | None = None
    repo: str | None = None
    rank: int | None = None
    score_reason: str | None = None
    offer_response: dict | None = None


class MinimalTaskWithHotkeyDetails(BaseModel):
    hotkey_details: list[HotkeyDetails]


class TaskType(str, Enum):
    INSTRUCTTEXT = "InstructText"
    CHAT = "Chat"
    CUSTOMDATASETCHAT = "CustomDatasetChat"
