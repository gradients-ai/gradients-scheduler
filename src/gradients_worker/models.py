from datetime import datetime
from enum import Enum
from typing import List, Optional
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
    base_model_repository: Optional[str]
    trained_model_repository: Optional[str]
    ds_repo: Optional[str]
    field_input: Optional[str]
    field_system: Optional[str]
    field_output: Optional[str]
    field_instruction: Optional[str]
    format: Optional[str]
    no_input_format: Optional[str]
    system_format: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    created_at: str
    hours_to_complete: int


class EvaluationConfig(BaseModel):
    enabled: bool = False
    tasks: List[str] = []
    batch_size: int = 20
    device: str = "cuda:0"
    output_path: str = "evaluation_results"


class TaskRequest(BaseModel):
    model_repo: str
    ds_repo: str
    field_instruction: str
    field_input: Optional[str] = None
    field_output: Optional[str] = None
    field_system: Optional[str] = None
    format: Optional[str] = None
    hours_to_complete: int
    no_input_format: Optional[str] = None
    file_format: Optional[str] = "hf"


class TaskWithFixedDatasetsRequest(TaskRequest):
    ds_repo: str | None = Field(
        None, description="Optional: The original repository of the dataset"
    )
    training_data: str = Field(..., description="The prepared training dataset")
    synthetic_data: str = Field(..., description="The prepared synthetic dataset")
    test_data: str = Field(..., description="The prepared test dataset")


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
    INSTRUCTTEXTWITHFIXEDDATASETS = "InstructTextWithFixedDatasets"
    CHAT = "Chat"
