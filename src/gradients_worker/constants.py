"""Constants used throughout the gradients worker."""

# =============================================================================
# Chat Dataset Standardization Defaults
# =============================================================================
DEFAULT_CHAT_COLUMN = "conversations"
DEFAULT_CHAT_ROLE_FIELD = "role"
DEFAULT_CHAT_CONTENT_FIELD = "content"
DEFAULT_CHAT_USER_REFERENCE = "user"
DEFAULT_CHAT_ASSISTANT_REFERENCE = "assistant"
DEFAULT_CHAT_TEMPLATE = "chatml"

# =============================================================================
# Standard Dataset Field Names
# =============================================================================
DEFAULT_INSTRUCTION_FIELD = "instruction"
DEFAULT_INPUT_FIELD = "input"
DEFAULT_OUTPUT_FIELD = "output"
TRAIN_SPLIT = "train"
SOURCE_INDEX_COLUMN = "_source_idx"

# =============================================================================
# Configuration Keys - Core Task Config
# =============================================================================
KEY_ENABLED = "enabled"
KEY_MODEL_REPO = "model_repo"
KEY_WANDB_PROJECT = "wandb_project"
KEY_TASK_TYPE = "task_type"
KEY_HOURS_TO_COMPLETE = "hours_to_complete"
KEY_TOKENIZER_ID = "tokenizer_id"
KEY_CHAT_TEMPLATE = "chat_template"

# =============================================================================
# Configuration Keys - Dataset Config
# =============================================================================
KEY_DATASETS = "datasets"
KEY_NAME = "name"
KEY_MAX_ROWS = "max_rows"
KEY_FIELD_INSTRUCTION = "field_instruction"
KEY_FIELD_INPUT = "field_input"
KEY_FIELD_OUTPUT = "field_output"
KEY_RANDOM_SEED = "random_seed"
KEY_FINAL_TEST_SIZE = "final_test_size"
KEY_SAMPLES_PER_TRAINING = "samples_per_training"
KEY_PER_CHUNK_TEST_PROPORTION = "per_chunk_test_proportion"

# =============================================================================
# Configuration Keys - Chat Dataset Config
# =============================================================================
KEY_CHAT_COLUMN = "chat_column"
KEY_CHAT_ROLE_FIELD = "chat_role_field"
KEY_CHAT_CONTENT_FIELD = "chat_content_field"
KEY_CHAT_USER_REFERENCE = "chat_user_reference"
KEY_CHAT_ASSISTANT_REFERENCE = "chat_assistant_reference"

# =============================================================================
# Configuration Keys - Run Intervals
# =============================================================================
KEY_RUN_INTERVALS = "run_intervals"
KEY_MIN_DAYS = "min_days"
KEY_MAX_DAYS = "max_days"
KEY_MIN_HOURS = "min_hours"
KEY_MAX_HOURS = "max_hours"

# =============================================================================
# Configuration Keys - WandB Metadata
# =============================================================================
KEY_TRAINING_NUMBER = "training_number"
KEY_MERGED_MODEL_REPO = "merged_model_repo"
KEY_PREVIOUSLY_FINETUNED_MODEL_REPO = "previously_finetuned_model_repo"
KEY_ORIGINAL_MODEL_REPO = "original_model_repo"
KEY_WINNING_MINER = "winning_miner"

# =============================================================================
# Configuration Keys - Metrics
# =============================================================================
KEY_TEST_LOSS = "test_loss"
KEY_SYNTH_LOSS = "synth_loss"
KEY_WEIGHTED_LOSS = "weighted_loss"
KEY_SCORE = "score"

# =============================================================================
# Magic Numbers - Dataset Defaults
# =============================================================================
DEFAULT_RANDOM_SEED = 42
DEFAULT_FINAL_TEST_SIZE = 0.05
DEFAULT_SAMPLES_PER_TRAINING = 150_000
DEFAULT_PER_CHUNK_TEST_PROPORTION = 0.02

# =============================================================================
# Magic Numbers - Dataset Split Ratios
# =============================================================================
TRAIN_SPLIT_RATIO = 0.90
TEST_SPLIT_RATIO = 0.1
# Synth split is the remainder: 1 - TRAIN_SPLIT_RATIO - TEST_SPLIT_RATIO = 0.0

# =============================================================================
# Magic Numbers - Time Constants (in seconds)
# =============================================================================
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
MINIO_EXPIRATION_DAYS = 7
CHECK_INTERVAL_SECONDS = 600
SLEEP_INTERVAL_SECONDS = 60

# =============================================================================
# Magic Numbers - Training Defaults
# =============================================================================
DEFAULT_HOURS_TO_COMPLETE = 8
INITIAL_TRAINING_NUMBER = -1
DEFAULT_RUN_INTERVAL = 0

# =============================================================================
# Magic Numbers - Retry Configuration
# =============================================================================
RETRY_ATTEMPTS_POST = 3
RETRY_START_TIMEOUT_POST = 2
RETRY_MAX_TIMEOUT_POST = 10
RETRY_ATTEMPTS_GET = 10
RETRY_START_TIMEOUT_GET = 60
RETRY_MAX_TIMEOUT_GET = 1800

# =============================================================================
# File & Path Constants - Extensions and Suffixes
# =============================================================================
JSON_EXTENSION = ".json"
DATA_JSON_SUFFIX = "_data.json"
TRAIN_DATA_SUFFIX = "_train_data.json"
TEST_DATA_SUFFIX = "_test_data.json"
SYNTH_DATA_SUFFIX = "_synth_data.json"
DATASET_SUFFIX = "_dataset.json"

# =============================================================================
# File & Path Constants - Directory Names
# =============================================================================
CACHE_DIR_NAME = ".cache"
WORKER_DIR_NAME = "gradients_worker"
DATASETS_DIR_NAME = "datasets"
MERGED_DATASET_DIR = "merged_dataset"

# =============================================================================
# File & Path Constants - File Prefixes
# =============================================================================
TRAIN_PREFIX = "train_"
TEST_PREFIX = "test_"
SYNTH_PREFIX = "synth_"
DATASET_PREFIX = "dataset_"

# =============================================================================
# File & Path Constants - Config Files
# =============================================================================
CONFIG_FILENAME = "config.yaml"

# =============================================================================
# API Constants - File Formats
# =============================================================================
FILE_FORMAT_HF = "hf"
FILE_FORMAT_S3 = "s3"

# =============================================================================
# RunPod Constants - Job Input/Output Keys
# =============================================================================
RUNPOD_KEY_INPUT = "input"
RUNPOD_KEY_LORA_MODEL_ID = "lora_model_id"
RUNPOD_KEY_BASE_MODEL_ID = "base_model_id"
RUNPOD_KEY_ANONIMIZE = "anonimize"
RUNPOD_KEY_HF_TOKEN = "hf_token"
RUNPOD_KEY_HF_USERNAME = "hf_username"
RUNPOD_KEY_MODEL_REPO_ID = "model_repo_id"
RUNPOD_KEY_STATUS = "status"
RUNPOD_KEY_ERROR = "error"

# =============================================================================
# RunPod Constants - Status Values
# =============================================================================
RUNPOD_STATUS_COMPLETED = "COMPLETED"
RUNPOD_STATUS_FAILED = "FAILED"
RUNPOD_STATUS_SUCCESS = "success"
RUNPOD_STATUS_ERROR = "failed"

# =============================================================================
# API Constants - Endpoint URLs
# =============================================================================
CREATE_TASK_ENDPOINT = "/v1/tasks/create"
TASKS_CREATE_ENDPOINT_CHAT = "/v1/tasks/create_chat"
TASKS_CREATE_WITH_FIXED_DATASETS_ENDPOINT = "/v1/tasks/create_with_fixed_datasets"
TASKS_CREATE_CUSTOM_DATASET_CHAT_ENDPOINT = "/v1/tasks/create_custom_dataset_chat"
GET_TASK_STATUS_ENDPOINT = "/v1/tasks/{task_id}"
GET_TASK_RESULTS_ENDPOINT = "/v1/tasks/task_results/{task_id}"
GET_TASKS_RESULTS_ENDPOINT = "/v1/tasks/breakdown/{task_id}"
AUDITING_TASKS_BY_ID_ENDPOINT = "/auditing/tasks/{task_id}"
