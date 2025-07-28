# Gradients Scheduler

A continuous worker service for training and evaluating models using the Gradients API.

## Installation

Clone the repository

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

Configuration
Create a .env file in the root directory with the following required variables (example in example.env):

```
GRADIENTS_API_KEY=your_api_key_here
GRADIENTS_API_URL=https://api.gradients.io
WANDB_ENTITY=your_wandb_entity_here
WANDB_API_KEY=your_wandb_api_key_here
CHECK_INTERVAL=600  # Optional, defaults to 600 seconds
HF_USERNAME=your_huggingface_username
HF_TOKEN=your_huggingface_token
S3_COMPATIBLE_ENDPOINT=your_s3_endpoint_url
S3_COMPATIBLE_ACCESS_KEY=your_s3_access_key
S3_COMPATIBLE_SECRET_KEY=your_s3_secret_key
S3_BUCKET_NAME=your_bucket_name

# Delay settings (all optional)
MIN_HOURS_BETWEEN_RUNS=6
MAX_HOURS_BETWEEN_RUNS=8
MIN_DAYS_BETWEEN_RUNS=2
MAX_DAYS_BETWEEN_RUNS=3
```

Running the Worker
To run the worker:

```bash
python -m gradients_worker.main
```

Configuration Files
The main configuration is in config.yaml

```yaml
########################################################
# Example task configuration #1
########################################################
# Multiple tasks can run in parallel, their names appear in the logs
name-of-the-task:
  # If enabled is false, the task will not be run
  enabled: true
  wandb_project: "wandb-project-name"
  # Time interval between the end of the previous run and the start of the next run
  run_intervals:
    min_days: 0
    max_days: 0
    min_hours: 0
    max_hours: 0
  # Datasets are downloaded, merged, shuffled, split into chunks, and uploaded to MinIO for each gradients task
  # Final dataset will default to instruction, output
  datasets:
    # Dataset #1, name is the huggingface dataset identifier, fields are the column names in the dataset
    - name: "yahma/alpaca-cleaned"
      field_instruction: "instruction"
      field_input: "input" # Optional, can be left empty
      field_output: "output"
    # Dataset #2 ...
    - name: "databricks/databricks-dolly-15k"
      field_instruction: "instruction"
      field_input: "context" # Optional, can be left empty
      field_output: "response"
    # etc
  # HuggingFace model identifier to be finetuned, it will be downloaded, verified, and merged with its tokenizer
  model_repo: "unsloth/Meta-Llama-3.1-8B"
  # If specified, the model tokenizer will be updated to the specified HuggingFace repository
  # tokenizer_id:
  # Time to complete the task, in hours
  hours_to_complete: 8
  # Number of samples to use for each gradients training job
  samples_per_training: 150_000
  # Size of the final test dataset, in percentage of the total dataset, this is never shared with gradients
  final_test_size: 0.05
  # Random seed for shuffling the dataset
  random_seed: 45
```

## Run

We advise that you create a screen session in which the scheduler can run indefinitely (until stopped):

```bash
cd gradients-scheduler
screen -S gradients-scheduler -L -Logfile ./gradients-scheduler.log -a
source .venv/bin/activate
python -m gradients_worker.main
```

Then to exit the screen session type, ctr+a then ctrl+d
To observe the logs continuously from outside the screen session, run:

```
tail -n 1000 -f gradients-scheduler.log
```

# Evaluation

The gradients scheduler will store all gradients-related metrics in your specified Wandb project.
