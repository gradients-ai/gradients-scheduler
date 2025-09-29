import hashlib
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .models import TaskType

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from .utils import save_dataset_to_temp, upload_to_minio

logger = logging.getLogger(__name__)


class DatasetsScheduler:
    """Handles dataset preparation for fine-tuning tasks."""

    def __init__(self, task_config: dict, cache_dir: Optional[str] = None):
        """Initialize the datasets scheduler.

        Args:
            task_config: The task configuration from config.yaml
            cache_dir: Directory to cache the datasets (default: None)
        """
        self.task_config = task_config
        self.cache_dir = cache_dir or os.path.join(
            Path.home(), ".cache", "gradients_worker", "datasets"
        )
        self.dataset_configs = task_config.get("datasets", [])
        self.random_seed = task_config.get("random_seed", 42)
        self.final_test_size = task_config.get("final_test_size", 0.05)
        self.samples_per_training = task_config.get("samples_per_training", 150_000)
        task_type_str = self.task_config.get("task_type", "InstructText")
        self.task_type = TaskType(task_type_str)

        if isinstance(self.dataset_configs, dict):
            items_str = str(sorted(self.dataset_configs.items()))
        else:
            items_str = str(self.dataset_configs)
        config_hash = hashlib.md5(items_str.encode()).hexdigest()[:10]

        self.task_name = task_config.get("wandb_project", "task")
        self.dataset_dir = os.path.join(
            self.cache_dir, f"{self.task_name}_{config_hash}"
        )

        os.makedirs(self.dataset_dir, exist_ok=True)
        self.merged_dataset_path = os.path.join(self.dataset_dir, "merged_dataset")

        # We'll determine total_samples when needed
        self._total_samples = None

    def _download_datasets(self) -> List[Tuple[Dataset, dict]]:
        """Download all datasets specified in the configuration.

        Returns:
            List[Tuple[Dataset, dict]]: List of downloaded datasets with their configs
        """
        downloaded_datasets = []

        for ds_config in self.dataset_configs:
            dataset_name = ds_config["name"]
            try:
                logger.info(f"Downloading dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

                # Most datasets have a 'train' split
                if isinstance(dataset, DatasetDict) and "train" in dataset:
                    dataset = dataset["train"]

                downloaded_datasets.append((dataset, ds_config))
                logger.info(
                    f"Successfully downloaded dataset: {dataset_name} with {len(dataset)} samples"
                )
            except Exception as e:
                logger.error(
                    f"Failed to download dataset {dataset_name}: {e}", exc_info=True
                )

        return downloaded_datasets

    def _standardize_dataset(self, dataset: Dataset, ds_config: dict) -> Dataset:
        """Standardize dataset column names to instruction, input, output.

        Args:
            dataset: The dataset to standardize
            ds_config: The dataset configuration with field mappings

        Returns:
            Dataset: Standardized dataset with canonical column names
        """
        field_instruction = ds_config.get("field_instruction")
        field_input = ds_config.get("field_input")
        field_output = ds_config.get("field_output")

        rename_mapping = {}
        if field_instruction and field_instruction != "instruction":
            rename_mapping[field_instruction] = "instruction"
        if field_output and field_output != "output":
            rename_mapping[field_output] = "output"
        if field_input and field_input != "input":
            rename_mapping[field_input] = "input"

        if rename_mapping:
            dataset = dataset.rename_columns(rename_mapping)

        # Check if we need to add an empty input field
        if not field_input or "input" not in dataset.column_names:
            logger.info(f"Adding empty 'input' field to dataset {ds_config['name']}")
            dataset = dataset.add_column("input", ["" for _ in range(len(dataset))])

        columns_to_keep = ["instruction", "input", "output"]
        existing_columns = set(dataset.column_names)

        # Ensure all required columns exist
        for col in columns_to_keep:
            if col not in existing_columns:
                raise ValueError(
                    f"Required column '{col}' not found in dataset after standardization"
                )

        dataset = dataset.select_columns(columns_to_keep)
        return dataset

    @property
    def total_samples(self) -> int:
        """Get total number of samples in the dataset.

        Lazy-loads the dataset if the sample count isn't known yet.

        Returns:
            int: Total number of samples
        """
        if self._total_samples is None:
            if not os.path.exists(self.merged_dataset_path):
                raise ValueError("Dataset has not been prepared yet")

            # Load dataset to get sample count
            dataset = load_from_disk(self.merged_dataset_path)
            self._total_samples = len(dataset)
            logger.info(f"Loaded dataset size: {self._total_samples} samples")

        return self._total_samples

    @total_samples.setter
    def total_samples(self, value: int) -> None:
        """Set the total number of samples.

        Args:
            value: Number of samples to set
        """
        self._total_samples = value

    def is_prepared(self) -> bool:
        """Check if the dataset has been prepared and saved to disk.

        Returns:
            bool: True if dataset is prepared, False otherwise
        """
        return os.path.exists(self.merged_dataset_path)

    def prepare_datasets(self) -> bool:
        """Download, standardize, merge, and save datasets if not already prepared.

        Returns:
            bool: True if datasets were prepared successfully, False otherwise
        """
        if self.is_prepared():
            logger.info(
                f"Datasets already prepared at {self.merged_dataset_path}, skipping preparation"
            )
            # Access the property to ensure total_samples is set
            _ = self.total_samples
            return True

        try:
            downloaded_datasets = self._download_datasets()
            if not downloaded_datasets:
                logger.error("No datasets were successfully downloaded")
                return False

            standardized_datasets = []
            for dataset, ds_config in downloaded_datasets:
                if self.task_type != TaskType.CHAT:
                    standardized_dataset = self._standardize_dataset(dataset, ds_config)
                else:
                    # For Chat task type, use the data  set as-is
                    standardized_dataset = dataset
                standardized_datasets.append(standardized_dataset)
                logger.info(
                    f"Standardized dataset {ds_config['name']} with {len(standardized_dataset)} samples"
                )

            merged_dataset = concatenate_datasets(standardized_datasets)
            logger.info(
                f"Merged {len(standardized_datasets)} datasets with total {len(merged_dataset)} samples"
            )

            merged_dataset = merged_dataset.shuffle(seed=self.random_seed)
            logger.info(f"Shuffled merged dataset with seed {self.random_seed}")

            self.total_samples = len(merged_dataset)

            merged_dataset.save_to_disk(self.merged_dataset_path)
            logger.info(f"Saved merged dataset to {self.merged_dataset_path}")

            return True

        except Exception as e:
            logger.error(f"Error preparing datasets: {e}", exc_info=True)
            return False

    @property
    def num_chunks(self) -> int:
        """Get the number of dataset chunks.

        Returns:
            int: Number of dataset chunks
        """
        train_size = int(self.total_samples * (1 - self.final_test_size))
        return math.ceil(train_size / self.samples_per_training)

    def get_chunk_indices(self, chunk_index: int) -> Tuple[int, int]:
        """Get the start and end indices for a specific chunk.

        Args:
            chunk_index: Index of the chunk

        Returns:
            Tuple[int, int]: Start and end indices
        """
        if not 0 <= chunk_index < self.num_chunks:
            raise IndexError(
                f"Chunk index {chunk_index} out of range (0-{self.num_chunks-1})"
            )

        train_size = int(self.total_samples * (1 - self.final_test_size))
        chunk_size = math.ceil(train_size / self.num_chunks)

        start_idx = chunk_index * chunk_size
        end_idx = min((chunk_index + 1) * chunk_size, train_size)

        return start_idx, end_idx

    def get_chunk(self, index: int) -> Dataset:
        """Get a specific dataset chunk by index.

        Args:
            index: Index of the chunk to retrieve

        Returns:
            Dataset: The requested chunk
        """
        if not self.is_prepared():
            raise ValueError("Datasets have not been prepared yet")

        start_idx, end_idx = self.get_chunk_indices(index)

        # Load only the required slice of the dataset
        dataset = load_from_disk(self.merged_dataset_path)
        chunk = dataset.select(range(start_idx, end_idx))

        logger.info(
            f"Loaded chunk {index} with {len(chunk)} samples (indices {start_idx}-{end_idx})"
        )
        return chunk

    def get_test_dataset(self) -> Dataset:
        """Get the test dataset.

        Returns:
            Dataset: The test dataset
        """
        if not self.is_prepared():
            raise ValueError("Datasets have not been prepared yet")

        train_size = int(self.total_samples * (1 - self.final_test_size))

        # Load only the test portion of the dataset
        dataset = load_from_disk(self.merged_dataset_path)
        test_dataset = dataset.select(range(train_size, self.total_samples))

        logger.info(
            f"Loaded test dataset with {len(test_dataset)} samples (indices {train_size}-{self.total_samples})"
        )
        return test_dataset

    def upload_test_dataset(self) -> str:
        """Get the test dataset and upload it to Minio.

        Returns:
            str: URL for the uploaded test dataset

        Raises:
            ValueError: If datasets aren't prepared
        """
        test_dataset = self.get_test_dataset()

        test_name = f"{os.urandom(8).hex()}_test_data.json"

        test_path, _ = save_dataset_to_temp(test_dataset, prefix="test_")
        test_url = upload_to_minio(test_path, test_name)

        logger.info(f"Uploaded test dataset with {len(test_dataset)} samples")

        return test_url

    def prepare_and_upload_chunk(self, chunk_index: int) -> Tuple[str, str, str]:
        """Get a chunk, split it into train/test/synth, save as JSONs and upload to Minio.

        Args:
            chunk_index: Index of the chunk to process

        Returns:
            Tuple[str, str, str]: URLs for train, test, and synth JSON files

        Raises:
            ValueError: If datasets aren't prepared or chunk index is invalid
        """
        chunk = self.get_chunk(chunk_index)
        chunk = chunk.shuffle()  # No need for seed=self.random_seed to contaminate

        total_size = len(chunk)
        train_size = int(total_size * 0.90)
        test_size = int(total_size * 0.08)
        # synth_size will be the remainder

        train_data = chunk.select(range(train_size))
        test_data = chunk.select(range(train_size, train_size + test_size))
        synth_data = chunk.select(range(train_size + test_size, total_size))

        train_name = f"{os.urandom(8).hex()}_train_data.json"
        test_name = f"{os.urandom(8).hex()}_test_data.json"
        synth_name = f"{os.urandom(8).hex()}_synth_data.json"

        train_path, _ = save_dataset_to_temp(train_data, prefix="train_")
        train_url = upload_to_minio(train_path, train_name)

        test_path, _ = save_dataset_to_temp(test_data, prefix="test_")
        test_url = upload_to_minio(test_path, test_name)

        synth_path, _ = save_dataset_to_temp(synth_data, prefix="synth_")
        synth_url = upload_to_minio(synth_path, synth_name)

        logger.info(
            f"Split and uploaded chunk {chunk_index} "
            f"(train: {len(train_data)}, test: {len(test_data)}, synth: {len(synth_data)} samples)"
        )

        return train_url, test_url, synth_url

    def prepare_and_upload_whole_chunk(self, chunk_index: int) -> str:
        """Get a chunk and upload it to Minio without splitting.

        Args:
            chunk_index: Index of the chunk to process

        Returns:
            str: URL for the uploaded dataset

        Raises:
            ValueError: If datasets aren't prepared or chunk index is invalid
        """
        chunk = self.get_chunk(chunk_index)
        chunk = chunk.shuffle()  # Shuffle the data

        base_name = os.urandom(8).hex()
        dataset_name = f"{base_name}_dataset.json"

        dataset_path, _ = save_dataset_to_temp(chunk, prefix="dataset_")
        dataset_url = upload_to_minio(dataset_path, dataset_name)

        logger.info(f"Uploaded whole chunk {chunk_index} with {len(chunk)} samples")

        return dataset_url
