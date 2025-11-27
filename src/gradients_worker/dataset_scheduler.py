import hashlib
import logging
import math
import os
from pathlib import Path

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from . import constants as cst
from .models import TaskType
from .utils import save_and_upload_dataset

logger = logging.getLogger(__name__)


class DatasetsScheduler:
    """Handles dataset preparation for fine-tuning tasks."""

    def __init__(self, task_config: dict, cache_dir: str | None = None):
        """Initialize the datasets scheduler.

        Args:
            task_config: The task configuration from config.yaml
            cache_dir: Directory to cache the datasets (default: None)
        """
        self.task_config = task_config
        self.cache_dir = cache_dir or os.path.join(
            Path.home(), cst.CACHE_DIR_NAME, cst.WORKER_DIR_NAME, cst.DATASETS_DIR_NAME
        )
        self.dataset_configs = task_config.get(cst.KEY_DATASETS, [])
        self.random_seed = task_config.get(cst.KEY_RANDOM_SEED, cst.DEFAULT_RANDOM_SEED)
        self.final_test_size = task_config.get(
            cst.KEY_FINAL_TEST_SIZE, cst.DEFAULT_FINAL_TEST_SIZE
        )
        self.samples_per_training = task_config.get(
            cst.KEY_SAMPLES_PER_TRAINING, cst.DEFAULT_SAMPLES_PER_TRAINING
        )
        self.per_chunk_test_proportion = task_config.get(
            cst.KEY_PER_CHUNK_TEST_PROPORTION, cst.DEFAULT_PER_CHUNK_TEST_PROPORTION
        )
        task_type_str = self.task_config.get(cst.KEY_TASK_TYPE, "InstructText")
        self.task_type = TaskType(task_type_str)

        if isinstance(self.dataset_configs, dict):
            items_str = str(sorted(self.dataset_configs.items()))
        else:
            items_str = str(self.dataset_configs)
        config_hash = hashlib.md5(items_str.encode()).hexdigest()[:10]

        self.task_name = task_config.get(cst.KEY_WANDB_PROJECT, "task")
        self.dataset_dir = os.path.join(
            self.cache_dir, f"{self.task_name}_{config_hash}"
        )

        os.makedirs(self.dataset_dir, exist_ok=True)
        self.merged_dataset_path = os.path.join(
            self.dataset_dir, cst.MERGED_DATASET_DIR
        )

        # We'll determine total_samples when needed
        self._total_samples = None

    def _download_datasets(self) -> list[tuple[Dataset, dict]]:
        """Download all datasets specified in the configuration.

        Returns:
            list[tuple[Dataset, dict]]: List of downloaded datasets with their configs
        """
        downloaded_datasets = []

        for ds_config in self.dataset_configs:
            dataset_name = ds_config[cst.KEY_NAME]
            try:
                logger.info(f"Downloading dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

                # Most datasets have a 'train' split
                if isinstance(dataset, DatasetDict) and "train" in dataset:
                    dataset = dataset["train"]

                # Apply max_rows subsampling if specified
                max_rows = ds_config.get(cst.KEY_MAX_ROWS)
                if max_rows is not None and len(dataset) > max_rows:
                    logger.info(
                        f"Subsampling dataset {dataset_name} from {len(dataset)} to {max_rows} rows"
                    )
                    dataset = dataset.shuffle(seed=self.random_seed).select(
                        range(max_rows)
                    )

                downloaded_datasets.append((dataset, ds_config))
                logger.info(
                    f"Successfully downloaded dataset: {dataset_name} with {len(dataset)} samples"
                )
            except Exception as e:
                logger.error(
                    f"Failed to download dataset {dataset_name}: {e}", exc_info=True
                )

        return downloaded_datasets

    def _standardize_dataset(
        self, dataset: Dataset, ds_config: dict, source_idx: int
    ) -> Dataset:
        """Standardize dataset column names to instruction, input, output.

        Args:
            dataset: The dataset to standardize
            ds_config: The dataset configuration with field mappings
            source_idx: Index of this dataset in the source list (for stratification)

        Returns:
            Dataset: Standardized dataset with canonical column names, or original dataset for Chat tasks
        """

        if (
            self.task_type == TaskType.CHAT
            or self.task_type == TaskType.CUSTOMDATASETCHAT
        ):
            chat_column = ds_config.get(cst.KEY_CHAT_COLUMN, cst.DEFAULT_CHAT_COLUMN)
            chat_role_field = ds_config.get(
                cst.KEY_CHAT_ROLE_FIELD, cst.DEFAULT_CHAT_ROLE_FIELD
            )
            chat_content_field = ds_config.get(
                cst.KEY_CHAT_CONTENT_FIELD, cst.DEFAULT_CHAT_CONTENT_FIELD
            )
            chat_user_reference = ds_config.get(
                cst.KEY_CHAT_USER_REFERENCE, cst.DEFAULT_CHAT_USER_REFERENCE
            )
            chat_assistant_reference = ds_config.get(
                cst.KEY_CHAT_ASSISTANT_REFERENCE, cst.DEFAULT_CHAT_ASSISTANT_REFERENCE
            )

            if chat_column in dataset.column_names:

                def standardize_conversation(example):
                    conversations = example[chat_column]
                    standardized = []
                    for conv in conversations:
                        role = conv.get(chat_role_field, "")
                        if role == chat_user_reference:
                            role = cst.DEFAULT_CHAT_USER_REFERENCE
                        elif role == chat_assistant_reference:
                            role = cst.DEFAULT_CHAT_ASSISTANT_REFERENCE

                        standardized_conv = {
                            cst.DEFAULT_CHAT_ROLE_FIELD: role,
                            cst.DEFAULT_CHAT_CONTENT_FIELD: conv.get(
                                chat_content_field, ""
                            ),
                        }
                        standardized.append(standardized_conv)
                    return {cst.DEFAULT_CHAT_COLUMN: standardized}

                cols_to_remove = [
                    col
                    for col in dataset.column_names
                    if col != cst.DEFAULT_CHAT_COLUMN
                ]
                dataset = dataset.map(
                    standardize_conversation, remove_columns=cols_to_remove
                )
                logger.info(
                    f"Standardized chat dataset {ds_config[cst.KEY_NAME]} to use '{cst.DEFAULT_CHAT_COLUMN}' with '{cst.DEFAULT_CHAT_ROLE_FIELD}' and '{cst.DEFAULT_CHAT_CONTENT_FIELD}' fields"
                )

            dataset = dataset.add_column(
                cst.SOURCE_INDEX_COLUMN, [source_idx] * len(dataset)
            )
            return dataset

        elif self.task_type in (
            TaskType.INSTRUCTTEXT,
            TaskType.INSTRUCTTEXTWITHFIXEDDATASETS,
        ):
            field_instruction = ds_config.get(cst.KEY_FIELD_INSTRUCTION)
            field_input = ds_config.get(cst.KEY_FIELD_INPUT)
            field_output = ds_config.get(cst.KEY_FIELD_OUTPUT)

            rename_mapping = {}
            if field_instruction and field_instruction != cst.DEFAULT_INSTRUCTION_FIELD:
                rename_mapping[field_instruction] = cst.DEFAULT_INSTRUCTION_FIELD
            if field_output and field_output != cst.DEFAULT_OUTPUT_FIELD:
                rename_mapping[field_output] = cst.DEFAULT_OUTPUT_FIELD
            if field_input and field_input != cst.DEFAULT_INPUT_FIELD:
                rename_mapping[field_input] = cst.DEFAULT_INPUT_FIELD

            if rename_mapping:
                dataset = dataset.rename_columns(rename_mapping)

            # Check if we need to add an empty input field
            if not field_input or cst.DEFAULT_INPUT_FIELD not in dataset.column_names:
                logger.info(
                    f"Adding empty '{cst.DEFAULT_INPUT_FIELD}' field to dataset {ds_config[cst.KEY_NAME]}"
                )
                dataset = dataset.add_column(
                    cst.DEFAULT_INPUT_FIELD, ["" for _ in range(len(dataset))]
                )

            columns_to_keep = [
                cst.DEFAULT_INSTRUCTION_FIELD,
                cst.DEFAULT_INPUT_FIELD,
                cst.DEFAULT_OUTPUT_FIELD,
            ]
            existing_columns = set(dataset.column_names)

            # Ensure all required columns exist
            for col in columns_to_keep:
                if col not in existing_columns:
                    raise ValueError(
                        f"Required column '{col}' not found in dataset after standardization"
                    )

            dataset = dataset.select_columns(columns_to_keep)

            dataset = dataset.add_column(
                cst.SOURCE_INDEX_COLUMN, [source_idx] * len(dataset)
            )
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
            for source_idx, (dataset, ds_config) in enumerate(downloaded_datasets):
                standardized_dataset = self._standardize_dataset(
                    dataset, ds_config, source_idx
                )
                standardized_datasets.append(standardized_dataset)
                logger.info(
                    f"Standardized dataset {ds_config['name']} (source_idx={source_idx}) with {len(standardized_dataset)} samples"
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

    def _get_train_test_indices_by_source(
        self, dataset: Dataset
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Get train and test indices for each source, ensuring no overlap.

        For each source, deterministically shuffles its indices and splits into
        train/test based on final_test_size. This ensures:
        - Stratified train/test split (same proportion from each source)
        - Complete separation between train and test indices

        Args:
            dataset: The merged dataset with _source_idx column

        Returns:
            tuple[dict, dict]: (train_indices_by_source, test_indices_by_source)
        """
        source_indices = dataset[cst.SOURCE_INDEX_COLUMN]

        indices_by_source: dict[int, list[int]] = {}
        for i, source_idx in enumerate(source_indices):
            if source_idx not in indices_by_source:
                indices_by_source[source_idx] = []
            indices_by_source[source_idx].append(i)

        train_by_source = {}
        test_by_source = {}

        for source_idx in sorted(indices_by_source.keys()):
            all_indices = indices_by_source[source_idx]

            rng = np.random.RandomState(self.random_seed)
            rng.shuffle(all_indices)

            n_test = max(1, int(len(all_indices) * self.final_test_size))
            test_by_source[source_idx] = all_indices[:n_test]
            train_by_source[source_idx] = all_indices[n_test:]

        return train_by_source, test_by_source

    def get_chunk(self, index: int) -> Dataset:
        """Get a specific dataset chunk with proportional representation from each source.

        Each chunk contains the same proportion of each source dataset, ensuring
        stratified sampling. Uses deterministic random state for reproducibility.
        Only uses train indices (completely separate from test indices).

        Args:
            index: Index of the chunk to retrieve

        Returns:
            Dataset: The requested chunk with proportional source representation
        """
        if not self.is_prepared():
            raise ValueError("Datasets have not been prepared yet")

        if not 0 <= index < self.num_chunks:
            raise IndexError(
                f"Chunk index {index} out of range (0-{self.num_chunks - 1})"
            )

        dataset = load_from_disk(self.merged_dataset_path)
        train_by_source, _ = self._get_train_test_indices_by_source(dataset)

        total_train = sum(len(indices) for indices in train_by_source.values())

        rng = np.random.RandomState(self.random_seed + index)

        chunk_indices = []
        for train_indices in train_by_source.values():
            source_count = len(train_indices)
            if source_count == 0:
                continue

            source_proportion = source_count / total_train
            samples_per_chunk = max(
                1, int(self.samples_per_training * source_proportion)
            )

            start = index * samples_per_chunk
            end = min(start + samples_per_chunk, source_count)

            if start >= source_count:
                start = start % source_count
                end = min(start + samples_per_chunk, source_count)

            chunk_indices.extend(train_indices[start:end])

        rng.shuffle(chunk_indices)

        chunk = dataset.select(chunk_indices)

        logger.info(
            f"Loaded chunk {index} with {len(chunk)} samples (stratified from {len(train_by_source)} sources)"
        )
        return chunk

    def get_test_dataset(self) -> Dataset:
        """Get the stratified test dataset.

        Returns a test dataset with proportional representation from each source.
        Test indices are completely separate from train indices used by chunks.

        Returns:
            Dataset: The stratified test dataset
        """
        if not self.is_prepared():
            raise ValueError("Datasets have not been prepared yet")

        dataset = load_from_disk(self.merged_dataset_path)
        _, test_by_source = self._get_train_test_indices_by_source(dataset)

        # Collect all test indices
        test_indices = []
        for indices in test_by_source.values():
            test_indices.extend(indices)

        test_dataset = dataset.select(test_indices)

        logger.info(
            f"Loaded stratified test dataset with {len(test_dataset)} samples from {len(test_by_source)} sources"
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

        if cst.SOURCE_INDEX_COLUMN in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns([cst.SOURCE_INDEX_COLUMN])

        test_name = f"{os.urandom(8).hex()}_test_data.json"
        test_url = save_and_upload_dataset(test_dataset, test_name, prefix="test_")

        logger.info(f"Uploaded test dataset with {len(test_dataset)} samples")

        return test_url

    def _split_by_source(
        self,
        dataset: Dataset,
        test_proportion: float,
        seed: int,
    ) -> tuple[Dataset, Dataset]:
        """Split dataset while maintaining proportions of each source.

        Args:
            dataset: The dataset to split (must have _source_idx column)
            test_proportion: Fraction of data to use for test
            seed: Random seed for reproducibility

        Returns:
            tuple[Dataset, Dataset]: Train and test datasets
        """
        rng = np.random.RandomState(seed)

        source_indices = dataset[cst.SOURCE_INDEX_COLUMN]

        indices_by_source: dict[int, list[int]] = {}
        for i, source_idx in enumerate(source_indices):
            if source_idx not in indices_by_source:
                indices_by_source[source_idx] = []
            indices_by_source[source_idx].append(i)

        train_indices = []
        test_indices = []

        for source in sorted(indices_by_source.keys()):
            indices = indices_by_source[source]
            rng.shuffle(indices)

            n_test = max(1, int(len(indices) * test_proportion))
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])

        return dataset.select(train_indices), dataset.select(test_indices)

    def prepare_and_upload_chunk(self, chunk_index: int) -> tuple[str, str, str]:
        """Get a chunk, split it into train/test/synth, save as JSONs and upload to Minio.

        Args:
            chunk_index: Index of the chunk to process

        Returns:
            tuple[str, str, str]: URLs for train, test, and synth JSON files

        Raises:
            ValueError: If datasets aren't prepared or chunk index is invalid
        """
        chunk = self.get_chunk(chunk_index)
        chunk = chunk.shuffle()  # No need for seed=self.random_seed to contaminate

        total_size = len(chunk)
        train_size = int(total_size * cst.TRAIN_SPLIT_RATIO)
        test_size = int(total_size * cst.TEST_SPLIT_RATIO)
        # synth_size will be the remainder

        train_data = chunk.select(range(train_size))
        test_data = chunk.select(range(train_size, train_size + test_size))
        synth_data = chunk.select(range(train_size + test_size, total_size))

        if cst.SOURCE_INDEX_COLUMN in train_data.column_names:
            train_data = train_data.remove_columns([cst.SOURCE_INDEX_COLUMN])
            test_data = test_data.remove_columns([cst.SOURCE_INDEX_COLUMN])
            synth_data = synth_data.remove_columns([cst.SOURCE_INDEX_COLUMN])

        train_name = f"{os.urandom(8).hex()}_train_data.json"
        test_name = f"{os.urandom(8).hex()}_test_data.json"
        synth_name = f"{os.urandom(8).hex()}_synth_data.json"

        train_url = save_and_upload_dataset(train_data, train_name, prefix="train_")
        test_url = save_and_upload_dataset(test_data, test_name, prefix="test_")
        synth_url = save_and_upload_dataset(synth_data, synth_name, prefix="synth_")

        logger.info(
            f"Split and uploaded chunk {chunk_index} "
            f"(train: {len(train_data)}, test: {len(test_data)}, synth: {len(synth_data)} samples)"
        )

        return train_url, test_url, synth_url

    def prepare_and_upload_chunk_train_test(self, chunk_index: int) -> tuple[str, str]:
        """Get a chunk, split into train/test with stratification, upload to Minio.

        Uses per_chunk_test_proportion for the split and maintains source proportions.
        No synthetic data is generated.

        Args:
            chunk_index: Index of the chunk to process

        Returns:
            tuple[str, str]: URLs for train and test JSON files

        Raises:
            ValueError: If datasets aren't prepared or chunk index is invalid
        """
        chunk = self.get_chunk(chunk_index)

        split_seed = self.random_seed + chunk_index * 1000
        train_data, test_data = self._split_by_source(
            chunk,
            test_proportion=self.per_chunk_test_proportion,
            seed=split_seed,
        )

        if cst.SOURCE_INDEX_COLUMN in train_data.column_names:
            train_data = train_data.remove_columns([cst.SOURCE_INDEX_COLUMN])
            test_data = test_data.remove_columns([cst.SOURCE_INDEX_COLUMN])

        train_name = f"{os.urandom(8).hex()}_train_data.json"
        test_name = f"{os.urandom(8).hex()}_test_data.json"

        train_url = save_and_upload_dataset(train_data, train_name, prefix="train_")
        test_url = save_and_upload_dataset(test_data, test_name, prefix="test_")

        logger.info(
            f"Split and uploaded chunk {chunk_index} "
            f"(train: {len(train_data)}, test: {len(test_data)} samples, stratified)"
        )

        return train_url, test_url

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

        if cst.SOURCE_INDEX_COLUMN in chunk.column_names:
            chunk = chunk.remove_columns([cst.SOURCE_INDEX_COLUMN])

        base_name = os.urandom(8).hex()
        dataset_name = f"{base_name}_dataset.json"

        dataset_url = save_and_upload_dataset(chunk, dataset_name, prefix="dataset_")

        logger.info(f"Uploaded whole chunk {chunk_index} with {len(chunk)} samples")

        return dataset_url
