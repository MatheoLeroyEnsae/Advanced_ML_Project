"""
"""

from src.adapter.data import load_dataset_TriviaQA, split_dataset
from omegaconf import DictConfig
import random


def job(config: DictConfig) -> None:

    train_dataset, validation_dataset = load_dataset_TriviaQA(dataset_name="trivia_qa", seed=42)

    answerable_indices, unanswerable_indices = split_dataset(train_dataset)
    val_answerable, _ = split_dataset(validation_dataset)

    unanswerable_indices = []
    validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, config.num_few_shot)
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))