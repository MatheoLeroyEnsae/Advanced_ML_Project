"""

functions to load data

"""
from datasets import Dataset, load_dataset
from typing import Optional, Tuple


def load_dataset_TriviaQA(dataset_name: str, seed: int = 42) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """

    Args:
        dataset_name (str): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        Tuple[Optional[Dataset], Optional[Dataset]]: _description_
    """

    if dataset_name == "trivia_qa":
        dataset = load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']    
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)

    return dataset['train'], dataset['test']


def split_dataset(dataset: Dataset) -> Tuple[Optional[int], Optional[int]]:
    """Get indices of answerable and unanswerable questions."""

    def length(instance):
        return len(instance["answers"]["text"])

    answerable_indices = [i for i, instance in enumerate(dataset) if length(instance) > 0]
    unanswerable_indices = [i for i, instance in enumerate(dataset) if length(instance) == 0]

    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices
