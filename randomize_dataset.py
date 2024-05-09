import random
def randomize_dataset(dataset, seed):
  """Randomizes the order of examples in a dataset while fixing the seed.

  Args:
    dataset: A Hugging Face Dataset object.
    seed: An integer to set the random seed for reproducibility.

  Returns:
    A new Dataset object with the order of examples randomized.
  """

  # Set the random seed for consistent shuffling
  random.seed(seed)

  # Create a shuffled list of indices
  shuffled_indices = list(range(len(dataset)))
  random.shuffle(shuffled_indices)

  # Reorder the dataset using the shuffled indices
  return dataset.select(shuffled_indices)
