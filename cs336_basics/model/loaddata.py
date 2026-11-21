# Data Loader
import numpy as np
import torch
def data_loading(dataset, batch_size, context_length, device, rng=None):
    """
        Given a dataset (a 1D numpy array of integers) and a desired batch size and
        context length, sample language modeling input sequences and their corresponding
        labels from the dataset.
    
        Args:
            dataset (np.array): 1D numpy array of integer token IDs in the dataset.
            batch_size (int): Desired batch size to sample.
            context_length (int): Desired context length of each sampled example.
            device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
                to place the sampled input sequences and labels on.
    
        Returns:
            Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
            is the sampled input sequences, and the second tuple item is the corresponding
            language modeling labels.
    """
    n = len(dataset)
    rng = rng if rng is not None else np.random.default_rng()
    starts = rng.integers(0, n - context_length, size=batch_size)
    inputs = np.stack([dataset[i:i+context_length] for i in starts])
    targets = np.stack([dataset[i+1:i+context_length+1] for i in starts])

    return (torch.tensor(inputs,  dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device))