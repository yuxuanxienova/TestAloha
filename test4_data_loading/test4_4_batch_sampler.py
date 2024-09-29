import numpy as np
"""
Why Is It Written This Way?
Purpose of the Sampler Design
Efficient Sampling Across Episodes:

Episode-Based Sampling:
The sampler first selects an episode and then a step within that episode.
This maintains the association between steps and their episodes, which can be important for context in sequence modeling.
Weighted Episode Sampling:

sample_weights:
Allows prioritizing certain episodes over others.
For example, if some episodes are more informative or underrepresented, you can assign them higher weights.
Handling Variable-Length Episodes:

Variable Episode Lengths:
The episodes may have different numbers of steps.
By using sum_dataset_len_l, the sampler correctly maps episode indices to the global indices in the flattened dataset.
Global Indexing:

Flattened Dataset:
Steps from all episodes are assumed to be concatenated into a single dataset.
The sampler provides indices that can directly index into this flattened dataset.
Advantages of This Approach
Flexibility:

Supports any number of episodes and varying lengths.
Can handle changes in episode lengths without altering the core logic.
Control Over Sampling Distribution:

By adjusting sample_weights, you can control the probability of sampling from each episode.
Scalability:

Efficient for large datasets since it avoids the need to preprocess all possible step indices.
Simplicity:

The code is concise yet powerful, leveraging NumPy functions for random sampling and cumulative sums.
"""
def BatchSampler(batch_size, episode_len_l, sample_weights):
    #Normalizes the sample_weights to create a probability distribution over episodes.
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    # Calculates the cumulative sum of episode lengths to map episode indices to global step indices.
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            #Randomly selects an episode index, using sample_probs if provided, or uniformly if sample_probs is None.
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            #Randomly selects a step index within the range of the selected episode.
            #sum_dataset_len_l[episode_idx]: The starting index of the selected episode.
            #sum_dataset_len_l[episode_idx + 1]: The starting index of the next episode, also the ending index of the selected episode.
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch
        
        