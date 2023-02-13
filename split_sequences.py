"""
This script is only relevant for the two bigger, imbalanced datasets
These datasets contain all frames (in Graph form) of counter attacking sequences - not only the events.

To avoid leaking information between test and train set we randomly split along the sequence ids ('id').
This means that all frames of one sequence will either be in the test set, or in the train set and never in both.
"""

import pickle
import numpy as np
import random
from collections import defaultdict
import collections

with open('women_imbalanced.pkl', 'rb') as f:
    data = pickle.load(f)

# script runs faster if we use 0.2 instead of 0.8
train_test_split_pct = 0.20


def randomly_split_set_test_train_along_sequence_ids(data, test_train_split, seed=None, print_info=True):
    """
    We do some extra effort to split along sequence ids, such that all frames from a sequence are either in the
    test set or in the train set, to avoid leaking information between test and train set

    This will mean that, depending on randomness, the percentage ratio of the two labels will not be equal,
    because we are dealing with sequences of differing lengths
    """
    assert 0 < test_train_split <= 1

    outcomes = np.unique(data['label'])
    train_set, test_set = defaultdict(list), defaultdict(list)

    for outcome in outcomes:
        # For each outcome type - in our case 0 and 1 - we split the data into a test and train set
        # We make sure we include values from a particular sequences only in one of
        # test/train set by splitting along the sequence_id

        outcome_idxs = [i for i, y in enumerate(data['label']) if y[0] == outcome]
        train_count = int(test_train_split * len(outcome_idxs))

        outcome_data = dict()
        keys = ['a', 'x', 'e', 'id', 'label']
        for k in keys:
            outcome_data[k] = data[k][outcome_idxs]

        test_set_idxs = list()
        unique_sequence_ids = set(outcome_data['id'])

        if seed:
            random.seed(seed)
        # sort the list of unique ids, such that the random seed actually works
        # it wouldn't work when just using the set, because set returns the values in a random order
        # that is not controlled by the seed
        unique_sequence_ids_list = sorted(list(unique_sequence_ids))
        random.shuffle(unique_sequence_ids_list)

        i = 0
        while len(test_set_idxs) < train_count:
            # Select a random 'sequence_id' from train_set_outcome['sequence_id']
            sequence_id = unique_sequence_ids_list[i]
            # Remove sequence_id from set of unique ids, so we don't pick it twice
            unique_sequence_ids.remove(sequence_id)
            # Get all indices of that sequence_id
            sequence_idxs = np.where(outcome_data['id'] == sequence_id)[0]
            # Add those indices to the idxs list
            test_set_idxs.extend(sequence_idxs)
            i += 1

        train_set_idxs = np.isin(
            outcome_data['id'],
            np.asarray(list(unique_sequence_ids))
        )
        train_set_idxs = np.where(train_set_idxs)[0]

        # create two datasets, one test one train, add the correct subsets to the correct dicts
        for set_idxs, t_set in zip([test_set_idxs, train_set_idxs], [test_set, train_set]):
            for k in keys:
                t_set[k].extend(data[k][set_idxs])

        if print_info:
            train_set_counts = collections.Counter(np.asarray(train_set['label']).flatten())
            print('Train set counts:', train_set_counts)
            print(f'Train set success pct: {train_set_counts[1] / (train_set_counts[0] + train_set_counts[1]):.2%}')
            test_set_counts = collections.Counter(np.asarray(test_set['label']).flatten())
            print('Test set counts:', test_set_counts)
            print(f'Test set success pct: {test_set_counts[1] / (test_set_counts[0] + test_set_counts[1]):.2%}')

        return test_set, train_set


test_data, train_data = randomly_split_set_test_train_along_sequence_ids(
    data, test_train_split=train_test_split_pct, seed=42, print_info=True
)

