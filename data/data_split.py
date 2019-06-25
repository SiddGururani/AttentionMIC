import numpy as np

DATA_NPZ_PATH = 'openmic-2018.npz'
TRAIN_SPLIT_PATH = 'split01_train.csv'
TEST_SPLIT_PATH = 'split01_test.csv'

# Load full dataset
data = np.load(DATA_NPZ_PATH)
X, Y_true, Y_mask, sample_key = data['X'], data['Y_true'], data['Y_mask'], data['sample_key']

# Load training split csv file
with open(TRAIN_SPLIT_PATH) as f:
    train_IDs = f.readlines()
    train_IDs = np.array([ID.strip() for ID in train_IDs])

# Load test split csv file
with open(TEST_SPLIT_PATH) as f:
    test_IDs = f.readlines()
    test_IDs = np.array([ID.strip() for ID in test_IDs])

# Get the training and testing split data into np arrays
train_index = np.array([i for i in range(20000) if sample_key[i] in train_IDs])
test_index = np.array([i for i in range(20000) if sample_key[i] in test_IDs])

np.savez('train.npz',X = X[train_index], Y_true = Y_true[train_index],
    Y_mask = Y_mask[train_index], sample_key = sample_key[train_index])

np.savez('test.npz',X = X[test_index], Y_true = Y_true[test_index],
    Y_mask = Y_mask[test_index], sample_key = sample_key[test_index])
	