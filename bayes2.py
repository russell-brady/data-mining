import numpy as np
import pandas as p
from data_preparation import *

data = pd.read_csv('Datasets/relevant_data/cleanedDataset.csv', index_col = 0)
test_data = pd.read_csv('Datasets/relevant_data/cleanedTestDataset.csv', index_col = 0)

attributes, test_attributes, target_label, test_target_label = split_data(data, test_data)

for label in target_label:
	print(label)