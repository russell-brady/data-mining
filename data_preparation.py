from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd

# data = pd.read_csv('Datasets/relevant_data/cleanedDataset.csv', index_col = 0)
# test_data = pd.read_csv('Datasets/relevant_data/cleanedTestDataset.csv', index_col = 0)

# data = pd.read_csv('Datasets/final_dataset_project.csv')
# test_data = pd.read_csv('Datasets/test_project.csv')


#data.drop(['Unnamed: 0'])

def get_target_label_for_data(data):
	attributes = data.drop(['FTR'],1)
	target_label = data['FTR']

	return attributes, target_label

def get_home_not_home(data):
	data['FTR'] = data.FTR.apply(only_hw)

	attributes = data.drop(['FTR'],1)
	target_label = data['FTR']

	return attributes, target_label


def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'


# attributes, target_label = get_target_label_for_data(data)

# test_attributes , test_target_label = get_target_label_for_data(test_data)

def standardise_data(attributes):
	columns = [['HTGD','ATGD','HTP','ATP','DiffLP']]
	for column in columns:
		attributes[column] = scale(attributes[column])
	
	attributes.HM1 = attributes.HM1.astype('str')
	attributes.HM2 = attributes.HM2.astype('str')
	attributes.HM3 = attributes.HM3.astype('str')
	attributes.AM1 = attributes.AM1.astype('str')
	attributes.AM2 = attributes.AM2.astype('str')
	attributes.AM3 = attributes.AM3.astype('str')
	
	return attributes

# attributes = standardise_data(attributes)

# test_attributes = standardise_data(test_attributes)

def convert_categorical_variables(attributes):
	output = pd.DataFrame(index = attributes.index)
	for col, col_data in attributes.iteritems():
			if col_data.dtype == object:
				col_data = pd.get_dummies(col_data, prefix = col)
			output = output.join(col_data)

	return output

# attributes = convert_categorical_variables(attributes)
# print ("Processed feature columns ({} total features):\n{}".format(len(attributes.columns), list(attributes.columns)))

# print ("\nFeature values:")
# print(attributes.head())

# print(test_target_label)

def split_data(data, test_data):
	attributes, target_label = get_target_label_for_data(data)
	test_attributes , test_target_label = get_target_label_for_data(test_data)
	attributes = standardise_data(attributes)
	test_attributes = standardise_data(test_attributes)
	attributes = convert_categorical_variables(attributes)
	test_attributes = convert_categorical_variables(test_attributes)

	return attributes, test_attributes, target_label, test_target_label


def split_data_only_hw(data, test_data):
	attributes, target_label = get_home_not_home(data)
	test_attributes , test_target_label = get_home_not_home(test_data)
	attributes = standardise_data(attributes)
	test_attributes = standardise_data(test_attributes)
	attributes = convert_categorical_variables(attributes)
	test_attributes = convert_categorical_variables(test_attributes)

	return attributes, test_attributes, target_label, test_target_label