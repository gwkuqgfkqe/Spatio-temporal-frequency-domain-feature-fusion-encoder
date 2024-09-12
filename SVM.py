
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from torch import nn
import torch
from braindecode.datasets import MOABBDataset
from skorch.helper import SliceDataset
subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

from numpy import multiply

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),
    # Keep EEG sensors
    #Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
    # Bandpass filter
    Preprocessor(exponential_moving_standardize,
                 # Exponential moving standardization
                 factor_new=factor_new)
]

# Transform the data
preprocess(dataset, preprocessors, n_jobs=-1)
from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

splitted = windows_dataset.split('session')
train_set = splitted['0train']  # Session train
valid_set = splitted['1test']  # Session evaluation

#生成数据库
data_X, data_Y = datasets.make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2,
    random_state=21, n_clusters_per_class=2,
    scale=100)
# 对数据库进行划分
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.3)





model1 = SVC(C = 0.5,decision_function_shape='ovo',kernel='poly',verbose=True)

train_y=train_set.datasets[0].y+train_set.datasets[1].y+train_set.datasets[2].y+train_set.datasets[3].y#+train_set.datasets[4].y+train_set.datasets[5].y
train_X=[]
for i in range(0, 192):
    sub_train=train_set[i][0]
    train_X.append(sub_train)
np.stack(train_X, axis=1)


combined_array = np.concatenate(train_X, axis=0)
Tensortrain = torch.Tensor(combined_array)
#Tensortrain=Tensortrain.unsqueeze(-1)
Tensortrain=Tensortrain.reshape(192,24750)
model2 = torch.load('trained_model/best1.pth')
print(model2)


x=model2(Tensortrain)
x=x.detach().numpy()
model1.fit(x,y=train_y)

test_y=train_set.datasets[4].y+train_set.datasets[5].y#+train_set.datasets[2].y+train_set.datasets[3].y#+train_set.datasets[4].y+train_set.datasets[5].y
test_X=[]
for i in range(192, 288):
    sub_train=train_set[i][0]
    # concatenate()
    test_X.append(sub_train)
np.stack(test_X, axis=1)
combined_array = np.concatenate(test_X, axis=0)
Tensortrain = torch.Tensor(combined_array)
#Tensortrain=Tensortrain.unsqueeze(-1)
Tensortrain=Tensortrain.reshape(96,24750)

x=model2(Tensortrain)
x=x.detach().numpy()


print(model1.score(x,test_y))
#
# train_X=train_set[0][0].reshape(1,24750)
# for i in range(22)
# for i in range(1, 192):
#     sub_train=train_set[i][0].reshape(1,24750)
#     for j in range(len(sub_train)):
#
#         train_X[j] += sub_train[j]
#     #train_X.append(sub_train)
# test_X=train_set[192][0].reshape(1,24750)
# for i in range(193, 288):
#     sub_train = train_set[i][0].reshape(1, 24750)
#     for j in range(len(sub_train)):
#         test_X[j] += sub_train[j]
#
# model2 = SVC(C = 0.5,decision_function_shape='ovo',kernel='poly',verbose=True)
# model2.fit(train_X,y=train_y)
# print(model1.score(x,test_y))
# print(model2.score(test_X,test_y))
