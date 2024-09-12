# # 载入数据集
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from braindecode.datasets import MOABBDataset
from skorch.helper import SliceDataset
from braindecode import EEGClassifier
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
from braindecode.models import EEGNetv4, EEGConformer, ATCNet, EEGITNet, EEGInception,ShallowFBCSPNet
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, TrainEndCheckpoint
from braindecode import EEGClassifier
from skorch.helper import SliceDataset
from skorch.helper import predefined_split
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
from braindecode.util import set_random_seeds
import nn_models.models as mynn
from braindecode.models import to_dense_prediction_model, get_output_shape
from braindecode.training import CroppedLoss
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from braindecode.augmentation import FrequencyShift
from torch.utils.data import Subset
from utils.augmentation import get_augmentation_transform
import dataset_loader
from sklearn.model_selection import train_test_split

from braindecode.augmentation import AugmentedDataLoader, SignFlip
from numpy import linspace
from braindecode.augmentation import FTSurrogate, SmoothTimeMask, ChannelsDropout
from sklearn.datasets._samples_generator import make_blobs

from sklearn.cluster import KMeans
import sklearn
from braindecode.datasets import MOABBDataset
from skorch.helper import SliceDataset
from braindecode import EEGClassifier
from sklearn.manifold import TSNE
subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2015_001", subject_ids=1)
#79.21,59.08,87.20,81.65,75.74,72.27,78.87,80.95,79.21...77.16
#76.73，59。37，87.5,82.98,73.95,73。95，82。98，81。94，78。81
from torchinfo import summary
from numpy import multiply

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)


factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV


preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),
    # Keep EEG sensors
    #Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    #Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
    # Bandpass filter
    Preprocessor(exponential_moving_standardize,
                 # Exponential moving standardization
                 factor_new=factor_new)
]

# Transform the data
preprocess(dataset, preprocessors)
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
print(windows_dataset.description)
splitted = windows_dataset.split('session')
train_set = splitted['0A']  # Session train
valid_set = splitted['1B']  # Session evaluation










# Tensortrain = torch.Tensor(np.array(train_X))

model2 = torch.load('trained_model/best1.pth')
print(model2)
summary(model2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model2 = model2.to(device)

net = EEGClassifier(
    model2,
    device=device
    # To train a neural network you need validation split, here, we use 20%.
)

net.initialize()
valid_set=valid_set
pre=net.forward(valid_set)
print(pre)
pre=pre.detach().numpy()

test_y= valid_set.datasets[0].y#+valid_set.datasets[1].y+valid_set.datasets[2].y+valid_set.datasets[3].y+valid_set.datasets[4].y+valid_set.datasets[5].y#+train_set.datasets[2].y+train_set.datasets[3].y#+train_set.datasets[4].y+train_set.datasets[5].y
print(net.score(valid_set,test_y))
test_y=np.array(test_y)
test_X=[]






y_true = valid_set.get_metadata().target
y_pred = net.predict(valid_set)

# generating confusion matrix
confusion_mat = confusion_matrix(y_true, y_pred)

# add class labels
# label_dict is class_name : str -> i_class : int
label_dict = windows_dataset.datasets[0].window_kwargs[0][1]['mapping']
# sort the labels by values (values are integer class labels)
labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]
#plt.figure(figsize=(20, 20)) 
# plot the basic conf. matrix
plot_confusion_matrix(confusion_mat, class_names=labels)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('confusion_matrix.png')
plt.show()




tsne = TSNE(n_components=2, random_state=42,perplexity=50)
X_tsne = tsne.fit_transform(pre)


x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['Times New Roman']  #

shape_list = ['o', 'o', 'o', 'o']  #
color_list = ['r', 'g', 'b', 'm']  #
label_list = ['Feet', 'Left Hand', 'Rest', 'Right Hand']

for i in range(len(np.unique(test_y))):
    plt.scatter(X_tsne[test_y == i, 0], X_tsne[test_y == i, 1], color=color_list[i],
                marker=shape_list[i], s=150, label=label_list[i], alpha=0.3)



# 添加图例，并设置字体大小
plt.legend(fontsize=20)

ax = plt.gca()  # gca:get current axis


plt.xticks(fontsize=20)  #
plt.yticks(fontsize=20)

plt.xlabel('t-SNE Dimension 1', fontsize=20)  #
plt.ylabel('t-SNE Dimension 2', fontsize=20)
plt.title('t-SNE Visualization', fontsize=24)

plt.show()  # 显示图形
plt.savefig('visualization.png', dpi=600)


y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X_tsne)



plt.figure(figsize=(10, 8))
shape_list = ['o', 'o', 'o', 'o']  #
color_list = ['r', 'g', 'b', 'm']  #
label_list1 = ['Feet', 'Left Hand', 'Rest', 'Right Hand']

# for i in range(len(np.unique(test_y))):
#     plt.scatter(X_tsne[test_y == i, 0], X_tsne[test_y == i, 1],
#                 marker=shape_list[i], s=150, alpha=0.3,c=y_pred)

#plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred,alpha=0.3)

for i in range(len(np.unique(y_pred))):
    plt.scatter(X_tsne[y_pred == i, 0], X_tsne[y_pred == i, 1], color=color_list[i],
                marker=shape_list[i], s=150, label=label_list1[i], alpha=0.3)
plt.legend(fontsize=20)

ax = plt.gca()

plt.xticks(fontsize=20)  #
plt.yticks(fontsize=20)



plt.show()  #
plt.savefig('visualizationk.png', dpi=600)  #


print(sklearn.metrics.calinski_harabasz_score(X_tsne, y_pred))