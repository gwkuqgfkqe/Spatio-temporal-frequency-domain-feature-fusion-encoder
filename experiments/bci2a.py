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
#import nn_models.models as mynn
from braindecode.models import to_dense_prediction_model, get_output_shape
from braindecode.training import CroppedLoss
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from braindecode.augmentation import FrequencyShift
from torch.utils.data import Subset
from utils.augmentation import get_augmentation_transform
from dataset_loader import DatasetBraindecode,DatasetBraindecode
from sklearn.model_selection import train_test_split
from nn_models.STFEnc import STFEnc
from utils.utils import save_str2file
from braindecode.augmentation import AugmentedDataLoader, SignFlip
from numpy import linspace
from braindecode.augmentation import FTSurrogate, SmoothTimeMask, ChannelsDropout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device= 'cuda'
class BCI2aExperiment:
    def __init__(self, args, config):
        # load and preprocess data using braindecode
        
        self.ds = DatasetBraindecode.DatasetFromBraindecode('bci2a', subject_ids=None)
        
        self.ds.preprocess(resample_freq=config['dataset']['resample'],
                           high_freq=config['dataset']['high_freq'],
                           low_freq=config['dataset']['low_freq'])
        
        # clip to create window dataset

        self.n_channels = self.ds.get_channel_num()
        self.n_classes = config['dataset']['n_classes']
        self.classes = list(range(self.n_classes))
        # training routine
        self.seed = args.seed
        self.n_epochs = config['fit']['epochs']
        self.lr = config['fit']['lr']
        self.batch_size = config['fit']['batch_size']
        self.augmentation=args.augmentation
        # self.freq_shift = FrequencyShift(
        #     probability=.5,
        #     sfreq=self.ds.get_sample_freq(),
        #     max_delta_freq=2.  # the frequency shifts are sampled now between -2 and 2 Hz
        # )
        #
        # self.sign_flip = SignFlip(probability=.1)
        transforms_freq = [FTSurrogate(probability=0.5, phase_noise_magnitude=phase_freq,
                                       random_state=self.seed) for phase_freq in linspace(0, 1, 2)]

        transforms_time = [SmoothTimeMask(probability=0.5, mask_len_samples=int(self.ds.get_sample_freq() * second),
                                          random_state=self.seed) for second in linspace(0.1, 2, 2)]

        transforms_spatial = [ChannelsDropout(probability=0.5, p_drop=prob,
                                              random_state=self.seed) for prob in linspace(0, 1, 2)]

        buffer=[]#=transforms_freq+transforms_time+transforms_spatial
        buffer.append(transforms_freq)
        buffer.append(transforms_time)
        buffer.append(transforms_spatial)
        self.transforms = []
        list1=[0,1,2]
        #for i ,j in zip(list1,self.augmentation):
            #if j==1:
                #self.transforms.append(buffer[i])
        self.transforms=[]#transforms_spatial#transforms_time#transforms_freq#get_augmentation_transform()#transforms_spatial+transforms_freq+transforms_time
        #print(self.transforms)


        # user options
        self.save = args.save
        self.save_dir = args.save_dir
        self.strategy = args.strategy
        self.model_name = args.model
        self.verbose = config['fit']['verbose']
        self.method=args.method
        self.method_train=args.method_train
        set_random_seeds(seed=self.seed, cuda=device)
        # load deep leaning model from braindecode
        if self.method=='trialwise':
            self.windows_dataset = self.ds.create_windows_dataset(
                trial_start_offset_seconds=config['dataset']['start_offset'],
                trial_stop_offset_seconds=config['dataset']['stop_offset']
            )
            self.n_times = self.ds.get_input_window_sample()
            if args.model == 'EEGNet':
                # self.model = EEGNetv4(in_chans=self.n_channels, n_classes=self.n_classes,
                #                     n_times=self.n_times)
                #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model = STFEnc(n_channels=self.n_channels, n_classes=self.n_classes,
                                     input_window_size=self.n_times)
                # self.model=models.BaseCNN(n_channels=self.n_channels, n_classes=self.n_classes,
                #                      input_window_size=self.n_times)
                # self.model = EEGITNet(n_chans=self.n_channels, n_outputs=self.n_classes, n_times=self.n_times,
                #                       add_log_softmax=False)
                # self.model =EEGConformer(n_outputs=self.n_classes, n_chans=self.n_channels, n_times=self.n_times,
                #                           final_fc_length='auto', add_log_softmax=False)
                print(self.model)
                #summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model=='ShallowFBCSPNet':
                self.model = ShallowFBCSPNet(n_chans=self.n_channels, n_outputs=self.n_classes,
                                         input_window_samples=self.n_times,final_conv_length='auto')
                summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model == 'EEGConformer':
                self.model = EEGConformer(n_outputs=self.n_classes, n_chans=self.n_channels, n_times=self.n_times,
                                          final_fc_length='auto', add_log_softmax=False)
                summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model == 'ATCNet':
                self.model = ATCNet(n_chans=self.n_channels, n_outputs=self.n_classes, n_times=self.n_times,
                                    add_log_softmax=False)
                summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model == 'EEGITNet':
                self.model = EEGITNet(n_chans=self.n_channels, n_outputs=self.n_classes, n_times=self.n_times,
                                      add_log_softmax=False)
                summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model == 'EEGInception':
                self.model = EEGInception(n_chans=self.n_channels, n_outputs=self.n_classes, n_times=self.n_times,
                                          add_log_softmax=False)
                summary(self.model, (1, self.n_channels, self.n_times))
            else:
                raise ValueError(f"model {args.model} is not supported on this dataset.")
            #if cuda:
            self.model.to(device)
        else:
            self.n_times=800
            if args.model == 'EEGNet':
                self.model = EEGNetv4(n_chans=self.n_channels, n_classes=self.n_classes,
                                      n_times=self.n_times,final_conv_length=21)
                print(self.model)
                #summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model=='ShallowFBCSPNet':
                self.model = ShallowFBCSPNet(n_chans=self.n_channels, n_outputs=self.n_classes,
                                         input_window_samples=self.n_times,final_conv_length=30)
                summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model == 'EEGConformer':
                self.model = EEGConformer(n_outputs=self.n_classes, n_chans=self.n_channels, n_times=self.n_times,
                                          final_fc_length='auto', add_log_softmax=False)
                summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model == 'ATCNet':
                self.model = ATCNet(n_chans=self.n_channels, n_outputs=self.n_classes, n_times=self.n_times,
                                    add_log_softmax=False)
                summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model == 'EEGITNet':
                self.model = EEGITNet(n_chans=self.n_channels, n_outputs=self.n_classes, n_times=self.n_times,
                                      add_log_softmax=False)
                summary(self.model, (1, self.n_channels, self.n_times))
            elif args.model == 'EEGInception':
                self.model = EEGInception(n_chans=self.n_channels, n_outputs=self.n_classes, n_times=self.n_times,
                                          add_log_softmax=False)
                summary(self.model, (1, self.n_channels, self.n_times))
            else:
                raise ValueError(f"model {args.model} is not supported on this dataset.")
            #if cuda:
            self.model.cuda()
            to_dense_prediction_model(self.model)
            n_preds_per_input = self.model.get_output_shape()[2]
            print(n_preds_per_input)
            self.windows_dataset = self.ds.create_windows_dataset(
                trial_start_offset_seconds=config['dataset']['start_offset'],
                trial_stop_offset_seconds=config['dataset']['stop_offset'],
                window_size_samples=self.n_times,
                window_stride_samples=n_preds_per_input,
            )

    def __get_classifier(self, val_set,lr):

        #
        # train_split=None
        # if self.method_train == 'train_test' or 'k_fold':
        #     train_split = None
        # elif self.method_train == 'train_val_test':
        #     train_split = predefined_split(test_set)
        # for different models, suit the training routines or other params in the [origin paper or code] for classifier
        if self.method=='trialwise':
            if self.model_name == 'EEGNet' or 'ShallowFBCSPNet':
                callbacks = ["accuracy",("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
                return EEGClassifier(module=self.model,
                                     classes=self.classes,
                                     criterion=torch.nn.CrossEntropyLoss,
                                     optimizer=torch.optim.Adam,
                                     optimizer__lr=lr,
                                     #iterator_train=AugmentedDataLoader,
                                     #iterator_train__transforms=self.transforms,
                                     train_split=(self.method_train=='train_val_test') and predefined_split(val_set) or None,
                                     optimizer__weight_decay=0,
                                     batch_size=self.batch_size,
                                     callbacks=callbacks,
                                     device='cuda:0' if device else 'cpu',
                                     #verbose=self.verbose
                                     max_epochs=1000
                                     )

            elif self.model_name == 'EEGConformer':
                callbacks = ["accuracy",("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
                return EEGClassifier(module=self.model,
                                     classes=self.classes,
                                     criterion=torch.nn.CrossEntropyLoss,
                                     optimizer=torch.optim.Adam,
                                     optimizer__betas=(0.5, 0.999),
                                     optimizer__lr=lr,
                                     train_split=(self.method_train=='train_val_test') and predefined_split(val_set) or None,
                                     iterator_train__shuffle=True,
                                     #iterator_train=AugmentedDataLoader,
                                     #iterator_train__transforms=self.transforms,
                                     batch_size=self.batch_size,
                                     callbacks=callbacks,
                                     device='cuda' if device else 'cpu',
                                     verbose=self.verbose
                                     )
            elif self.model_name == 'ATCNet' or self.model_name == 'EEGITNet' or self.model_name == 'EEGInception':
                callbacks = ["accuracy",("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
                return EEGClassifier(module=self.model,
                                     classes=self.classes,
                                     criterion=torch.nn.CrossEntropyLoss,
                                     optimizer=torch.optim.Adam,
                                     optimizer__lr=lr,
                                     #iterator_train=AugmentedDataLoader,
                                     #iterator_train__transforms=self.transforms,
                                     train_split=(self.method_train=='train_val_test') and predefined_split(val_set) or None,
                                     iterator_train__shuffle=True,
                                     batch_size=self.batch_size,
                                     callbacks=callbacks,
                                     device='cuda' if device else 'cpu',
                                     #verbose=self.verbose
                                     )
        else:
            if self.model_name == 'EEGNet' or 'ShallowFBCSPNet':
                callbacks = [
                    "accuracy",
                    ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
                return EEGClassifier(self.model,
                                     cropped=True,
                                     classes=self.classes,
                                     criterion=CroppedLoss(torch.nn.functional.cross_entropy),  #CrossEntropyLoss,
                                     optimizer=torch.optim.AdamW,
                                     #iterator_train=AugmentedDataLoader,
                                     # This tells EEGClassifier to use a custom DataLoader
                                     #iterator_train__transforms=self.transforms,
                                     #criterion_loss_function=torch.nn.functional.nll_loss,
                                     optimizer__lr=lr,
                                     train_split=(self.method_train=='train_val_test') and predefined_split(val_set) or None,
                                     optimizer__weight_decay=0,
                                     batch_size=self.batch_size,
                                     iterator_train__shuffle=True,
                                     callbacks=callbacks,
                                     device='cuda' if device else 'cpu',
                                     #verbose=self.verbose
                                     )

            elif self.model_name == 'EEGConformer':
                callbacks = ["accuracy",("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
                return EEGClassifier(module=self.model,
                                     cropped=True,
                                     classes=self.classes,
                                     criterion=CroppedLoss,  # CrossEntropyLoss,
                                     optimizer=torch.optim.AdamW,
                                     criterion_loss_function=torch.nn.functional.nll_loss,
                                     optimizer__betas=(0.5, 0.999),
                                     optimizer__lr=lr,
                                     iterator_train=AugmentedDataLoader,
                                     iterator_train__transforms=self.transforms,
                                     train_split=(self.method_train=='train_val_test') and predefined_split(val_set) or None,
                                     iterator_train__shuffle=True,
                                     batch_size=self.batch_size,
                                     callbacks=callbacks,
                                     device='cuda' if device else 'cpu',
                                     verbose=self.verbose
                                     )
            elif self.model_name == 'ATCNet' or self.model_name == 'EEGITNet' or self.model_name == 'EEGInception':
                callbacks = ["accuracy",("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
                return EEGClassifier(module=self.model,
                                     cropped=True,
                                     classes=self.classes,
                                     criterion=CroppedLoss,  # CrossEntropyLoss,
                                     optimizer=torch.optim.AdamW,
                                     criterion_loss_function=torch.nn.functional.nll_loss,
                                     optimizer__lr=lr,
                                     iterator_train=AugmentedDataLoader,
                                     iterator_train__transforms=self.transforms,
                                     train_split=(self.method_train=='train_val_test') and predefined_split(val_set) or None,
                                     iterator_train__shuffle=True,
                                     batch_size=self.batch_size,
                                     callbacks=callbacks,
                                     device='cuda' if device else 'cpu',
                                     #verbose=self.verbose
                                     )

    def __within_subject_experiment(self):
        #  split dataset for single subject
        subjects_windows_dataset = self.windows_dataset.split('subject')
        n_subjects = len(subjects_windows_dataset.items())
        avg_accuracy = 0
        result = ''
        i=1
        for subject, windows_dataset in subjects_windows_dataset.items():
            if i!=1:
                self.lr=0.001
            i+=1
            # evaluate the model by test accuracy for "Hold-Out" strategy
            print(windows_dataset.description)
            train_dataset = windows_dataset.split('session')['0train']
            test_dataset = windows_dataset.split('session')['1test']
            train_X = SliceDataset(train_dataset, idx=0)
            train_y = SliceDataset(train_dataset, idx=1)
            test_X = SliceDataset(test_dataset, idx=0)
            test_y = SliceDataset(test_dataset, idx=1)

            #X_train = SliceDataset(train_set, idx=0)
            #y_train = np.array([y for y in SliceDataset(train_set, idx=1)])
            train_indices, val_indices = train_test_split(
                train_X.indices_, test_size=0.1, shuffle=False
            )
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)

            clf = self.__get_classifier(val_subset,self.lr)
            # save the last epoch model for test
            if self.save:
                clf.callbacks.append(TrainEndCheckpoint(dirname=self.save_dir + f'\\S{subject}'))
            if self.method_train=='k_fold':
                train_val_split = KFold(n_splits=6, shuffle=False)
                # By setting n_jobs=-1, cross-validation is performed
                # with all the processors, in this case the output of the training
                # process is not printed sequentially
                cv_results = cross_val_score(
                    clf, train_X, train_y, scoring="accuracy", cv=train_val_split, n_jobs=1
                )
                # print(
                #     f"Validation accuracy: {np.mean(cv_results * 100):.2f}"
                #     f"+-{np.std(cv_results * 100):.2f}%"
                # )

                avg_accuracy += np.mean(cv_results)
                print(f"Subject{subject} test accuracy: {(np.mean(cv_results ) * 100):.4f}%")
                result += f"Subject{subject} test accuracy: {(np.mean(cv_results ) * 100):.4f}%\n"
            else:
                if self.method_train=='train_test':
                    clf.fit(train_dataset,y=None, epochs=self.n_epochs)
                else:
                    clf.fit(train_subset, y=None,epochs=self.n_epochs)
            # calculate test accuracy for subject
                test_accuracy = clf.score(test_X, y=test_y)
                avg_accuracy += test_accuracy
                print(f"Subject{subject} test accuracy: {(test_accuracy * 100):.4f}%")
                result += f"Subject{subject} test accuracy: {(test_accuracy * 100):.4f}%\n"
            torch.save(self.model,'best'+str(subject)+'.pth')
            # if i==10:
            #     # generate confusion matrices
            #     # get the targets
            #     y_true = test_dataset.get_metadata().target
            #     y_pred = clf.predict(test_dataset)

            #     # generating confusion matrix
            #     confusion_mat = confusion_matrix(y_true, y_pred)
                
            #     # add class labels
            #     # label_dict is class_name : str -> i_class : int
            #     label_dict = self.windows_dataset.datasets[0].window_kwargs[0][1]['mapping']
            #     # sort the labels by values (values are integer class labels)
            #     labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]
            #     #plt.figure(figsize=(20, 20)) 
            #     # plot the basic conf. matrix
            #     plot_confusion_matrix(confusion_mat, class_names=labels)
            #     plt.xticks(fontsize=9)
            #     plt.yticks(fontsize=9)
            #     plt.savefig('confusion_matrix.pdf')
            #     plt.show()
        print(f"Average test accuracy: {(avg_accuracy / n_subjects * 100):.4f}%")
        # save the result
        result += f"Average test accuracy: {(avg_accuracy / n_subjects * 100):.4f}%"
        save_str2file(result, self.save_dir, 'result.txt')
        # Extract loss and accuracy values for plotting from history object
        results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
        df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                          index=clf.history[:, 'epoch'])

        # get percent of misclass for better visual comparison to loss
        df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                       valid_misclass=100 - 100 * df.valid_accuracy)

        fig, ax1 = plt.subplots(figsize=(8, 3))
        df.loc[:, ['train_loss', 'valid_loss']].plot(
            ax=ax1, style=['-', ':'],  color='tab:blue', legend=False, fontsize=14)

        ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
        ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        df.loc[:, ['train_misclass', 'valid_misclass']].plot(
            ax=ax2, style=['-', ':'],  color='tab:red', legend=False)
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
        ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
        ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
        ax1.set_xlabel("Epoch", fontsize=14)

        # where some data has already been plotted to ax
        handles = []
        handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
        handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
        plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
        plt.tight_layout()
        plt.savefig('myfigure.png')

    def __cross_subject_experiment(self):
        pass

    def run(self):
        if self.strategy == 'within-subject':
            self.__within_subject_experiment()
        elif self.strategy == 'cross-subject':
            self.__cross_subject_experiment()
