import argparse
import os
import time

from experiments.bci2a import BCI2aExperiment
#from experiments.physionet import physionet
from utils.utils import read_yaml, save_json2file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bci2a', choices=['bci2a', 'physionet'],
                        help='data set used of the experiments')
    parser.add_argument('--model', type=str, default='EEGNet',
                        choices=['EEGNet', 'EEGConformer', 'ATCNet', 'EEGInception', 'EEGITNet','ShallowFBCSPNet'],
                        help='model used of the experiments')
    parser.add_argument('--config', type=str, default='default', help='config file name(.yaml format)')
    parser.add_argument('--strategy', type=str, default='within-subject', choices=['cross-subject', 'within-subject'],
                        help='experiments strategy on subjects')
    parser.add_argument('--save', action='store_true', help='save the pytorch model and history')
    parser.add_argument('--method', type=str, default='trialwise', choices=['trialwise', 'corpped'],
                        help='experiments method on training')
    parser.add_argument('--seed', type=int, default=20337190,help='help to reproduce results')
    parser.add_argument('--method_train', type=str, default='train_val_test', choices=['train_test', 'train_val_test','k_fold'],
                        help='experiments method on training')
    parser.add_argument('--augmentation', type=list, default=[0,0,0],
                        choices=[[0,0,0], [0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0], [1,1,1]],
                        help='dataset augmentation(Frequency, Time, Spatial)')
    args = parser.parse_args()
    # suit default config for specific dataset and model
    if args.config == 'default':
        args.config = f'{args.dataset}_{args.model}_{args.config}.yaml'
    # read config from yaml file
    config = read_yaml(f"{os.getcwd()}/config/{args.config}")
    # result save directory
    save_dir = f"{os.getcwd()}/save/{args.dataset}/{int(time.time())}_{args.dataset}_{args.model}/"
    args.save_dir = save_dir
    print(config)
    if args.save:
        save_json2file(config, save_dir, f'{args.dataset}_{args.model}_config.json')
    # for every dataset
    if args.dataset == 'bci2a':
        exp = BCI2aExperiment(args=args, config=config)
        exp.run()
    elif args.dataset == 'physionet':
        raise Warning('physionet experiments are developing.')
        # physionet(args, config)
