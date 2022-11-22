from models import siamese
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from utils import save_checkpoint
from collections import Counter
from datasets.minerals import MINERAL_107
from datasets.minerals import RRUFF_CMP
from datasets.minerals import RRUFF_USGS ,RRUFF_USGS1,Lorentz_18
from datasets.substances import ChemKitchen
from pair_generator import create_random_pairs
from pair_generator import create_pairs_wrt_means
from models.lenet import LeNet
from models.siamese import SiameseNetwork
from sklearn.metrics import f1_score, accuracy_score
from models.baseline import compute_accuracy
from models.baseline import PairwiseDistances
from utils import get_logger
from train_siamese import train as train_siamese_model
from train_siamese import set_config as set_config_siamese
from train_cnn import set_config as set_config_cnn
from train_siamese import split_training_test_classes
from test_siamese import one_shot_test
from test_siamese import multi_class_test
from train_cnn import train as train_cnn_model
from models.lenet import LeNetClassifier
from models.res2net import res2net50_26w_4s, res2net101_26w_4s
from models.pyconvnet import pyconvresnet50
from models.scalenet import scalenet50
# from models.fanres2 import res2netfan
# from models.fanres21 import res2netfan21
# from models.fanres22 import res2netfan22
from models.fanres24 import res2netfan24
# from models.fanres27 import res2netfan27
from models.resnet import resnet50

from train_cnn import multi_class_test as cnn_multi_class_test
import os
import shutil
import argparse
import matplotlib.pyplot as plt
import random
parser = argparse.ArgumentParser(description='Train ConvNets for Spectrum Recognition')
parser.add_argument('--model', dest="model", type=str)
parser.set_defaults(model="siamese")
parser.add_argument('--epochs', default=50, help='number of epochs', type=int)
parser.add_argument('--batch_size', default=16, help='batch size', type=int)
parser.add_argument('--train', dest='train', action='store_true', help='flag to start training')
parser.set_defaults(train=True)
parser.add_argument('--test', dest='test', action='store_true', help='run test')
parser.set_defaults(test=False)
parser.add_argument('--eval', dest='eval', action='store_true', help='run series of tests to evaluate performance')
parser.set_defaults(eval=False)
parser.add_argument('--eval_stat', dest='eval_stat', action='store_true', help='run series of tests to evaluate performance')
parser.set_defaults(eval_stat=False)
parser.add_argument('--verbose', dest='verbose', action='store_true', help='flag to start training')
parser.set_defaults(verbose=True)
parser.add_argument('--plot', dest='plot', action='store_true', help='Plot')
parser.set_defaults(plot=False)
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
parser.add_argument('--models_dir', dest="models_dir", type=str)
parser.set_defaults(models_dir="TrainedModels")
parser.add_argument('--dataset', dest="dataset", type=str)
parser.set_defaults(dataset="lz18")
parser.add_argument('--log', dest="log", type=str)
parser.set_defaults(log="main.log")
parser.add_argument('--dataset_resplitting', dest='dataset_resplitting', action='store_true', help='flag to resplit dataset to training and test sets')
parser.set_defaults(dataset_resplitting=False)
parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true', help='use data augmentation or not.')
parser.set_defaults(data_augmentation=False)
parser.add_argument('--num_random_pairs_per_sample', default=30, help='number of random pairs per sample', type=int)
parser.add_argument('--num_ways', default=5, help='number of classes', type=int)
parser.add_argument('--num_shots', default=5, help='number of examples per class', type=int)
parser.add_argument('--num_queries', default=1, help='number of querieis per episode', type=int)
parser.add_argument('--num_episodes', default=2000, help='number of test episodes', type=int)
parser.add_argument('--num_episodes_per_epoch', default=500, help='number of episodes', type=int)
parser.add_argument('--num_episodes_halve_lr', default=5000, help='number of episodes', type=int)
args = parser.parse_args()


DATASET_DIR = "datasets"
################################################################################
# GLOBAL SETTINGS
settings = {
    'model': None,
    'dataset': None,
    'dataset_name': args.dataset,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'log_dir': args.log,
    'logger': None,
    'seed':None,
    'sedd':None,
    'verbose': args.verbose,
    'cuda': args.cuda,
    'models_dir': args.models_dir,
    'data_augmentation': args.data_augmentation,
    'dataset_resplitting': args.dataset_resplitting,
    'num_random_pairs_per_sample': args.num_random_pairs_per_sample,
    'num_episodes': args.num_episodes, 
    'num_ways': args.num_ways, 
    'num_shots': args.num_shots, 
    'num_queries': args.num_queries,
    'num_episodes_per_epoch': args.num_episodes_per_epoch,
    'num_episodes_halve_lr': args.num_episodes_halve_lr}

logger = get_logger(log_file_name=args.log)
settings['logger'] = logger
###############################################################################
# Dataset
if  args.dataset.lower() == "rruff_cmp":
    settings['dataset_name'] = "RRUFF_CMP"
    settings['dataset'] =  RRUFF_CMP()
elif args.dataset.lower() == "usg":
    settings['dataset_name'] = "usg"
    settings['dataset'] =  RRUFF_USGS()
elif args.dataset.lower() == "usgs1":
    settings['dataset_name'] = "usgs1"
    settings['dataset'] =  RRUFF_USGS1()
elif args.dataset.lower() == "lz18":
    settings['dataset_name'] = "lz18"
    settings['dataset'] =  Lorentz_18()
else:
    print('Not exist')
   
     
###############################################################################
# Methods
lenet_settings = {
    'ConvLayers': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M'],    
    # 'ConvLayers': [16, 'M', 32,  'M', 64,'M'],    
    # 'ConvLayers': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M'],    
    'Conv1x1Layers':[1000, 'A'],
    'TensorChannels': 1,
}
def setnet_config(model_name):
    if model_name == 'siamese':    
        net = SiameseNetwork(embedding_network=LeNet(settings=lenet_settings))
        args.models_dir = os.path.join(args.models_dir, 'Siamese')

    if model_name == 'res2net':    
        net = SiameseNetwork(embedding_network= res2net50_26w_4s(pretrained=False))
        args.models_dir = os.path.join(args.models_dir, 'Res2net')
    
    if model_name == 'fanres2':    
        net = SiameseNetwork(embedding_network= res2netfan24()) 
        args.models_dir = os.path.join(args.models_dir, 'fanres2')

    if model_name == 'resnet':    
        net = SiameseNetwork(embedding_network= resnet50(pretrained=False))
        args.models_dir = os.path.join(args.models_dir, 'Resnet')

    if model_name == 'google':    
        net = SiameseNetwork(embedding_network= GoogleNet(num_classes=212, aux_logits=True, init_weight=False))
        args.models_dir = os.path.join(args.models_dir, 'Resnet')
    
    if model_name == 'pyconv':    
        net = SiameseNetwork(embedding_network= pyconvresnet50(pretrained=False))
        args.models_dir = os.path.join(args.models_dir, 'Pyconv')

    if model_name == 'scale':    
        net = SiameseNetwork(embedding_network= scalenet50("models/scale50.json"))
        args.models_dir = os.path.join(args.models_dir, 'scale')

    if model_name == 'cnn': 
        net= LeNetClassifier(settings=lenet_settings, num_classes=settings['dataset'].n_classes)
        args.models_dir = os.path.join(args.models_dir, 'CNN')
    return net

################################################################################
def setup_seed(seed,sedd):
    #  sedd=1
     torch.manual_seed(sedd)

     torch.cuda.manual_seed_all(sedd)

     np.random.seed(seed)
     
     random.seed(seed)

     torch.backends.cudnn.deterministic = True

################################################################################
def main():
    if args.train:
        # print(settings)
        if args.dataset_resplitting: # Default False
            print('----------------------------------------------------------')
            print('Notice: resplitting the dataset into training and test '+
                     'sets and  save to disk for further usage')
            print('----------------------------------------------------------')
            data = split_training_test_classes(settings['dataset'])
            torch.save(data, os.path.join(DATASET_DIR, 
                'train_test_classes.tensors'))
        else:
            print('------------------------------------')
            print('Notice: use existing data splits.')
            print('------------------------------------')
            data = torch.load(os.path.join(DATASET_DIR, 
                'train_test_classes.tensors'))

        if args.model.lower() == 'siamese':
            train_siamese_model(data)
            print("hello")
        elif args.model.lower() == 'res2net':
            train_siamese_model(data)
            print("hello")
        elif args.model.lower() == 'fanres2':
            train_siamese_model(data)
        elif args.model.lower() == 'google':
            train_siamese_model(data)
        elif args.model.lower() == 'resnet':
            train_siamese_model(data)
        elif args.model.lower() == 'pyconv':
            train_siamese_model(data)
        elif args.model.lower() == 'scale':
            train_siamese_model(data)
        elif args.model.lower() == 'cnn':
            for i in range(1):
                print("2222222")
                train_cnn_model(data,i)
            # os.remove('net_0.tar')            
        else:
            print('Model not found')
        return 

    if args.test:
        # data = torch.load(os.path.join(DATASET_DIR, 
        #     'train_test_classes.tensors'))
        data = split_training_test_classes(settings['dataset'])
        if args.model.lower() == 'siamese':
            one_shot_test(data, settings)
        elif args.model.lower() == 'res2net':
            one_shot_test(data, settings)
        elif args.model.lower() == 'resnet':
            one_shot_test(data, settings)
        elif args.model.lower() == 'fanres2':
            one_shot_test(data, settings)
            # multi_class_test(data, settings)
        elif args.model.lower() == 'cnn':
            # cnn_multi_class_test(data, settings)
            for ii in range(3):
                print("3333333")
                cnn_multi_class_test(data, settings,ii)         
        else:
            print('Model not found')
        return

if __name__ == "__main__":

    sedlist=[0,1,2,10,16]#[2,10,16] #[3,4,2,6,10,11,12] [0,1,2,10,16] [3,5,7,15,20]
    seddlist=0
    
    for i in sedlist:
        setup_seed(i,i)
        settings['seed'] = i
        net=setnet_config(args.model.lower())
        settings['model'] = net
        logger.info(settings)
        logger.info('seed:{}  sedd:{}'.format(i,i))
        # setup_seed(1)
        # a=np.array([1,2,3,4,5])
        # b=np.array([1,2,3,4,5])
        # plt.plot(a,b)
        # # 设置横轴标签
        # plt.xlabel("label")
        # # 设置纵轴标签
        # plt.ylabel("numbers")
        # # plt.show()
        # plt.savefig('./data_analysis.jpg')
        if args.model.lower() =='siamese':
            set_config_siamese(settings)
            main()
        elif args.model.lower() =='cnn':
            set_config_cnn(settings)
            main()
        elif args.model.lower() =='res2net':
            set_config_siamese(settings)
            main()
        elif args.model.lower() =='fanres2':
            set_config_siamese(settings)
            main()
        elif args.model.lower() =='google':
            set_config_siamese(settings)
            main()
        elif args.model.lower() =='pyconv':
            set_config_siamese(settings)
            main()
        elif args.model.lower() =='scale':
            set_config_siamese(settings)
            main()
        elif args.model.lower() =='resnet':
            set_config_siamese(settings)
            main()
        else:
            print('Model not found!')

