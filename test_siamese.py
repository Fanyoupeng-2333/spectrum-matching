import os
import numpy as np
import torch
import torch.utils.data as data
from utils import split_dataset
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from models.baseline import compute_accuracy, PairwiseDistances
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def one_shot_test(splitted_data, settings, num_runs=100):

    device = torch.device("cuda:1" if settings['cuda'] else "cpu")
    saved_net = torch.load(os.path.join(settings['models_dir'], 'best_net_'+str(settings['seed'])+'.tar'))
    net = saved_net['net'].to(device)
    net.eval()

    X_te, y_te = splitted_data['test_input'], splitted_data['test_target']
    print(X_te.shape,y_te.shape)
    X_te = X_te.reshape(X_te.shape + (1,))
    print(X_te.shape,y_te.shape)
    X_te = np.transpose(X_te, axes=[0, 2, 1]).astype('float32')
    y_te = np.squeeze(y_te)
    print(X_te.shape,y_te.shape)

    # Need to re-arrange labels 
    labels = []
    letters = np.sort(np.unique(y_te))
    dict = {l : i for i, l in enumerate(letters)}
    y_te_rep = np.array([dict[y] for y in y_te])

    sm_accs = []
    l2_accs = []
    cos_accs = []
    top3acc  =[]
    top5acc  =[]
    top10acc =[]

    for iter in range(num_runs):
        X_train, y_train, X_test, y_test = split_dataset(X_te, y_te_rep, 
            n_training=1, rnd_index=True)

        references = np.array(X_train)
        refer_labels = np.array(y_train)
        num_test = X_test.shape[0]

        pred_labels = []
        acc = 0
        ind3dd=[]
        ind5dd=[]
        ind10dd=[]
        pred_labels3=0
        pred_labels5=0
        pred_labels10=0

        num_classes = references.shape[0]
        for i in range(num_test):
            x = np.array([X_test[i, :]] * num_classes)
            refers = torch.from_numpy(references).to(device)
            x = torch.from_numpy(x).to(device)
            # print(x.size())
            pred = net(refers, x)
            # print(pred)
            val, index = torch.min(pred, 0)

            _, ind3 = pred.topk(k=3, dim=0, largest=False)
            ind3=ind3.cpu().numpy().tolist()
            ind3dd = [y_train[i] for i in ind3]
            if [y_test[i, :]] in ind3dd:
                pred_labels3+=1
            
            _, ind5 = pred.topk(k=5, dim=0, largest=False)
            ind5=ind5.cpu().numpy().tolist()
            ind5dd = [y_train[i] for i in ind5]
            if [y_test[i, :]] in ind5dd:
                pred_labels5+=1
        
            _, ind10 = pred.topk(k=10, dim=0, largest=False)
            ind10=ind10.cpu().numpy().tolist()
            ind10dd = [y_train[i] for i in ind10]
            if [y_test[i, :]] in ind10dd:
                pred_labels10+=1

            # print(val, index)
            pred_labels.append(refer_labels[index.item()])
        
        acc= pred_labels3/ num_test
        top3acc += [acc]   

        acc=0
        acc= pred_labels5/ num_test
        top5acc += [acc]

        acc=0
        acc= pred_labels10 / num_test
        top10acc += [acc]

        acc=0
        pred_labels = np.array(pred_labels)
        acc = accuracy_score(y_test, pred_labels)  
        sm_accs += [acc]

        X_train = X_train.squeeze()
        X_test = X_test.squeeze()

        acc = 0
        cls = PairwiseDistances(metric='l2')
        cls.fit(X_train, y_train)
        y_pred = cls.predict_labels(X_test)
        acc = accuracy_score(y_test, y_pred)
        l2_accs += [acc]

        acc = 0
        cls = PairwiseDistances(metric='cosine')
        cls.fit(X_train, y_train)
        y_pred = cls.predict_labels(X_test)
        acc = accuracy_score(y_test, y_pred)
        cos_accs += [acc]

        # print('Run {0}, Siamese_acc {1:.3f}, cos_acc {2:.3f}, l2_acc {3:.3f}'.
        #    format(iter, sm_accs[-1], cos_accs[-1], l2_accs[-1])) 
        print('Run {0}, Siamese_acc {1:.3f}, cos_acc {2:.3f}, l2_acc {3:.3f}, top3acc {4:.3f}, top5acc {5:.3f}, top10acc {6:.3f}'.
           format(iter, sm_accs[-1], cos_accs[-1], l2_accs[-1],top3acc[-1], top5acc[-1], top10acc[-1]))    

    # print('Siamese_acc {0:.3f}({1:.3f}), cos_acc {2:.3f}({3:.3f}), l2_acc {4:.3f}'
    #     '({5:.3f})'.format(np.mean(sm_accs), np.std(sm_accs), np.mean(cos_accs), 
    #     np.std(cos_accs), np.mean(l2_accs), np.std(l2_accs)))
    print('Siamese_acc {0:.3f}({1:.3f}), cos_acc {2:.3f}({3:.3f}), l2_acc {4:.3f}'
        '({5:.3f}), top3acc {6:.3f}({7:.3f}), top5acc {8:.3f}({9:.3f}),' 
        'top10acc {10:.3f}({11:.3f})'.format(np.mean(sm_accs), np.std(sm_accs), np.mean(cos_accs), 
        np.std(cos_accs), np.mean(l2_accs), np.std(l2_accs),
        np.mean(top3acc), np.std(top3acc),np.mean(top5acc), 
        np.std(top5acc),np.mean(top10acc), np.std(top10acc)))

    return np.mean(sm_accs), np.mean(cos_accs), np.mean(l2_accs)
