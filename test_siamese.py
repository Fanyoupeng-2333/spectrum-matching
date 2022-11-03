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


def multi_class_test(splitted_data, settings):
    """Perform multi-class classification on test data"""
    
    val_loss = np.load(os.path.join(settings['models_dir'], 'val_loss.npy'))
    num_nets = len(val_loss)
    best_index = np.argmin(val_loss)

    saved_net = torch.load(os.path.join(settings['models_dir'], 
       'best_net.tar'))
    # net = torch.load(os.path.join(settings['models_dir'], 
    #   'net_'+str(num_nets-1)+'.tar'))

    net = saved_net['net']
    acc = _net_multi_class_test(net, splitted_data, settings)
    print('Best Siamese Accuracy: {} {:.5f}'.format(best_index, acc))

    settings['verbose']=False
    
    if settings['verbose']:
        print('---------------------------------------------------------')
        test_err = []
        for iter in range(0, num_nets):
            saved_net = torch.load(os.path.join(settings['models_dir'], 
                'net_'+str(iter)+'.tar'))
            net = saved_net
            acc = _net_multi_class_test(net, splitted_data, settings)
            print('iter {}, accuracy {:.5f}'.format(iter, acc))
            test_err.append(1-acc)
        
    # plt.plot(val_loss, 'b')
    # plt.plot(test_err, 'k')
    # plt.plot(best_index, test_err[best_index], 'rd')
    # plt.show()
    return acc


def _net_multi_class_test(net, data, settings):
    """Helper function."""
    
    device = torch.device("cuda" if settings['cuda'] else "cpu")
    # Make sure the network is in eval mode
    net = net.to(device)
    net.eval()

    X_tr, y_tr = data['train_input'], data['train_target']
    X_va, y_va = data['valid_input'], data['valid_target']
    X_te, y_te = data['test_input'], data['test_target']
    X_test, y_test = data['loo_input'], data['loo_target']        
    X_train = np.vstack((X_tr, X_va, X_te))
    y_train = np.vstack((y_tr, y_va, y_te))

    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    X_train = np.transpose(X_train, axes=[0, 2, 1]).astype('float32')
    X_test = np.transpose(X_test, axes=[0, 2, 1]).astype('float32')
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    num_classes = settings['dataset'].n_classes
    num_test = X_test.shape[0]        
        
    # Prepare references
    spec_templates = []
    for i in range(num_classes):
        # spec_templates.append(np.mean(X_train[np.where(y_train==i)[0], :],axis=0))
        idx = np.where(y_train==i)[0]
        spec_templates.append(X_train[idx[0],:])
    spec_templates = np.array(spec_templates)

    pred_labels = []
    # batchsize1=40
    acc = 0
    print("hello")
    templates = torch.from_numpy(spec_templates).to(device)
    for i in range(num_test):
        x = np.array([X_test[i, :]] * num_classes)
        print(type(x),type(spec_templates))
        x = torch.from_numpy(x).to(device)

        dataset = torch.utils.data.TensorDataset(templates, x)
        loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=50, drop_last=False)
        print(i)
        for step, (x_batch, y_batch) in enumerate(loader):
            pred = net(x_batch, y_batch)
            pred=pred.view(1,pred.size(0))
            if step==0:
                predd=pred.data.cpu()
            else:
                predd=torch.cat((predd,pred.data.cpu()),1)
        val, index = torch.min(predd, 1)
        pred_labels.append(index.data[0].item())
        
        
        # pred = net(templates, x)
        # val, index = torch.min(pred, 0)
        # pred_labels.append(index.data[0].item())

    y_pred = np.array(pred_labels)
    acc = accuracy_score(y_test, y_pred)
    return acc

# for i in range(x.size(0)//batchsize1+1):
#     print(i)
#     if i==x.size(0)//batchsize1:
#         pred = net(templates[i*batchsize1:-1,:], x[i*batchsize1:-1,:])
#     else:
#         pred = net(templates[i*batchsize1:(i+1)*batchsize1,:], x[i*batchsize1:(i+1)*batchsize1,:])
#     pred=pred.view(1,pred.size(0))
#     if i==0:
#         predd=pred
#     else:
#         predd=torch.cat((predd,pred),1)
#     print(predd.size())
# val, index = torch.min(predd, 0)
# pred_labels.append(index.data[0].item())