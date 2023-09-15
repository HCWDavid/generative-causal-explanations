"""
    make_fig3.py
    
    Reproduces Figure 3 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: global
    explanation for CNN classifier trained on MNIST 3/8 digits.
"""

import numpy as np
import scipy.io as sio
import os
import torch
import util
import plotting
from GCE import GenerativeCausalExplainer
from captum.attr import Saliency

# --- parameters ---
# dataset
dataset = 'medmnist'  # 'mnist' or 'fmnist' or 'medmnist'
type_med = 'blood'
if dataset == 'medmnist':
    dataset = dataset + '_' + type_med  
data_classes = [1, 2]
# classifier
classifier_path = f'./pretrained_models/{dataset}_{"".join([str(i) for i in data_classes])}_classifier'

# vae
K = len(data_classes) - 1
L = 7
train_steps = 800
Nalpha = 25
Nbeta = 100
lam = 0.05
batch_size = 64
lr = 5e-4
# other
randseed = 0
gce_path = f'./pretrained_models/{dataset}_{"".join([str(i) for i in data_classes])}_gce'
retrain_gce = False # train explanatory VAE from scratch
save_gce = False # save/overwrite pretrained explanatory VAE at gce_path
show_heatmap = True

# --- initialize ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if randseed is not None:
    np.random.seed(randseed)
    torch.manual_seed(randseed)
ylabels = range(0,len(data_classes))
from load_mnist import load_med_mnist_classSelect, load_mnist_classSelect, load_fashion_mnist_classSelect
if dataset.startswith('medmnist'):
    X, Y, tridx = load_med_mnist_classSelect('train', data_classes, ylabels, type_med=type_med)
    vaX, vaY, vaidx = load_med_mnist_classSelect('val', data_classes, ylabels, type_med=type_med)
elif dataset == 'mnist':
    X, Y, tridx = load_mnist_classSelect('train', data_classes, ylabels)
    vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes, ylabels)
elif dataset == 'fmnist':
    X, Y, tridx = load_fashion_mnist_classSelect('train', data_classes, ylabels)
    vaX, vaY, vaidx = load_fashion_mnist_classSelect('val', data_classes, ylabels)


# --- load data ---
# from load_mnist import load_mnist_classSelect
# X, Y, tridx = load_mnist_classSelect('train', data_classes, ylabels)
# vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes, ylabels)


ntrain, nrow, ncol, c_dim = X.shape
x_dim = nrow*ncol


# --- load classifier ---
from models.CNN_classifier import CNN
classifier = CNN(len(data_classes), c_dim=c_dim).to(device)
checkpoint = torch.load('%s/model.pt' % classifier_path, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict_classifier'])


# --- train/load GCE ---
from models.CVAE import Decoder, Encoder
if retrain_gce:
    encoder = Encoder(K+L, c_dim, x_dim).to(device)
    decoder = Decoder(K+L, c_dim, x_dim).to(device)
    encoder.apply(util.weights_init_normal)
    decoder.apply(util.weights_init_normal)
    gce = GenerativeCausalExplainer(classifier, decoder, encoder, device)
    traininfo = gce.train(X, K, L,
                          steps=train_steps,
                          Nalpha=Nalpha,
                          Nbeta=Nbeta,
                          lam=lam,
                          batch_size=batch_size,
                          lr=lr)
    if save_gce:
        if not os.path.exists(gce_path):
            os.makedirs(gce_path)
        torch.save(gce, os.path.join(gce_path,'model.pt'))
        sio.savemat(os.path.join(gce_path, 'training-info.mat'), {
            'data_classes' : data_classes, 'classifier_path' : classifier_path,
            'K' : K, 'L' : L, 'train_step' : train_steps, 'Nalpha' : Nalpha,
            'Nbeta' : Nbeta, 'lam' : lam, 'batch_size' : batch_size, 'lr' : lr,
            'randseed' : randseed, 'traininfo' : traininfo})
else: # load pretrained model
    gce = torch.load(os.path.join(gce_path,'model.pt'), map_location=device)
    gce.device = device

# --- compute final information flow ---
I = gce.informationFlow()
Is = gce.informationFlow_singledim(range(0,K+L))
print('Information flow of K=%d causal factors on classifier output:' % K)
print(Is[:K])
print('Information flow of L=%d noncausal factors on classifier output:' % L)
print(Is[K:])

# --- draw sample from each class and visualize---
sample_indices = np.concatenate([np.where(vaY == i)[0][:1] for i in range(len(data_classes))])
x_each_class = torch.from_numpy(vaX[sample_indices])
plotting.explain_sample(x_each_class, save_path='figs/fig3')
    

# --- generate explanation and create figure ---
sample_ind = np.concatenate((np.where(vaY == 0)[0][:4],
                             np.where(vaY == 1)[0][:4]))
x = torch.from_numpy(vaX[sample_ind])

# quick check on x min and max:
print('x min: %f, max: %f' % (x.min(), x.max()))
zs_sweep = [-3., -2., -1., 0., 1., 2., 3.]
Xhats, yhats = gce.explain(x, zs_sweep)
if show_heatmap:
    def classifier_wrapper(*args, **kwargs):
        prob_out, out = gce.classifier(*args, **kwargs)
        return out
    attr_method = Saliency(classifier_wrapper) #.reshape(8 * 8 * 7, 28, 28, 3) 
    #.permute(0,3,1,2)
    reshaped_Xhats =torch.tensor(Xhats).reshape(-1, nrow, ncol, c_dim).permute(0,3,1,2).float().to(device)

    reshaped_yhats = torch.tensor(yhats).reshape(-1).long().to(device)
    attributions = attr_method.attribute(reshaped_Xhats, target=reshaped_yhats)
    # permute(0,2,3,1) back
    
    Xhats_attributions = attributions.permute(0,2,3,1).reshape(*Xhats.shape).detach().cpu().numpy()
    print(Xhats_attributions.shape)

plotting.plotExplanation(Xhats, yhats,heatmaps=Xhats_attributions, save_path='figs/fig3') # it was 1. - Xhats