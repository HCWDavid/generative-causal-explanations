import os
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import TissueMNIST, BloodMNIST
def load_med_mnist_classSelect(dataset,class_use,newClass, type_med='tissue'):

    X, Y, idx = load_med_mnist_idx(dataset,type_med=type_med)
    # num_classes = np.max(np.unique(Y)) - np.min(np.unique(Y)) + 1 # total number of classes
    assert np.min(class_use) >= np.min(np.unique(Y)) and np.max(class_use) <= np.max(np.unique(Y)), 'class_use must be a subset of the dataset classes'
    class_idx_total = np.zeros((0,0))
    Y_use = Y

    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        print(len(class_idx_total))
        count_y = count_y +1

    class_idx_total = np.sort(class_idx_total).astype(int)
    # print(np.max(X), np.min(X))
    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    # squeeze the channel dimension in Y
    Y = np.squeeze(Y)
    return X,Y,idx

def load_med_mnist_idx(data_type, type_med='tissue'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # in trainning phase, the data shape order was permuted "batch_images_torch = batch_images_torch.permute(0,3,1,2).float()", we want to keep consistent with it:
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Rearrange dimensions to PyTorch convention
    
        # Add other transformations if needed
        ])
    if type_med == 'tissue':
        dataset_train = TissueMNIST(split='train', download=False, root='./datasets/medmnist', transform=transform)
        dataset_val = TissueMNIST(split='val', download=False, root='./datasets/medmnist', transform=transform)
        dataset_test = TissueMNIST(split='test', download=False, root='./datasets/medmnist', transform=transform)
    elif type_med == 'blood':
        dataset_train = BloodMNIST(split='train', download=True, root='./datasets/medmnist', transform=transform)
        dataset_val = BloodMNIST(split='val', download=True, root='./datasets/medmnist', transform=transform)
        dataset_test = BloodMNIST(split='test', download=True, root='./datasets/medmnist', transform=transform)
    else:
        raise ValueError(f'type_med {type_med} is not supported')
    # dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    if data_type == 'train':
        dataloader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    elif data_type == 'val':
        dataloader = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=False)
    elif data_type == 'test':
        dataloader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    for batch in dataloader:
        X, y = batch

    X = X.numpy().astype(float)
    y = y.numpy().astype(int)
    # print(y)
    idxUse = np.arange(0, y.shape[0])
    seed = 73054772
    np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    # np.random.seed(seed)
    # np.random.shuffle(idxUse)

    perm = np.random.permutation(len(X))

    # Shuffle all arrays according to the permutation
    X = X[perm]
    y = y[perm]
    idxUse = idxUse[perm]
    return X, y, idxUse


def load_mnist_idx(data_type):
       data_dir = 'datasets/mnist/'
       fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
       fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       trY = loaded[8:].reshape((60000)).astype(np.float)
       fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
       fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       teY = loaded[8:].reshape((10000)).astype(np.float)
       trY = np.asarray(trY)
       teY = np.asarray(teY)
       if data_type == "train":
           X = trX[0:50000,:,:,:]
           y = trY[0:50000].astype(np.int)
       elif data_type == "test":
           X = teX
           y = teY.astype(np.int)
       elif data_type == "val":
           X = trX[50000:60000,:,:,:]
           y = trY[50000:60000].astype(np.int)
       idxUse = np.arange(0,y.shape[0])
       seed = 547
       np.random.seed(seed)
       np.random.shuffle(X)
       np.random.seed(seed)
       np.random.shuffle(y)
       np.random.seed(seed)
       np.random.shuffle(idxUse)

       return X/255.,y,idxUse
   
def load_mnist_classSelect(data_type,class_use,newClass):
    
    X, Y, idx = load_mnist_idx(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    return X,Y,idx

def load_fashion_mnist_idx(data_type):
    import mnist_reader
    data_dir = 'datasets/fmnist/'
    if data_type == "train":
        X, y = mnist_reader.load_mnist(data_dir, kind='train')
    elif data_type == "test" or data_type == "val":
        X, y = mnist_reader.load_mnist(data_dir, kind='t10k')
        if data_type == "test":
            X = X[:4000,:]
            y = y[:4000]
        else:
            X = X[4000:,:]
            y = y[4000:]
    X = X.reshape((X.shape[0],28,28,1))        
    X = X.astype(np.float)
    y = y.astype(np.int)
    
    idxUse = np.arange(0,y.shape[0])
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    np.random.seed(seed)
    np.random.shuffle(idxUse)
    return X/255.,y,idxUse

def load_fashion_mnist_classSelect(data_type,class_use,newClass):
    
    X, Y, idx = load_fashion_mnist_idx(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    return X,Y,idx
    

def load_svhn_idx(data_type):
       data_dir = 'datasets/SVHN/'
       if data_type == "train":
           data = sio.loadmat(data_dir + 'train_32x32.mat')
           X = data['X'].astype(np.float)
           y = data['y'].astype(np.int)
       elif data_type == "val": 
           data = sio.loadmat(data_dir + 'test_32x32.mat')
           X = data['X']
           X = X[:,:,:,:10000].astype(np.float)
           y = data['y']
           y = y[:10000].astype(np.int)
       elif data_type == "test": 
           data = sio.loadmat(data_dir + 'test_32x32.mat')
           X = data['X']
           X = X[:,:,:,10000:].astype(np.float)
           y = data['y']
           y = y[10000:].astype(np.int)
       
       X = X.transpose(3,2,0,1)
       zero_idx = np.where(y == 10)[0]
       y[zero_idx] = 0

       idxUse = np.arange(0,y.shape[0])
       seed = 547
       np.random.seed(seed)
       np.random.shuffle(X)
       np.random.seed(seed)
       np.random.shuffle(y)
       np.random.seed(seed)
       np.random.shuffle(idxUse)

       return X/255.,y,idxUse
   
def load_svhn_classSelect(data_type,class_use,newClass):
    
    X, Y, idx = load_svhn_idx(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    return X,Y,idx
