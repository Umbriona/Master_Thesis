import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt


def get_gausse_model(train_data):
    cov = np.cov(train_data)
    mean = [np.mean(train_data[0,:]), np.mean(train_data[1,:])]
    return [mean, cov]

def fit_models(latent_space, labels):
    z_dim = np.shape(latent_space)[0]
    label_values = np.nonzero(labels)[1]
    
    label_values_unique = np.unique(label_values)
    gmm_list = []
    for i in label_values_unique:
        shape = np.shape(latent_space[:,label_values==i])
        data = latent_space[:,label_values==i]
        for j in range(z_dim//2):
            data = latent_space[:,label_values==i]
            data = data[j*2:j*2+2]
            gmm_list.append(get_gausse_model(data))
    return gmm_list 

   
    
def sample_models(gmm_list, z_dim, lable_names = None, sample_size = 2000):
    rgb = [(1.0, 0.0, 0.0),(0.0, 1.0, 0.0),(0.0, 0.0, 1.0),(1.0, 1.0, 0.0),(1.0, 0.0, 1.0),
           (0.0, 1.0, 1.0),(0.0, 0.0, 0.0),(0.5, 0.2, 1.0),(1.0, 0.5, 0.7),(0.6, 1.0, 0.2)]
    sample_dict = {}
    if lable_names == None:
        lable_names = [ str(num) for num in range(len(gmm_list)//z_dim)]
    [sample_dict.update({i : []} )for i in lable_names]   
    for num in range(len(gmm_list)//z_dim):
        tmp = np.random.multivariate_normal(gmm_list[z_dim*num][0],gmm_list[z_dim*num][1],sample_size)
        tmp = np.concatenate((tmp, np.random.multivariate_normal(gmm_list[z_dim * num + 1][0],gmm_list[z_dim * num + 1][1],sample_size)), axis = 1)
        tmp = np.concatenate((tmp, np.random.multivariate_normal(gmm_list[z_dim * num + 2][0],gmm_list[z_dim * num + 2][1],sample_size)), axis = 1)
        sample_dict[lable_names[num]] = tmp.astype(np.float32)
    return sample_dict

def generate_latent(latent_space, labels):
    z_dim = np.shape(latent_space)[0]//2
    gmm_list = fit_models(latent_space, labels)
    return sample_models(gmm_list, z_dim)
                       