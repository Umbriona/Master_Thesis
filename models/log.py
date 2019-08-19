import scipy.misc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import data_process as dp
import matplotlib.pyplot as plt
import os

def log_scatter(latent_vectors, labels, epoch):
    
    
    pca = PCA(n_components=2)
   
    features = ['Node' + str(num) for num, i in latent_vectors[0,:]]
    df = pd.DataFrame(latent_vectors, columns=features)
    df['label'] = np.nonzero(labels)[1]
    
    path, files, data_bases = dp.data_path_parse()
    
    if FLAG_label:
        path_part_1 = path['Path to cluster data'][0]
    else:
        path_part_1 = path['Path to cluster data'][1]
        
    _path = os.path.join(part_path_1, ''.join(data_sets.keys()))
    if not os.path.exists(_path):
        os.makedirs(_path) 
    df.to_csv(os.path.join(_path, files['File name'] + str(epoch) + '.csv'))
    return 0

def log_train_data(i, arg, flag = True, fps = 'Morgan', dSet = 'LD50', prop = 'psa', n_class = 2,preTrain='pretrain'):        
    path, files, data_sets, fingerprints = dp.data_path_parse()
    dset_list = []
    for j in data_sets.keys():
        dset_list.append(str(j))
    
    _path = os.path.join(path['Path to training data'][0], dSet)
    if flag:
        file_name = files['File name'][0] + '_' + fps + '_' + prop + '_classes' +str(n_class)+'_'+preTrain
    else:
        file_name = files['File name'][1] + '_' + fps + '_' + prop + '_classes' +str(n_class)+'_' +preTrain
        
    if not os.path.exists(_path):
        os.makedirs(_path)
    if not os.path.isfile(os.path.join(_path,file_name)) or i == 0:
        fo = open(os.path.join(_path,file_name), 'w+')
    else:
        fo = open(os.path.join(_path,file_name), 'a') 
    
    _str ='%d'
    for key, value in arg.items():
        for j in value:
            _str = _str + ';%f'
    _str = _str + '\n' 
    tup = tuple([i]) + tuple(arg['Train_loss'] + arg['Val_loss'] + arg['Train_acc'] + arg['Val_acc']) 
    fo.write(_str % tup)
    fo.close()
    return 0

def log_sim_data(i, arg, flag = True, fps = 'Morgan', dSet = 'LD50', prop = 'psa', n_class = 2, preTrain='preTrain'):        
    path, files, data_sets, fingerprints = dp.data_path_parse()
      
    _path = os.path.join(path['Path to training data'][0], dSet)
    if flag:
        file_name = files['File name'][2] + '_' + fps + '_' + prop + '_classes' +str(n_class)+'_' + preTrain
    else:
        file_name = files['File name'][3] + '_' + fps + '_' + prop + '_classes' +str(n_class)+'_' + preTrain
        
    if not os.path.exists(_path):
        os.makedirs(_path)
    if not os.path.isfile(os.path.join(_path,file_name)) or i == 0:
        fo = open(os.path.join(_path,file_name), 'w+')
    else:
        fo = open(os.path.join(_path,file_name), 'a') 
    
    _str ='%d'
    for key, value in arg.items():
        for j in value:
            _str = _str + ';%f'
    _str = _str + '\n' 
    tup = tuple([i]) + tuple(arg['Average_tversky'] + arg['Max_tversky'] + arg['Min_tversky'] + arg['Useful_tversky'] + arg['Semiuseful_tversky'] + arg['Notuseful_tversky']) 
    fo.write(_str % tup)
    fo.close()
    return 0
                         
                         
              