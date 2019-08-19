import numpy as np
import math
import scipy.misc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from utils import *
import pandas as pd

# Molecule data
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys 
from rdkit.Chem.Fingerprints import FingerprintMols

import os

from multiprocessing.dummy import Pool as ThreadPool 

import distrib

import matplotlib.pyplot as plt

## Data extraction
def data_path_parse():
    ## Path and parse

    paths = {}
    files = {}
    data_sets = {}
    fingerprints = {}
    tmp = []
    f =open('config.txt','r')
    string = f.read()
    list_1 = string.split('##')


    [tmp.append(x.split(': ')[0].rstrip()) for x in list_1[1].split('#')[1:]]
    [paths.update({i : []} )for i in tmp]
    [[paths[x.split(': ')[0].rstrip()].append(i.rstrip()) for i in x.split(', ')[1:]] for x in list_1[1].split('#')[1:]]                                      
    tmp=[]
    [tmp.append(x.split(': ')[0].rstrip()) for x in list_1[2].split('#')[1:]]
    [files.update({i : []} )for i in tmp]
    [[files[x.split(': ')[0].rstrip()].append(i.rstrip()) for i in x.split(', ')[1:]] for x in list_1[2].split('#')[1:]]
    tmp = []
    [tmp.append(x.split(': ')[0].rstrip()) for x in list_1[3].split('#')[1:]]
    [data_sets.update({i : []} )for i in tmp]
    [[data_sets[x.split(': ')[0].rstrip()].append(i.rstrip()) for i in x.split(', ')[1:]] for x in list_1[3].split('#')[1:]]
    tmp = []
    [tmp.append(x.split(': ')[0].rstrip()) for x in list_1[4].split('#')[1:]]
    [fingerprints.update({i : []} )for i in tmp]
    [[fingerprints[x.split(': ')[0].rstrip()].append(i.rstrip()) for i in x.split(', ')[1:]] for x in list_1[4].split('#')[1:]]
    f.close()
    return paths, files, data_sets, fingerprints

def morgan_fps(mol, n=2048):
    fps_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=n)
    fps_morgan = np.array(list(fps_morgan.ToBitString())).astype('float32')
    #print('Morgan', (sum(fps_morgan)/fps_size))
    return fps_morgan
    
def maccs_fps(mol):
    fps_maccs = MACCSkeys.GenMACCSKeys(mol)#, nBits =1024)
    return np.array(list(fps_maccs.ToBitString())).astype('float32')

def topological_fps(mol, n= 2048):
    fps_top = FingerprintMols.FingerprintMol(mol, minPath=1, 
                                                maxPath=5, 
                                                fpSize=n,
                                                bitsPerHash=2,
                                                useHs=False,
                                                tgtDensity = 0.0,
                                                minSize = 128 
                                                )
    fps_top = np.array(list(fps_top.ToBitString())).astype('float32')
    #print('Top', (sum(fps_top)/fps_size))
    return fps_top
def scale_data(value, units):
    value[units == 'nM'] *= 0.001
    return value
    
def process_data_np(ds_name = 'LD50_Test_data'):
    #print('process_data_np is depricated use process_data instead')
    paths, files, data_sets = data_path_parse()

    
    data_dir = paths['Path to training set'][0]
    ds_name = data_sets['LD50'][0]
    df = pd.read_csv(os.path.join(data_dir, ds_name + '.csv'))
    pool = ThreadPool(pool_size)
    fps_morgan, fps_maccs, fps_top, mol = [], [], [], []
    fps_dict = {}
    l = np.array(df.value, dtype = np.float32)

    lable_scale = np.arange(int(min(l)), int(max(l)),int((max(l)-min(l))/10))
    for i in range(len(lable_scale)-1):
        if(i < len(lable_scale)-2):
            l[np.logical_and(np.greater_equal(l, lable_scale[i]),np.less_equal(l, lable_scale[i+1]))] = i
        else:
            l[(np.greater_equal(l, lable_scale[i]))] = i
    labels = [np.zeros([10], dtype = np.float32) for i in range(len(l))]
    for i in range(len(l)):
        labels[i][int(l[i])] = 1
    n =[2048, 512]    
    mol = [Chem.MolFromSmiles(x) for x in df.smiles]

    fps_morgan1 = [morgan_fps(x, n=n[0]) for x in mol]
    fps_morgan2 = [morgan_fps(x, n=n[1]) for x in mol]
    
    fps2 = np.array(fps_morgan2)
    fps2 = np.reshape(fps2, [len(mol), n[1]])
    fps2 =fps2 - (np.equal(fps2,np.zeros(np.shape(fps2))))

    fps_top = [topological_fps(x) for x in mol]
    fps1 = np.array([fps_morgan1, fps_top], dtype = np.float32)
    fps1 = np.reshape(fps1,[len(mol),n[0],2])
    fps1 =fps1 - (np.equal(fps1,np.zeros(np.shape(fps1)))) # from [0,1] -> [-1,1]
    
    fps3 = [maccs_fps(x) for x in mol]
    fps3 = np.array(fps3, dtype = np.float32)
    fps3 = fps3 - (np.equal(fps3,np.zeros(np.shape(fps3))))
    fps_dict['fps1_train_np'], fps_dict['fps1_test_np'], fps_dict['labels1_train_np'], fps_dict['labels1_test_np'] = train_test_split(fps1,
                                                                      labels,
                                                                      test_size=0.20,
                                                                      random_state=42)
    fps_dict['fps2_train_np'], fps_dict['fps2_test_np'], fps_dict['labels2_train_np'], fps_dict['labels2_test_np'] = train_test_split(fps2,
                                                                      labels,
                                                                      test_size=0.20,
                                                                      random_state=42)
    
    fps_dict['fps3_train_np'], fps_dict['fps3_test_np'], fps_dict['labels3_train_np'], fps_dict['labels3_test_np'] = train_test_split(fps3,
                                                                      labels,
                                                                      test_size=0.20,
                                                                      random_state=42)
    
    return fps_dict

def process_data(ds, fps_type = 'Morgan', nBits = 512, n_classes = 5, test_size = 300, prop = 'psa'):
    
    
    
    paths, files, data_sets, fingerprints = data_path_parse() 
    print('The data set that is being used is ' + ''.join(data_sets.keys()))
    data_dir = paths['Path to training set'][0]
    ds_name = data_sets[ds][0]
    df = pd.read_csv(os.path.join(data_dir, ds_name ))

    #df[prop] = scale_data(df.value.values, df.Unit.values) #scaling data to have same units
    fps_morgan, fps_maccs, fps_top, mol = [], [], [], []
    fps_dict = {}
    df = df.sort_values(by=[prop])
    #l = np.array(df.value, dtype = np.float32)
    
    l = np.empty(len(df.Unit), dtype = np.float32)
    #l[df['Type'].values == 'N'] = 0.0
    #l[df['Type'].values == 'Y'] = 2.0
    
    #label_scale = np.arange(int(min(l)), int(max(l)),int((max(l)-min(l))/n_classes))
    #print(label_scale)
    #for i in range(np.size(label_scale)):
    #    if(i < np.size(label_scale)-1):
    #        l[np.logical_and(np.greater_equal(l, label_scale[i]),np.less_equal(l, label_scale[i+1]))] = i
    #    else:
    #        l[(np.greater_equal(l, label_scale[i]))] = i
    
    for h in range(n_classes):
        l[h*len(l)//n_classes:h*len(l)//n_classes+len(l)//n_classes] = h

    df['label'] = l
    df_scaled_test = df.loc[df['label'] == 0][:1]
    for i in range(test_size):
        index = np.random.randint(int(df.loc[df['label']==0]['label'].count()))
        df_scaled_test = df_scaled_test.append(df.loc[df['label'] == 0][index-1:index])

    
    df_scaled = df.loc[df['label'] == 0][:30000]
    for i in range(n_classes-1):
        df_tmp_test = df.loc[df['label'] == i+1][:1]
        for _ in range(test_size):
            index = np.random.randint(int(df.loc[df['label']==i+1]['label'].count()))
            df_tmp_test = df_tmp_test.append(df.loc[df['label'] == i+1][index-1:index])

        df_tmp = df.loc[df['label'] == i+1][:30000]
        df_scaled_test = pd.concat([df_scaled_test, df_tmp_test],ignore_index=True)
        df_scaled = pd.concat([df_scaled, df_tmp],ignore_index=True)
   
    l_scaled = list(df_scaled['label'])
    labels = [np.zeros([n_classes], dtype = np.float32) for i in range(len(l_scaled))]
    for i in range(len(l_scaled)):
        labels[i][int(l_scaled[i])] = 1

 
    mol = [Chem.MolFromSmiles(x) for x in df_scaled.smiles]
    mol_test = [Chem.MolFromSmiles(x) for x in df_scaled_test.smiles]
  
    if fps_type == 'Morgan':
        fps = [morgan_fps(x, n=nBits) for x in mol]
        fps_test = [morgan_fps(x, n=nBits) for x in mol_test]
    elif fps_type == 'Top':
        fps = [topological_fps(x, n =nBits) for x in mol]
        fps_test = [topological_fps(x, n =nBits) for x in mol_test]
    elif fps_type == 'Maccs':
        fps = [maccs_fps(x) for x in mol]
        fps_test = [maccs_fps(x) for x in mol_test]
    else:
        print('fps_type must be ither Morgan Top or Maccs')
        return 0
    
    fps_test = np.array(fps_test, dtype = np.float32)
    fps = np.array(fps, dtype = np.float32)
    fps =fps - (np.equal(fps,np.zeros(np.shape(fps))))
    
    #fps, fpsTest, labels, labelTest = train_test_split(fps, labels, test_size=test_size)
    fpsTrain, fpsVal, labelTrain, labelVal = train_test_split(fps, labels, test_size=0.1)
    fps_dict['fps_train'] = fpsTrain
    fps_dict['fps_val'] = fpsVal
    fps_dict['labels_train'] = labelTrain
    fps_dict['labels_val'] = labelVal
    fps_dict['fps_test'] = []
    fps_dict['labels_test'] = []
    fps_dict['prop'] = []
    for i in range(n_classes):
        fps_dict['fps_test'].append(fps_test[df_scaled_test['label'].values == i,:])
        fps_dict['labels_test'].append(df_scaled_test[df_scaled_test['label'].values == i].values)
        fps_dict['prop'].append(min(df_scaled.loc[df_scaled['label']==i][prop]))
        fps_dict['prop'].append(max(df_scaled.loc[df_scaled['label']==i][prop]))
    return fps_dict

def batch_gen(fps, label, batch_size = 32,n_dim = 10,n_labels = 10, label_Flag = False, FLAG = True, dic_iter=5):   
    
    num_batch_epoch = math.floor(len(fps)/batch_size) * dic_iter
    fps_batch_list = []
    label_batch_list = []
    real_z_batch_list = []
    if FLAG:
        batch_index = np.random.randint(0, len(label), (batch_size, num_batch_epoch))  
    else:
        batch_index = np.arange(batch_size*num_batch_epoch).reshape((batch_size,num_batch_epoch))

    for i in range(num_batch_epoch):
   
        tmp1, tmp2 = [], []
        tmp1 = [fps[x] for x in batch_index[:,i]]
        tmp2 = [label[x] for x in batch_index[:,i]]
        
        label_batch_list.append(np.array(tmp2))
        if label_Flag:
            real_z = distrib.normal_mixture(np.array(tmp2), batch_size, n_dim=n_dim, n_labels = n_labels, x_var = 0.5, y_var = 0.1)
            #plt.scatter(real_z[:,0], real_z[:,1])
            real_z = np.concatenate((real_z,tmp2), axis = 1)

        else:
            real_z = distrib.normal(batch_size = batch_size, n_dim = n_dim)
        fps_batch_list.append(np.array(tmp1))
        real_z_batch_list.append(np.array(real_z))
        
    batch={'fps': fps_batch_list, 'label': label_batch_list, 'real_z': real_z_batch_list}
    return batch