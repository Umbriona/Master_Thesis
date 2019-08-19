#%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from matplotlib.animation import FuncAnimation 

import data_process as dp

class View_scatter:
    targets = ['1','2','3','4','5','6','7','8','9','10']
    rgb = [(1.0, 0.0, 0.0),(0.0, 1.0, 0.0),(0.0, 0.0, 1.0),(1.0, 1.0, 0.0),(1.0, 0.0, 1.0),
           (0.0, 1.0, 1.0),(0.0, 0.0, 0.0),(0.5, 0.2, 1.0),(1.0, 0.5, 0.7),(0.6, 1.0, 0.2)]
    #rgb = [tuple(np.random.uniform(0,1, size = 3))  for i in range(10)]
    bottom = -4.0
    top = 4.0
    def __init__(self, xlabel = 'Component 0,2,4',
        ylabel = 'Component 1,3,5',
        title = 'Latent space clustering',
        prop = 'psa'):
        
        plt.ion()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.fig = plt.figure(figsize = (8,8))
        #self.fig.ion()
        self.ax = self.fig.add_subplot(1,1,1) 
        self.ax.set_xlabel(self.xlabel, fontsize = 15)
        self.ax.set_ylabel(self.ylabel, fontsize = 15)
        self.ax.set_title(self.title, fontsize = 20)
        self.prop = prop
        self.targets = []
        self.ax.grid()    
        self.hl = plt.scatter([], [], marker = 'o')
        plt.show()
    

    def update(self, x_list,y_list, labels, nlabel = 10, prop=[], path=None):
        self.ax.clear()
        self.ax.set_xlabel(self.xlabel, fontsize = 15)
        self.ax.set_ylabel(self.ylabel, fontsize = 15)
        self.ax.set_title(self.title, fontsize = 20)
        self.ax.grid() 
        legends = [self.prop + ' ' + str(prop[i]) + '-' + str(prop[i+1]) for i in range(0,len(prop),2)]
        for i in range(nlabel):
            self.hl = plt.scatter(x_list[labels==i],y_list[labels==i], marker = 'o', c =self.rgb[i])
        self.ax.legend(legends)
        self.ax.set_xlim(self.bottom, self.top)
        self.ax.set_ylim(self.bottom, self.top)
        if path != None:
            print('Saving...')
            self.fig.savefig(path)
        
        
        
class View_plot:
    
    def __init__(self, label_mode = "No"):
        path, files, data_sets, fingerprints = dp.data_path_parse()
        
        self.path_train_data = os.path.join(path['Path to training data'], ''.join(data_sets.keys()))
        if label_mode == 'No':
            index = 1
            title_str = 'Telemetry for unlabeld training'
        elif label_mode == 'Yes':
            index = 0
            title_str = 'Telemetry for labeld training'
        else:
            raise ValueError('label_mode must be "Yes" or "No"')
        self.file_name = files['File name'][index]
        
        self.fig = plt.figure(figsize = (8,8))
        self.ax = self.fig.add_subplot(1,1,1) 
        self.ax.set_xlabel('Epoch', fontsize = 15)
        self.ax.set_ylabel('Loss', fontsize = 15)
        self.ax.set_title(title_str, fontsize = 20)
        #self.ax.legend(['Discriminator', 'Encoder', 'Decoder'])

    def train_data_plot(self, epochs=None):
        
        f_l = open(os.path.join(self.path_train_data, self.file_name),'r')

        string_labeld_data = f_l.read()
        list_labeld_data = string_labeld_data.split('\n')

        l = [np.float_(i.split(';')) for i in list_labeld_data[:epochs]]
        l = np.array(l)
        
        for i in range(len(l[0,:])):
            ax.plot(l[:,0],l[:,i])
        
