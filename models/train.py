
import time
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import matplotlib.pyplot as plt

import distrib 
import data_process as dp
import modules as module
import log as log
import view
import Gaussian_mixture as GM
import similarity as sim
import math

import os
from multiprocessing import Process

def train(vi=None,device = '/gpu:1', label_Flag = False, fps = 'Morgan',pro='psa', fps_size = 512,nclass = 2):
    #Data set
    ds = 'HIV'
    prop = pro
    
    #Settings
    
    if fps == 'Maccs':
        fps_size = 167
    
    continue_training = True
    
    if continue_training:
        preTrain = 'preTrain'
    else:
        preTrain = 'scratch'
        
    save_model = False
    batch_size = 256
    _flag_noise = False
    nEpoch = 501
    fps_dim = fps_size
    latent_space = 6
    n_classes = nclass
    layers_dim = np.array([fps_dim//2, fps_dim//8, latent_space])

    data = dp.process_data(ds, fps_type = fps, n_classes=n_classes, nBits = fps_dim, test_size = 600, prop = prop)

    len_train = len(data['fps_train'])
    len_val = len(data['fps_test'])
    
    if fps == 'Morgan':
        val_lr_enc = 0.00001
        val_lr_dec = 0.0001
        val_lr_dis = 0.00001
    elif fps == 'Maccs':
        val_lr_enc = 0.00003
        val_lr_dec = 0.0001
        val_lr_dis = 0.000012
    
    decay_steps = 1000
    thrs_noise = 0.8
    
    
    #Paths
    model_path = os.path.join('/mnt/HDD1/models/', ds+'_'+fps+'_'+prop+'_classes'+str(n_classes))
    preTrain_model_path = os.path.join('/mnt/HDD1/models/', 'Train'+'_'+fps+'_'+prop+'_classes'+str(n_classes))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    #File name
    
    

    with tf.device(device):
        with tf.variable_scope('input'):
            #real and fake image placholders
            real_fps = tf.placeholder(tf.float32, shape = [None, fps_dim], name='real_fps')
            gen_fps = tf.placeholder(tf.float32, shape=[None, fps_dim], name='gen_fps')
            if label_Flag:
                dist_encode = tf.placeholder(tf.float32, shape=[None, layers_dim[2]+n_classes], name='real_z')
            else:
                dist_encode = tf.placeholder(tf.float32, shape=[None, layers_dim[2]], name='real_z')
            labels = tf.placeholder(tf.float32, shape= [None, n_classes], name = 'labels')
            
            is_train_enc = tf.placeholder(tf.bool, name='is_train_enc')
            is_train_dec = tf.placeholder(tf.bool, name= 'is_tain_dec')
            is_train_dis = tf.placeholder(tf.bool, name= 'is_train_dis')

            global_step = tf.placeholder(tf.float32, name='global_step')
            
            lengt = tf.placeholder(tf.float32, name='lengt')
            
            l = tf.placeholder(tf.float32, shape=[None, latent_space], name = 'l')
            
            fp = tf.placeholder(tf.float32, shape = [None, fps_dim], name = 'fp')
            
        lr_dis = tf.train.polynomial_decay(val_lr_dis, global_step, decay_steps, end_learning_rate=0.000001, power=1.0)
        lr_enc = tf.train.polynomial_decay(val_lr_enc, global_step, decay_steps, end_learning_rate=0.000001, power=1.0)
        lr_dec = tf.train.polynomial_decay(val_lr_dec, global_step, decay_steps, end_learning_rate=0.000001, power=2.0)
        # wgan
        real_encode = module.dense_encoder(real_fps, fps_dim, layers_dim, is_train =is_train_enc, reuse=False)
        real_decode = module.dense_decoder(real_encode, fps_dim, layers_dim, is_train = is_train_dec, reuse= False)
        
        if label_Flag:
            real_encode = tf.concat([real_encode, labels], 1)
            
        #Discriminator
        real_result = module.dense_discriminator(dist_encode,layers_dim, is_train =is_train_dis,n_classes=n_classes, reuse =False, label_Flag = label_Flag)
        fake_result = module.dense_discriminator(real_encode, layers_dim, is_train = is_train_dis,n_classes=n_classes, reuse=True, label_Flag = label_Flag)
        
        decode = module.heavside(module.dense_decoder(l, fps_dim, layers_dim, is_train = False, reuse= True))
        encode = module.dense_encoder(fp, fps_dim, layers_dim, is_train =False, reuse=True)
        #Loss calculations
        #dis_loss_real = tf.losses.mean_squared_error(real_result, tf.ones_like(real_result))
        #dis_loss_fake = tf.losses.mean_squared_error(fake_result, -tf.ones_like(fake_result))
        #dis_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(real_result),logits =real_result)
        #dis_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(fake_result), logits = fake_result)
        dis_loss_fake = tf.reduce_mean(fake_result)
        dis_loss_real = - tf.reduce_mean(real_result)
        dis_loss = tf.reduce_mean([dis_loss_real, dis_loss_fake])
        
        enc_loss = -tf.reduce_mean(fake_result) 
        #enc_loss = tf.losses.mean_squared_error(fake_result, tf.ones_like(fake_result))
        dec_loss = tf.losses.mean_squared_error(real_fps, real_decode)
        #dec_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(real_fps - real_decode)))  

        #Trainers
        t_vars = tf.trainable_variables()
        dis_vars = [var for var in t_vars if 'dense_discriminator' in var.name]
        enc_vars = [var for var in t_vars if 'dense_encoder' in var.name]
        dec_vars = [var for var in t_vars if 'dense_decoder' in var.name]
        
        trainer_dis_real = tf.train.AdamOptimizer(learning_rate = lr_dis, beta1=0.9, beta2=0.999,
                                             epsilon=1e-08, use_locking=False,
                                             name='Adam_discriminator').minimize(dis_loss_real, var_list=dis_vars)
        trainer_dis_fake = tf.train.AdamOptimizer(learning_rate = lr_dis, beta1=0.9, beta2=0.999,
                                             epsilon=1e-08, use_locking=False,
                                             name='Adam_discriminator').minimize(dis_loss_fake, var_list=dis_vars)
        
        
        
        trainer_enc = tf.train.AdamOptimizer(learning_rate =lr_enc, beta1=0.9, beta2=0.999,
                                             epsilon=1e-08, use_locking=False,
                                             name='Adam_encoder').minimize(enc_loss, var_list=enc_vars)
        
        trainer_dec = tf.train.AdamOptimizer(learning_rate =lr_dec, beta1=0.9, beta2=0.999,
                                             epsilon=1e-08, use_locking=False,
                                             name='Adam_decoder').minimize(dec_loss, var_list=dec_vars)
        
        d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in dis_vars]
        
        #Accuracy calculations
        less_then_05 = tf.cast(tf.math.less_equal(tf.zeros_like(real_result), real_result), tf.float32)
        count = tf.reduce_sum(less_then_05)
        acc_real = tf.divide(count, lengt)
        acc_fake = tf.divide(tf.reduce_sum(tf.cast(tf.math.less_equal(fake_result,tf.zeros_like(fake_result)), tf.float32)),lengt)
        acc_dis = tf.reduce_mean([acc_real, acc_fake])
        
        acc_enc = 1-acc_fake
        gen_fps = module.heavside(real_decode)
        
        acc_dec = tf.metrics.accuracy(module.heavside(real_fps), module.heavside(real_decode))
   
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.43)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    if continue_training:    
        ckpt = tf.train.latest_checkpoint(preTrain_model_path)
        saver.restore(sess, ckpt)


    #gpu_options = tf.GPUOptions(allow_growth=True)
    #session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
    #                                gpu_options=gpu_options)
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_num = math.floor(len(data['fps_train'])/batch_size) 
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, nEpoch))
    print('start training...')

    dec_iters, dis_iters, enc_iters = 5, 1, 0
    trainLoss_dis, trainAcc_dis = 0, 0
    trainLoss_enc, trainAcc_enc = 0, 0
    trainLoss_dec, trainAcc_dec = 0, 0
    valLoss_dis, valLoss_enc, valLoss_dec = 0,0,0
    valAcc_dis, valAcc_enc, valAcc_dec = 0,0,0  
    for i in range(nEpoch):

        if(trainAcc_dis < 0.505 and trainAcc_enc > 0.98 ):
            enc_iters = 1
            dis_iters = 7
            thrs_noise = 0.9
            _flag_noise = False
        elif(trainAcc_dis < 0.505 and trainAcc_enc < 0.01):
            dis_iters = 1
            enc_iters = 1
            thrs_noise = 0.7
            _flag_noise = False
        else:
            _flag_noise = False
            thrs_noise = 0.95
            dis_iters = 5
            enc_iters = 1
            
        trainLoss_dis, trainAcc_dis = 0, 0
        trainLoss_enc, trainAcc_enc = 0, 0
        trainLoss_dec, trainAcc_dec = 0, 0
        valLoss_dis, valLoss_enc, valLoss_dec = 0,0,0
        valAcc_dis, valAcc_enc, valAcc_dec = 0,0,0   
        
        max_iter = max([dec_iters, dis_iters, enc_iters])
        batch = dp.batch_gen(data['fps_train'], data['labels_train'],
                             batch_size = batch_size,n_dim = layers_dim[2],
                             n_labels = n_classes,
                             label_Flag = label_Flag, dic_iter= max_iter)
        print("Epoch %d" % i)
        train_real_z = distrib.normal_mixture(data['labels_train'],
                                              np.shape(data['labels_train'])[0],
                                              n_dim=layers_dim[2],
                                             n_labels = n_classes)
        val_real_z = distrib.normal_mixture(data['labels_val'],
                                            np.shape(data['labels_val'])[0],
                                            n_dim=layers_dim[2],
                                           n_labels = n_classes)
        if label_Flag:
            train_real_z = np.concatenate((train_real_z,data['labels_train']), axis = 1)
            val_real_z = np.concatenate((val_real_z,data['labels_val']), axis = 1)
              
        for j in range(batch_num):
           

            
            if _flag_noise and np.random.uniform(0,1)> thrs_noise:
                _real_fps = batch['fps'][j] + np.random.normal(0, 0.4, size = np.shape(batch['fps'][j]))
            else:
                _real_fps = batch['fps'][j]
            enc_dict = {real_fps:_real_fps , labels: batch['label'][j], global_step: i,
                        is_train_enc: True, is_train_dis: False}
            
            
          
            for k in range(dis_iters):
                if _flag_noise and np.random.uniform(0,1)>thrs_noise:
                    _real_fps = batch['fps'][j*dis_iters + k] + np.random.normal(0, 0.4, size = np.shape(batch['fps'][j*dis_iters + k]))
                else:
                    _real_fps = batch['fps'][j*dis_iters + k]
                dis_dict ={real_fps: _real_fps, labels: batch['label'][j*dis_iters +k ],
                       dist_encode: batch['real_z'][j*dis_iters +k],global_step: i, 
                       is_train_enc: False, is_train_dis: True}
                
                sess.run([trainer_dis_real], feed_dict= dis_dict)
                sess.run([trainer_dis_fake], feed_dict= dis_dict)
                
            # Update the encoder
            for k in range(enc_iters):
                sess.run([trainer_enc], feed_dict=enc_dict)

            # Update decoder
            for k in range(dec_iters):
                if _flag_noise and np.random.uniform(0,1)>thrs_noise:
                    _real_fps = batch['fps'][j*dis_iters + k] + np.random.normal(0, 0.2, size = np.shape(batch['fps'][j*dis_iters + k]))
                else:
                    _real_fps = batch['fps'][j*dis_iters + k]
                dec_dict = {real_fps: _real_fps, global_step: i, is_train_dec: True, is_train_enc: False}
                sess.run([trainer_dec], feed_dict= dec_dict)
        nom = 10000
        ds_size_nom = np.shape(data['fps_train'])[0]//nom+1
       
        if i%10 == 0:
            l_space = np.zeros([latent_space,np.shape(data['fps_train'])[0]],dtype=np.float32)
            for b in range(ds_size_nom):
                
                l_space[:,b*nom:b*nom+nom] = (np.array(sess.run([encode], feed_dict = {fp: data['fps_train'][b*nom:b*nom+nom,:]}))[0].T)
                
            sample = GM.generate_latent(l_space, np.array(data['labels_train']))
            for j in sample.keys():
                generated_fingerprints = np.array(sess.run([decode], feed_dict = {l: sample[j]})[0])
                for k in range(n_classes):
                    avg_tver, max_tver, min_tver, u_tver, su_tver, nu_tver = sim.tversky(data['fps_test'][k],generated_fingerprints,1,1)
                    arg1 = {'Average_tversky': [avg_tver], 'Max_tversky': [max_tver], 'Min_tversky': [min_tver], 'Useful_tversky': [u_tver], 'Semiuseful_tversky':[su_tver], 'Notuseful_tversky':[nu_tver]}
                    log.log_sim_data(i, arg1, flag = label_Flag, fps = fps, dSet = ds, prop = prop, n_class = n_classes,preTrain=preTrain)
        d = np.zeros([np.shape(data['fps_train'])[0],latent_space],dtype=np.float32)
        for b in range(ds_size_nom):

            train_loss_dict = {real_fps: data['fps_train'][b*nom:b*nom+nom], labels: data['labels_train'][b*nom:b*nom+nom],
                               dist_encode: train_real_z, is_train_dec: False,
                               is_train_enc: False, is_train_dis: False, lengt: len_train}
            val_loss_dict = {real_fps: data['fps_val'], labels: data['labels_val'],
                               dist_encode: val_real_z, is_train_dec: False,
                               is_train_enc: False, is_train_dis: False, lengt: len_val}

            d[b*nom:b*nom+nom,:] = sess.run([encode], feed_dict = {fp: data['fps_train'][b*nom:b*nom+nom,:]} )[0]
            
            trainLoss_dis += sess.run([dis_loss], feed_dict = train_loss_dict)[0]
            trainLoss_enc += sess.run([enc_loss], feed_dict = train_loss_dict)[0]
            trainLoss_dec += sess.run([dec_loss], feed_dict = train_loss_dict)[0]
            
            valLoss_dis += sess.run([dis_loss], feed_dict = val_loss_dict)[0]
            valLoss_enc += sess.run([enc_loss], feed_dict = val_loss_dict)[0]
            valLoss_dec += sess.run([dec_loss], feed_dict = val_loss_dict)[0]

            trainAcc_dis += sess.run([acc_dis], feed_dict = train_loss_dict)[0]
            valAcc_dis += sess.run([acc_dis], feed_dict = val_loss_dict)[0]
            trainAcc_enc += sess.run([acc_enc], feed_dict = train_loss_dict)[0]
            valAcc_enc += sess.run([acc_enc], feed_dict = val_loss_dict)[0]
            trainAcc_dec += sess.run([acc_dec], feed_dict= train_loss_dict)[0][0]
            valAcc_dec += sess.run([acc_dec], feed_dict= val_loss_dict)[0][0]
        
        print(sess.run([lr_dis], feed_dict = {global_step: i}))
        print('Discriminator trainLoss = %f valLoss = %f trainAcc = %f valAcc = %f' % (trainLoss_dis/(ds_size_nom), valLoss_dis/(ds_size_nom), trainAcc_dis/(ds_size_nom+1), valAcc_dis/(ds_size_nom)))
        print('Encoder trainLoss = %f valLoss = %f trainAcc = %f valAcc = %f' % (trainLoss_enc/(ds_size_nom), valLoss_enc/(ds_size_nom), trainAcc_enc/(ds_size_nom), valAcc_enc/(ds_size_nom)))
        print('Decoder trainLoss = %f valLoss = %f trainAcc = %f valAcc = %f' % (trainLoss_dec/(ds_size_nom), valLoss_dec/(ds_size_nom), trainAcc_dec/(ds_size_nom), valAcc_dec/(ds_size_nom)))
        arg = {'Train_loss': [], 'Val_loss': [], 'Train_acc': [], 'Val_acc': []}
        arg['Train_loss'] = [trainLoss_dis/(ds_size_nom), trainLoss_enc/(ds_size_nom), trainLoss_dec/(ds_size_nom)]
        arg['Val_loss'] = [valLoss_dis/(ds_size_nom), valLoss_enc/(ds_size_nom), valLoss_dec/(ds_size_nom)]
        arg['Train_acc'] = [trainAcc_dis/(ds_size_nom), trainAcc_enc/(ds_size_nom), trainAcc_dec/(ds_size_nom)]
        arg['Val_acc'] = [valAcc_dis/(ds_size_nom), valAcc_enc/(ds_size_nom), valAcc_dec/(ds_size_nom)]
        
        
        
        log.log_train_data(i, arg, flag = label_Flag, fps = fps, dSet = ds, prop = prop, n_class = n_classes, preTrain = preTrain)
        if vi!=None:
            d1 = np.empty(np.shape(d)[0]*latent_space//2, dtype = np.float32)
            d2 = np.empty(np.shape(d)[0]*latent_space//2, dtype = np.float32)
            c =  np.empty(np.shape(d)[0]*latent_space//2, dtype = np.int32)
            for h in range(latent_space//2):
                d1[np.shape(d)[0]*h:np.shape(d)[0]*h + np.shape(d)[0]] = d[:,2*h] 
                d2[np.shape(d)[0]*h:np.shape(d)[0]*h + np.shape(d)[0]] = d[:,2*h+1]
                c[np.shape(d)[0]*h:np.shape(d)[0]*h + np.shape(d)[0]] = np.nonzero(data['labels_train'])[1]
            #vi.update(d1_avg, d2_avg, np.nonzero(data['labels_train'])[1])
            vi.update(d1, d2,c)
        if i%50 == 0 and i != 0:
            if save_model:
                saver.save(sess, os.path.join(model_path, 'model' + str(i) + '.ckpt')) 
    sess.close()
    
#if __name__ is '__main__':
print('Start')
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #train(vi0,r'/gpu:0', True, 'Maccs','alogp', 512,nclass=3)
    
p0 = Process(target=train, args=(None,r'/gpu:0', True, 'Morgan','psa',512,2)) 
p1 = Process(target=train, args=(None,r'/gpu:0', True, 'Morgan','psa',512,3))
p2 = Process(target=train, args=(None,r'/gpu:0', True, 'Maccs','psa',512,5))

p3 = Process(target=train, args=(None,r'/gpu:0', True, 'Morgan','alogp',512,2)) 
p4 = Process(target=train, args=(None,r'/gpu:0', True, 'Morgan','alogp',512,3)) 
p5 = Process(target=train, args=(None,r'/gpu:0', True, 'Maccs','alogp',512,5))

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#p0.start()
#p1.start()
p2.start()
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#p3.start()
#p4.start()
#p5.start()


#p0.join()
#p1.join()
p2.join()
#p3.join()
#p4.join()
#p5.join()
