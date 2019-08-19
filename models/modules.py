import tensorflow as tf
import numpy as np

## Activation functions

#Leaky relu
def lrelu(x, n = None, leak=0.1): 
    return tf.maximum(x, x*leak, name=n) 

#Heavside
def heavside(x):
    return tf.maximum(0.0,tf.sign(x))

#Linear
def linear(x):
    return x

def denseLayer(input, in_size, out_size, batch_norm = False, activation = lrelu, is_train = True, reuse = False, reg = 0.001 ):   
    w = tf.get_variable('w', shape=[in_size, out_size], dtype=tf.float32,
                     initializer=tf.truncated_normal_initializer(stddev=0.02),
                     regularizer=tf.keras.regularizers.l2(l=reg))
    b = tf.get_variable('b', shape=[out_size], dtype=tf.float32,
                     initializer=tf.constant_initializer(0.0),
                     regularizer=tf.keras.regularizers.l2(l=reg))
    dense = tf.add(tf.matmul(input, w), b, name='mat_mul_dense1')
    
    #dense = activation(dense)
    
    if batch_norm:
        dense = tf.contrib.layers.batch_norm(dense, is_training = is_train, epsilon=1e-5, decay = 0.9,
                                       updates_collections=None)  
 
   
   
    return dense

def conv1dLayer(input, nfilter, kernel_size, stride, reuse = False):
    
    out = tf.layers.conv1d(inputs = input, filters = nfilter, kernel_size = kernel_size,
                        strides = stride, padding = 'valid', data_format ='channels_last',
                        dilation_rate = 1, activation = None,
                        use_bias = True, kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
                        bias_initializer = tf.constant_initializer(0.0),
                        kernel_regularizer = tf.keras.regularizers.l2(l=0.005),
                        name='conv1d_layer', reuse = reuse)
    return out
def dense_encoder(input, fps_dim, layers_dim, is_train, reuse=False):
    

    with tf.variable_scope('dense_encoder') as scope:
        if reuse:
            scope.reuse_variables()
        with tf.variable_scope('Layer0'):
            out_layer0 = denseLayer(input, fps_dim, layers_dim[0], is_train = is_train, batch_norm = True, reg = 0.001)
            out_layer0 = lrelu(out_layer0, leak = 0.1)
            #out_layer0 = tf.nn.dropout(out_layer0, keep_prob = 0.8)
        with tf.variable_scope('Layer1'):
            out_layer1 = denseLayer(out_layer0, layers_dim[0], layers_dim[1], is_train = is_train, batch_norm = True, reg = 0.001)
            out_layer1 = lrelu(out_layer1, leak = 0.1)
        with tf.variable_scope('Layer2'):
            out_layer2 = denseLayer(out_layer1, layers_dim[1], layers_dim[2], is_train = is_train, batch_norm = True, reg = 0.02)
            #out_layer2 = tf.math.minimum(out_layer2,3)
    return out_layer2
        
def dense_decoder(input,fps_dim, layers_dim, is_train, reuse=False):
   
    with tf.variable_scope('dense_decoder') as scope:
        if reuse:
            scope.reuse_variables()
        with tf.variable_scope('Layer0'):
            out_layer0 = denseLayer(input, layers_dim[2], layers_dim[1], is_train = is_train,batch_norm = True)
            out_layer0 = lrelu(out_layer0 , leak = 0.1)
           # out_layer0 = tf.nn.dropout(out_layer0, keep_prob = 0.9)
        with tf.variable_scope('Layer1'):
            out_layer1 = denseLayer(out_layer0, layers_dim[1], layers_dim[0], is_train = is_train,batch_norm = True)
            out_layer1 = lrelu(out_layer1, leak=0.1)
        with tf.variable_scope('Layer2'):
            out_layer2 = denseLayer(out_layer1, layers_dim[0], fps_dim , is_train = is_train, batch_norm = False)
            out_layer2 = tf.math.tanh(out_layer2)
    return out_layer2

def dense_discriminator(input, layers_dim, is_train, n_classes, reuse = False, label_Flag = False):
    with tf.variable_scope('dense_discriminator') as scope:
        if reuse:
            scope.reuse_variables()
            
        if label_Flag:
            in_size = layers_dim[2] + n_classes
        else:
            in_size = layers_dim[2]
        with tf.variable_scope('Layer0'):
            out_layer0 = denseLayer(input, in_size, layers_dim[2]*10, is_train = is_train, batch_norm = False)
            out_layer0 = lrelu(out_layer0)
           # out_layer0 = tf.nn.dropout(out_layer0, keep_prob = 0.9)
        with tf.variable_scope('Layer1'):
            out_layer1 = denseLayer(out_layer0, layers_dim[2]*10, layers_dim[2]*5, is_train = is_train, batch_norm = False)
            out_layer1 = lrelu(out_layer1)
        with tf.variable_scope('Layer2'):
            out_layer2 = denseLayer(out_layer1, layers_dim[2]*5, 1, is_train = is_train, batch_norm = False)
            #out_layer2 = tf.math.sigmoid(out_layer2)
            #out_layer2 = tf.math.tanh(out_layer2)
    return out_layer2        

def cnn_encoder(input, layers_dim, is_train_enc, reuse=False):

    with tf.variable_scope('encode') as scope:
        if reuse:
            scope.reuse_variables()
        
        #Convolution module M
        with tf.variable_scope('first_layer'):
            out_layer1 = conv1dLayer(input, nfilter=64, kernel_size = 128, stride = 6, reuse = reuse)
            out_layer1 = lrelu(out_layer1, leak = 0.0)
            
        with tf.variable_scope('second_layer'):
            out_layer2 = conv1dLayer(out_layer1, nfilter=64, kernel_size = 64, stride = 4, reuse = reuse)
            out_layer2 = lrelu(out_layer2, leak = 0.0)
        
        
        out_layer3 = tf.layers.dropout(out_layer2, rate=0.1)
        
        with tf.variable_scope('fourth_layer'):
            out_layer4 = conv1dLayer(out_layer3, nfilter=92, kernel_size = 32, stride = 1, reuse = reuse)
            out_layer4 = lrelu(out_layer4, leak = 0.0)
        
        with tf.variable_scope('fifth_layer'):
            out_layer5 = conv1dLayer(out_layer4, nfilter=128, kernel_size =8, stride = 1, reuse = reuse)
            out_layer5 = lrelu(out_layer5, leak = 0.0)
            out_layer5 = tf.layers.batch_normalization(out_layer5)        
            out_layer5 = tf.layers.dropout(out_layer5, rate=0.1)

        #Flattening 
        out_flat = tf.layers.flatten(out_layer5, name = 'encode_flatten')
        
        # Input layer (layers_dim[0]) encoding layer
        with tf.variable_scope('sixth_layer'):
            out_layer6 = denseLayer(out_flat, out_flat.get_shape().as_list()[1], 256)
            
        with tf.variable_scope('seventh_layer'):
            out_layer7 = denseLayer(out_layer6, 256, 64)

        with tf.variable_scope('latent_layer'):
            out_latent = denseLayer(out_layer7, 64, 10, activation = tf.nn.softmax)
        sizes_encoder = {}
        sizes_encoder['enc_input'] = input.get_shape().as_list()
        sizes_encoder['cnn1_out'] = out_layer1.get_shape().as_list()
        sizes_encoder['cnn2_out'] = out_layer2.get_shape().as_list()

        sizes_encoder['cnn3_out'] = out_layer4.get_shape().as_list()
        sizes_encoder['cnn4_out'] = out_layer5.get_shape().as_list()

        sizes_encoder['flatten'] = out_flat.get_shape().as_list()
        sizes_encoder['dense0'] = out_layer6.get_shape().as_list()
        sizes_encoder['dense1'] = out_latent.get_shape().as_list()
                                                        
        return out_latent, sizes_encoder