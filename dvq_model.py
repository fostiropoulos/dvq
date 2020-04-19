"""

Official Implementation of Depthwise Vector Quantization.

Adopted from: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py

"""

import tensorflow as tf

import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras import layers

class ResNet(layers.Layer):

    def __init__(self, num_hiddens,num_res_hiddens, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.first_res=tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu)
        self.second_res=tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu)

    def call(self, inputs):
        first_res=self.first_res(inputs)
        second_res=self.second_res(first_res)
        inputs+=second_res 
        return inputs

class ResNetBlock(layers.Layer):

    def __init__(self, num_hiddens,num_res_hiddens,strides, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.pre_res=(tf.keras.layers.Conv2D(num_hiddens,3,strides=strides,padding="same", activation=None))
        self.first_res=ResNet(num_hiddens,num_res_hiddens)
        self.second_res=ResNet(num_hiddens,num_res_hiddens)

    def call(self, inputs):
        inputs=self.pre_res(inputs)
        inputs=self.first_res(inputs)
        inputs=self.second_res(inputs)
        return tf.nn.relu(inputs)



class ConvBlock(layers.Layer):

    def __init__(self, filters, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.convs=[]
        for f in filters:
            self.convs.append(tf.keras.layers.Conv2D(f,4,strides=(2,2),padding="same", activation=tf.nn.relu))

    def call(self, inputs):
        out=inputs
        for conv in self.convs:
            out=conv(out)
        return out

class DeConvBlock(layers.Layer):

    def __init__(self, filters, **kwargs):
        super(DeConvBlock, self).__init__(**kwargs)
        self.deconvs=[]
        for f in filters:
            self.deconvs.append(tf.keras.layers.Conv2DTranspose(f, 4, strides=(2, 2), padding="same", activation=tf.nn.relu))

    def call(self, inputs):
        out=inputs
        for deconv in self.deconvs:
            out=deconv(out)
        return out



class VQ(tf.keras.layers.Layer):


    def __init__(self, D,K,commitment_beta=1):
        super(VQ, self).__init__(dtype=tf.float32 )
        self.D=D
        self.K=K
        self.commitment_beta=commitment_beta
        self.embeddings=[]
        init=tf.random.normal((self.D, self.K),mean=0., stddev=.1)
        self.embeddings = tf.Variable(shape=(self.D, self.K),dtype=tf.float32,initial_value=init, trainable=True)
    
    #def build(self, input_shape):


    def call(self, x):

        embeddings=self.embeddings
        distances = (
            tf.reduce_sum(x**2, 1, keepdims=True) -
            2 * tf.matmul(x, embeddings) +
            tf.reduce_sum(embeddings**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.K)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(x)[:-1])
        quantized = self.quantize(embeddings,encoding_indices)
        e_k=quantized


        # Sraight Throught
        quantized = x + tf.stop_gradient(quantized - x)

        commitment_loss = tf.reduce_mean((x - tf.stop_gradient(e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(x) - e_k) ** 2)

        self.add_loss(commitment_loss*self.commitment_beta)
        self.add_loss(vq_loss)
        
        return quantized


    def quantize(self, embeddings,encoding_indices):
        w = tf.transpose(embeddings, [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices)


class DVQ(tf.keras.layers.Layer):


    def __init__(self, D,K,L,commitment_beta=1):
        super(DVQ, self).__init__(dtype=tf.float32 )
        self.D=D
        self.K=K
        self.L=L
        self.vqs=[]

        for i in range(L):
            self.vqs.append(VQ(D,K,commitment_beta))


    def call(self, x):
        vq_inputs=tf.split(x,self.L,axis=-1)

        ouputs=[]
        for i in range(self.L):
            inputs=vq_inputs[i]
            vq=self.vqs[i]
            ouputs.append(vq(inputs))
        self.z=tf.concat(ouputs,axis=-1)
        return self.z

class MVQ(tf.keras.layers.Layer):


    def __init__(self, D,K,L,commitment_beta=1):
        super(MVQ, self).__init__(dtype=tf.float32 )
        self.D=D
        self.K=K
        self.L=L
        self.vqs=[]

        for i in range(L):
            self.vqs.append(VQ(D,K,commitment_beta))


    def call(self, x):
        vq_inputs=x

        ouputs=[]
        for i in range(self.L):
            inputs=vq_inputs
            vq=self.vqs[i]
            ouputs.append(vq(inputs))
        self.z=tf.concat(ouputs,axis=-1)
        return self.z



class DVQVAE(tf.keras.Model):

    def __init__(self,channels=3,D=255,K=512,L=1,commitment_beta=1,mse=False):
        super(DVQVAE, self).__init__()
        self.channels=channels
        num_hiddens=255
        num_res_hiddens=64
        size=2
        

        self.encoder_resnetblock = ResNetBlock(num_hiddens,num_res_hiddens,strides=2, name="encoder_resnet")
        self.encoder_convblock=ConvBlock([num_hiddens,num_hiddens])
        self.pre_vq=tf.keras.layers.Conv2D(D*L,3,strides=(1,1),padding="same", activation=None)
        self.vq=DVQ(D,K,L,commitment_beta)
        self.decoder_resnetblock = ResNetBlock(num_hiddens,num_res_hiddens, strides=1,name="decoder_resnet")
        self.decoder_convblock=DeConvBlock([num_hiddens//2,num_hiddens//2])
        self.mse=mse

        if self.mse:
            self.last_conv = tf.keras.layers.Conv2DTranspose( self.channels, 4, strides=(2, 2), padding="same", activation=None) 
        else:
            self.last_conv = tf.keras.layers.Conv2DTranspose( self.channels*256, 4, strides=(2, 2), padding="same", activation=None)

    def call(self, inputs):
        
        out=tf.cast(inputs,tf.float32)/255-0.5
        out=self.encoder_resnetblock(out)
        out=self.encoder_convblock(out)
        out=self.pre_vq(out)
        out_shape=tf.shape(out)
        out=tf.reshape(out,(-1,out_shape[-1]))
        out=self.vq(out)
        out=tf.reshape(out,out_shape)
        out=self.decoder_resnetblock(out)
        out=self.decoder_convblock(out)

        if self.mse:
            out = self.last_conv(out)
            alpha=200
            reconstruction = tf.reduce_mean((out-x_norm)**2)*alpha
            self.add_loss(reconstruction)


        else:

            out = self.last_conv(out)
            logits=tf.reshape(out,[-1,256])
            inputs=tf.cast(tf.reshape(inputs,(-1,)),dtype=tf.int32)
            reconstruction=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=logits))
            self.add_loss(reconstruction)

        return out



class VQVAE(tf.keras.Model):

    def __init__(self,channels=3,D=255,K=512,L=1,commitment_beta=1,mse=False):
        super(VQVAE, self).__init__()
        self.channels=channels
        num_hiddens=255
        num_res_hiddens=64
        size=2
        
        self.L=L
        self.encoder_resnetblock = ResNetBlock(num_hiddens,num_res_hiddens,strides=2, name="encoder_resnet")
        self.encoder_convblock=ConvBlock([num_hiddens,num_hiddens])
        self.pre_vq=tf.keras.layers.Conv2D(D,3,strides=(1,1),padding="same", activation=None)
        self.vq=MVQ(D,K,L,commitment_beta)
        self.decoder_resnetblock = ResNetBlock(num_hiddens,num_res_hiddens, strides=1,name="decoder_resnet")
        self.decoder_convblock=DeConvBlock([num_hiddens//2,num_hiddens//2])
        self.mse=mse

        if self.mse:
            self.last_conv = tf.keras.layers.Conv2DTranspose( self.channels, 4, strides=(2, 2), padding="same", activation=None) 
        else:
            self.last_conv = tf.keras.layers.Conv2DTranspose( self.channels*256, 4, strides=(2, 2), padding="same", activation=None)

    def call(self, inputs):
        
        out=tf.cast(inputs,tf.float32)/255-0.5
        out=self.encoder_resnetblock(out)
        out=self.encoder_convblock(out)
        out=self.pre_vq(out)
        out_shape=out.shape
        out=tf.reshape(out,(-1,out_shape[-1]))
        out=self.vq(out)
        reshaped=(-1,int(out_shape[1]),int(out_shape[2]),int(out_shape[-1])*self.L)
        out=tf.reshape(out,reshaped)
        out=self.decoder_resnetblock(out)
        out=self.decoder_convblock(out)

        if self.mse:
            out = self.last_conv(out)
            alpha=200
            reconstruction = tf.reduce_mean((out-x_norm)**2)*alpha
            self.add_loss(reconstruction)


        else:

            out = self.last_conv(out)
            logits=tf.reshape(out,[-1,256])
            inputs=tf.cast(tf.reshape(inputs,(-1,)),dtype=tf.int32)
            reconstruction=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=logits))

            self.add_loss(reconstruction)

        return out
