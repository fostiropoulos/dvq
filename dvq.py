"""

Official Implementation of Depthwise Vector Quantization.

Adopted from: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py

"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt 
def plot_loss(loss_monitor,loss_name="", title=""):
    fig=plt.figure(figsize=(10,5))
    for i in range(loss_monitor.shape[-2]):
        plt.plot(loss_monitor[:,i,1].astype(float),label="%s %s"%(loss_name,loss_monitor[0,i,0]))
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title(title)
    plt.show()

class DVQ:        


    def __init__(self,image_size,channels, D,K,L,commitment_beta=1,lr=2e-4, num_convs=1,vq_type="DVQ", mse=False):
        """ Returns a VQVAE model with Spatial indepedence (SVQ), Feature indepedence (DVQ) or no assumptions on the indepedence (VQ)

        Parameters:
            image_size (int):All images will be resized to (image_size x image_size)
            channels (int):All images will be converted to mode "L" (see PIL.Image for more details) for 1 or "RGB" for 3. 
            D (int): Embedding Dimension
            K (int): Number of Quantization vectors in each codebook
            L (int): Number of Codebooks to use
            commitment_beta (int): Penalty coefficient on the trained prior
            lr (float): Learning rate for Adam optimizer
            num_convs (int): Number of convolutions/deconvolutions in the encoder/decoder network. Higher number leads to smaller embedding dim.
            vq_type (str): The type of network, if invalid or empty, is DVQ by default. Options: [DVQ,SVQ,VQ] 
                                SVQ: Will create indepedent codebooks (L) for a set of pixels in the embedding tensor
                                DVQ: Will create indepedent codebooks (L) for a set of features in the embedding tensor
                                VQ: Will create indepedent codebooks (L) for the entire embedding tensor
            mse (bool): If set to true, reconstruction loss will be in MSE. Use only for improved reconstructions, not evaluation. 


        """
        self.D=D
        self.K=K
        self.L=L

        # FLAGS
        self.mse=mse
        self.vq_type=vq_type

        self.image_size=image_size
        self.channels=channels


        self.X=None
        self.z=None
        self.display_layer=None

        self.losses=[]
        self.embeddings=[]
        self.summary_op=None
        self.build_model(lr,num_convs,commitment_beta)
        self.saver = tf.train.Saver()
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
        self.summary_op=tf.summary.merge_all()
        

    def save(self,file):
        self.saver.save(self.sess, file)

    def load(self,file):
        self.saver.restore(self.sess, file)

    def quantize(self, embeddings,encoding_indices):
        w = tf.transpose(embeddings, [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices)
    def get_codebooks(self):
        return self.embeddings
    def vq_layer(self,inputs,D,K,name="lookup_table",init=tf.truncated_normal_initializer(mean=0., stddev=.1)):


        embeddings = tf.get_variable(name,shape=[D, K],dtype=tf.float32,initializer=init)
        self.embeddings.append(embeddings)
        z_e=inputs
        flat_inputs = tf.reshape(inputs, [-1, D])
        distances = (
            tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
            2 * tf.matmul(flat_inputs, embeddings) +
            tf.reduce_sum(embeddings**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, K)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = self.quantize(embeddings,encoding_indices)
        e_k=quantized


        # Sraight Throught
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        avg_probs = tf.reduce_mean(encodings,0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                       tf.math.log(avg_probs + 1e-10)))

        commitment_loss = tf.reduce_mean((z_e - tf.stop_gradient(e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - e_k) ** 2)
        return {"out":quantized,"perplexity":perplexity,"commitment_loss":commitment_loss,
        "vq_loss":vq_loss,"encodings":encoding_indices}
        
    def _pre_vq(self,z_e):
        if self.vq_type=="VQ":
            # Vanilla VQ-VAE
            pre_vq=(tf.keras.layers.Conv2D(self.D,3,strides=(1,1),padding="same", activation=None, name="pre_vq"))(z_e)
            vq_inputs=pre_vq

        elif self.vq_type=="SVQ":
            pre_vq=(tf.keras.layers.Conv2D(self.D,3,strides=(1,1),padding="same", activation=None, name="pre_vq"))(z_e)
            # Reshape to flatten the spatial dimension such that (B,w,h,D)=(B,w*h,D)
            pre_vq=tf.reshape(pre_vq,(-1,int(pre_vq.shape[1])**2,self.D))
            # Split pre_vq along the first axis which is the spatial dimension. NOTE: w*h%L==0
            assert int(pre_vq.shape[1])**2%self.L == 0, "Condition must be met w*h%L==0 (w,h of the embedding tensor z_e)"
            vq_inputs=tf.split(pre_vq,self.L,axis=1)

        else:
            pre_vq=tf.keras.layers.Conv2D(self.D*self.L,3,strides=(1,1),padding="same", activation=None, name="pre_vq")(z_e)
            vq_inputs=tf.split(pre_vq,self.L,axis=-1)
        return vq_inputs

    def _post_vq(self,z):
        if self.vq_type=="SVQ":
            z_q_shape=[-1]+list(np.array(self.z_e.shape[1:]).astype(np.int32))
            z_q=tf.reshape(tf.concat(z,axis=1),z_q_shape)            
        else:
            z_q=tf.concat(z,axis=-1)
        return z_q

    def vq(self,x):
        with tf.variable_scope("vq"):
            vq_loss=[] 
            commitment_loss=[] 
            z=[]
            encodings=[]
            self.z_e=self._pre_vq(x)
            for i in range(self.L):
                if self.vq_type=="VQ":
                    z_e_i=self.z_e 
                else:
                    z_e_i=self.z_e[i]
                
                if self.vq_type=="SVQ":
                    D_i=self.D*self.L
                else:
                    D_i=self.D
                
                vq=self.vq_layer(z_e_i,D_i,self.K,name="lookup_table_0_%d"%i)
                vq_loss.append(vq["vq_loss"])
                commitment_loss.append(vq["commitment_loss"])
                z.append(vq["out"])
                encodings.append(tf.cast(tf.expand_dims(vq["encodings"],-1),tf.float32))

       
            self.z=self._post_vq(z)
            self.encodings=tf.concat(encodings,axis=-1)
            self.vq_loss=tf.reduce_sum(vq_loss)
            self.commitment_loss=tf.reduce_sum(commitment_loss)
        return self.z

    def build_model(self, lr, num_convs,commitment_beta):

        tf.reset_default_graph()

        self.X=tf.placeholder(tf.int32,[None,self.image_size,self.image_size,self.channels], name="x_input")
        x_norm=tf.cast(self.X,tf.float32)/255-0.5

        num_hiddens=255
        num_res_hiddens=64

        with tf.variable_scope("encoder"):
            conv=(tf.keras.layers.Conv2D(num_hiddens//2,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="conv_0"))(x_norm)

            for i in range(num_convs):
               
                conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="conv_%d"%(i+1)))(conv)

            for i in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_conv_%d"%i))(conv)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_conv_2_%d"%i))(first_res)
                conv+=second_res 
            z_e=tf.nn.relu(conv)


        self.z_q=self.vq(z_e)

        with tf.variable_scope("decoder"):


            conv=(tf.keras.layers.Conv2D(num_hiddens,3,strides=(1,1),padding="same", activation=None, name="conv_0"))(self.z_q)

            for i in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_conv_%d"%i))(conv)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_conv_2_%d"%i))(first_res)
                conv+=tf.nn.relu(second_res) 
            
            deconv=conv
            for i in range(num_convs):
                deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i)) (deconv)        

        if self.mse:

            last_layer = tf.keras.layers.Conv2DTranspose( self.channels, 4, strides=(2, 2), padding="same", activation=None, name="dec_deconv_%d"%(i+1)) (deconv)
            outputs=last_layer
            inputs=x_norm
            self.reconstruction = tf.reduce_mean((outputs-inputs)**2)*200
            self.display_layer=tf.cast(tf.clip_by_value((outputs+0.5)*255,0,255),tf.int32,name="output")
        else:

            last_layer = tf.keras.layers.Conv2DTranspose( self.channels*256, 4, strides=(2, 2), padding="same", activation=None, name="dec_deconv_%d"%(i+1)) (deconv)
            logits_shape=[-1,self.channels*self.image_size**2,256] 
            logits=tf.reshape(last_layer,logits_shape)
            self.logits=logits
            inputs=tf.reshape(self.X,(-1,self.channels*self.image_size**2), name="inputs")
            self.inputs=inputs
            self.reconstruction=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=logits))
            
            self.display_layer=tf.cast(tf.reshape(tf.math.argmax(tf.nn.softmax(logits,name="output"),axis=-1),tf.shape(self.X)),tf.int32)



        self.loss=self.reconstruction +   self.vq_loss + self.commitment_loss * commitment_beta 

        self.losses={}
        self.losses["Total"]=self.loss
        self.losses["VQ"]=self.vq_loss
        self.losses["Commitment"]=self.commitment_loss
        self.losses["Reconstruction"]=self.reconstruction

        hstack=tf.cast(tf.concat(([self.display_layer,self.X]),axis=1),tf.float32)
        tf.summary.image("reconstruction",hstack)
        tf.summary.scalar('VQ', self.vq_loss )
        tf.summary.scalar('Commitment', self.commitment_loss )
        tf.summary.scalar("Reconstruction", self.reconstruction)
        tf.summary.scalar("Total", self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train = [self.optimizer.minimize(self.loss)]

    def reconstruct(self,x):
        return self.sess.run(self.display_layer,feed_dict={self.X:x})

    def plot_reconstruction(self,original, per_row=10):
        reconstructions=self.reconstruct(original)
        num_imgs=original.shape[0]
        overflow=1 if num_imgs%per_row!=0 else 0
        f,a=plt.subplots((num_imgs//per_row+overflow)*2,per_row,figsize=(20,4))
        for i in range(num_imgs):
            img_dim=original.shape[-2]
            if original.shape[-1]==1:
                reshape_dims=original.shape[1:-1]
            else:
                reshape_dims=original.shape[1:]
            a[i//per_row*2][i%per_row].imshow(original[i%per_row].reshape(reshape_dims))
            a[i//per_row*2+1][i%per_row].imshow(reconstructions[i%per_row].reshape(reshape_dims))
        plt.show()


    def get_feed_dict(self,X_batch):
        feed_dict={self.X:X_batch}
        return feed_dict

    def read_batch(self,paths):
        imgs=np.zeros([len(paths),self.image_size,self.image_size,self.channels],dtype=np.int32)
        mode="RGB" if(self.channels==3) else "L"
        for i,img in enumerate(paths):
            _img=Image.open(img).convert(mode)
            imgs[i]=np.array(_img.resize((self.image_size,self.image_size))).reshape(self.image_size,self.image_size,self.channels)
        return imgs

    def partial_fit(self,X,X_test=None, batch_size=64,epochs=None):

        np.random.shuffle(X)
        num_batches=len(X)//batch_size
        train_out=[]
        with tqdm(range(num_batches)) as t:
            for i in t:
                X_batch=X[i*batch_size:(i+1)*batch_size]
                X_images=self.read_batch(X_batch)

                losses,_=self.sess.run( [list(self.losses.values())]+[self.train],feed_dict=self.get_feed_dict(X_images))
                loss_monitor=list(zip(self.losses.keys(),losses))
                desc="Epochs: %s/%s "%epochs
                for name,val in loss_monitor:
                    desc+=("%s: %.2f\t"%(name,val))
                t.set_description(desc)
                train_out.append(loss_monitor)

        if(X_test is not None):
            
            np.random.shuffle(X_test)
            X_images=self.read_batch(X_test[:batch_size])
            losses=self.sess.run( list(self.losses.values()),feed_dict=self.get_feed_dict(X_images))
            test_out=[list(zip(self.losses.keys(),losses))]
        else:
            test_out=[[.0]*len(self.losses)]

        return train_out, test_out


    def fit(self,X,X_test=None,epochs=20,batch_size=128, plot=True, verbose=True,log_dir=None):
        train_monitor=[]
        test_monitor=[]

        np.random.shuffle(X)
        summary_batch=self.read_batch(X[:10])
        writer_test=None

        if not X_test is None:
            np.random.shuffle(X_test)
            test_summary_batch=self.read_batch(X_test[:10])
            writer_test = tf.summary.FileWriter(log_dir+"/test",self.sess.graph) if log_dir else None
        writer_train = tf.summary.FileWriter(log_dir+"/train",self.sess.graph) if log_dir else None

        for epoch in range(epochs):

            train_out,test_out=self.partial_fit(X,X_test, batch_size,(epoch,epochs))
            train_monitor+=train_out
            test_monitor+=test_out
            
            if plot:
                self.plot_reconstruction(summary_batch)

            if writer_train!=None:
                summary=self.sess.run(self.summary_op, feed_dict={self.X:summary_batch})
                writer_train.add_summary(summary, epoch)
                writer_train.flush()
                if writer_test!=None:
                    summary=self.sess.run(self.summary_op, feed_dict={self.X:test_summary_batch})
                    writer_test.add_summary(summary, epoch)
                    writer_test.flush()

            if(verbose):
                print("Train {}".format(train_monitor[-1]))
                print("Test {}".format(test_monitor[-1]))
        if plot:
            plot_loss(np.array(train_monitor),"train")
            plot_loss(np.array(test_monitor),"test")
        return train_monitor,test_monitor
