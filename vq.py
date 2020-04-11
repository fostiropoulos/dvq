"""

Official Implementation of Depthwise Vector Quantization.

Adopted from: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py

"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class VQ:

    def __init__(self,D,K,L=1,commitment_beta=1,lr=0.02):
        self.commitment_beta=commitment_beta
        self.D=D
        self.K=K
        self.lr=lr
        self.L=L
        tf.reset_default_graph()
        self.X=tf.placeholder(tf.float32,[None,self.D*self.L], name="x_input")
        self.embeddings=[]
        self.build_model(self.X)
        self.saver = tf.train.Saver()
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
        self.summary_op=tf.summary.merge_all()
        

    def get_feed_dict(self,X_batch):
        feed_dict={self.X:X_batch}
        return feed_dict


    def save(self,file):
        self.saver.save(self.sess, file)

    def load(self,file):
        self.saver.restore(self.sess, file)

    def quantize(self, embeddings,encoding_indices):
        w = tf.transpose(embeddings, [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices)
    
    def vq_layer(self,inputs,D,K,name="lookup_table",init=tf.truncated_normal_initializer(mean=0., stddev=1)):

        embeddings = tf.get_variable(name,shape=[D, K],dtype=tf.float32,initializer=init)#
        self.embeddings.append(embeddings)
        z_e=inputs
        flat_inputs = tf.reshape(inputs, [-1, self.D])
        distances = (
            tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
            2 * tf.matmul(flat_inputs, embeddings) +
            tf.reduce_sum(embeddings**2, 0, keepdims=True))
        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.K)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = self.quantize(embeddings,encoding_indices)
        e_k=quantized


        quantized = inputs + tf.stop_gradient(quantized - inputs)

        avg_probs = tf.reduce_mean(encodings,0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                       tf.math.log(avg_probs + 1e-10)))

        commitment_loss = tf.reduce_mean((z_e - tf.stop_gradient(e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - e_k) ** 2)



        return {"outputs":quantized,"perplexity":perplexity,"commitment_loss":commitment_loss,"vq_loss":vq_loss,"encodings":encodings}
    
    def distance_metric(self,x,y):
        return  (tf.reduce_sum(x**2, 1, keepdims=True) -2 * tf.matmul(x, y) + tf.reduce_sum(y**2, 0, keepdims=True))

    def build_model(self,inputs):

        with tf.variable_scope("vq"):
            self.vq_loss=tf.constant([0.])
            self.commitment_loss=tf.constant([0.]) 
            x=inputs   



            self.vq_inputs=tf.split(inputs,self.L,axis=-1)
            self.z=tf.constant(0.)
            z=[]
            self.distance=tf.constant(0.)
            self.perplexity=tf.constant(0.)
            for i in range(self.L):
                out=self.vq_layer(self.vq_inputs[i],self.D,self.K,name="lookup_table_%d"%i)
                self.vq_loss+=out["vq_loss"]
                self.commitment_loss+=out["commitment_loss"]
                
                self.distance+=tf.reduce_mean(tf.reduce_min(out["distance"],axis=-1))
                self.perplexity+=out["perplexity"]
                z.append(out["outputs"])


            for emb in self.embeddings:
                x=tf.transpose(emb,[1,0])
                y=emb

            self.z=tf.concat(z,axis=-1)

            self.vq_loss=tf.reshape( self.vq_loss,[])
            self.commitment_loss=tf.reshape( self.commitment_loss,[])
        self._loss_init(self.X,self.z)
        self._train_init()


    def partial_fit(self,X,X_test=None, batch_size=64,epochs=None):

        np.random.shuffle(X)
        num_batches=X.shape[0]//batch_size
        train_out=[]
        with tqdm(range(num_batches),disable=True) as t:
            for i in t:
                X_batch=X[i*batch_size:(i+1)*batch_size]
                X_batch
                losses,_=self.sess.run([list(self.losses.values())]+[self.train],feed_dict=self.get_feed_dict(X_batch))
                loss_monitor=list(zip(self.losses.keys(),losses))
                desc="%s "%epochs
                for name,val in loss_monitor:
                    desc+=("%s: %.2f\t"%(name,val))
                t.set_description(desc)
                train_out.append(loss_monitor)

        if(X_test is not None):
            
            np.random.shuffle(X_test)
            losses=self.sess.run( list(self.losses.values()),    feed_dict=self.get_feed_dict(X_test[:batch_size]))
            test_out=list(zip(self.losses.keys(),losses))
        else:
            test_out=[.0]*len(self.losses)

        return train_out, [test_out]

    def plot_reconstruction(self,X):
        z=self.sess.run(self.z,feed_dict={self.X:X})
        embd=self.sess.run(self.embeddings)
        plt.scatter(embd[0,:],embd[1,:])
        plt.legend()
        plt.show()

    def fit(self,X,X_test=None,epochs=20,batch_size=64, plot=False, verbose=True,log_dir=None):
        train_monitor=[]
        test_monitor=[]
        np.random.shuffle(X)
        train_batch_sum=self.read_batch(X[:10])
        writer_test=None

        if not X_test is None:
            x_test_indices=list(range(X_test.shape[0]))
            np.random.shuffle(X_test)
            test_batch_sum=self.read_batch(X_test[x_test_indices[:10]])
            writer_test = tf.summary.FileWriter(log_dir+"/test",self.sess.graph) if log_dir else None
        writer_train = tf.summary.FileWriter(log_dir+"/train",self.sess.graph) if log_dir else None

        for epoch in range(epochs):
            train_out,test_out=self.partial_fit(X,X_test, batch_size,epochs="%s/%s"%(epoch,epochs))
            train_monitor+=train_out
            test_monitor+=test_out
            
            if plot:
                self.plot_reconstruction(train_batch_sum)

            if(verbose):
                print("Train {}".format(train_monitor[-1]))
                print("Test {}".format(test_monitor[-1]))
        if plot:
            plot_loss(np.array(train_monitor),"train")
            plot_loss(np.array(test_monitor),"test")
        return train_monitor,test_monitor


    def _loss_init(self,inputs,outputs):

        """
        VAE-LOSS
        """

        reconstr_loss=tf.keras.losses.MSE(inputs,outputs)
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)

        """
        VQ-LOSS
        """
        self.loss=self.reconstr_loss +self.vq_loss

    
        self.losses={}
        self.losses["total loss"]=self.loss
        self.losses["VQ"]=self.vq_loss
        self.losses["Commitment"]=self.commitment_loss
        self.losses["reconstruction"]=self.reconstr_loss
        self.losses["distance"]=self.distance


        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("total_loss", self.loss)

    def _train_init(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.optimizer_distance = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train = [self.optimizer.minimize(self.loss)]