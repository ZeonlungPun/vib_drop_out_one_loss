import tensorflow.keras.models
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from keras import optimizers
import math


class VIB(Layer):
    def __init__(self, **kwargs):
        super(VIB, self).__init__(**kwargs)
        """specific vib layers"""
    def call(self, input1,input2):
        z_mean, z_log_var = input1,input2
        u = K.random_normal(shape=K.shape(z_mean))
        self.kl_loss = - K.sum(K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), 0))/math.log(2)
        u = K.in_train_phase(u, 0.)
        return z_mean +K.softplus(K.exp(z_log_var / 2)) * u,self.kl_loss
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VIBModel(tensorflow.keras.models.Model):
    def __init__(self,INPUT_SHAPE,k):
        super(VIBModel, self).__init__()
        self.VIB=VIB()
        self.INPUT_SHAPE=INPUT_SHAPE
        self.FC1=Dense(1024,activation='relu')
        self.FC2=Dense(1024,activation='relu')
        self.mean=Dense(k,activation='relu')
        self.logvar=Dense(k,activation='relu')
        #self.last=Dense(1,activation='sigmoid')
        self.last=Dense(1,activation=None)

    def call(self,input):
        hidden1=self.FC1(input)
        hidden2=self.FC2(hidden1)
        z_mean=self.mean(hidden2)
        z_log_var=self.logvar(hidden2)
        after_vib,kl_loss=self.VIB(input1=z_mean,input2=z_log_var)

        output=self.last(after_vib)
        return output,kl_loss,z_mean




#get the coefficient of each variables in one experiment
def get_result(mask_variable,x,y,epochs,lamda1,lamda2,opt,K,Gaussian_noise=False,classification=False,times=3):

    """
    :param times: repeat times in one experiment
    :param mask_variable: drop out variable no.;msak==-1:do not drop out
              Gaussian_noise: replace dropping variables with gaussian noise
    :param x: fitting x
    :param y: fitting y
    lamda1: parameters for IB balancing I(X,Z) AND I(Z,Y)
    lamda2: regular for weights
    opt : optimizer
    K: hidden layer num
    """
    if mask_variable !=-1:
        all=set(range(x.shape[1]))
        all.discard(mask_variable)
        all=list(all)
        x=x[:,all]
        if Gaussian_noise:
            x_=0.01*np.random.rand(x.shape[0],1)
            x=np.concatenate([x,x_],axis=1)

        else:
            x=x

    score1=0
    score2=0
    score3=0
    for _ in range(times):

        model = VIBModel(INPUT_SHAPE=x.shape[1],k=K)

        if opt=="sgd":
            optimizer = optimizers.SGD(learning_rate=0.05, nesterov=True)
        else:
            optimizer=optimizers.Adam(learning_rate=0.05)

        iz_ = []
        Ixz_=[]
        Iyz_=[]
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                train_pred, izx,encoded = model(x)
                if classification:
                    loss_func = tf.keras.losses.BinaryCrossentropy()
                else:
                    loss_func=tf.keras.losses.MeanSquaredError()
                y = y.reshape(train_pred.shape)
                loss=loss_func(y, train_pred) + lamda1 * izx
                izy=math.log(10,2)-loss_func(y,train_pred)

                loss+=lamda2*tf.reduce_sum(tf.abs(model.FC1.kernel))/model.FC1.kernel.shape[0]

                iz_.append(float(-izy+lamda1*izx))
                Ixz_.append(float(izx))
                Iyz_.append(float(izy))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print('in epoch {},izx is {},izy is {}'.format(epoch,izx,izy))
        score1=score1+np.array(iz_)
        score2=score2+np.array(Ixz_)
        score3=score3+np.array(Iyz_)

    return score1/times,model


def get_coeff(EPOCH,x,y,k):
    WJ = np.zeros((x.shape[1] + 1, EPOCH))
    wj, model = get_result(times=3, mask_variable=-1, x=x, y=y,K=k,classification=False,opt='sgd',epochs=EPOCH)
    WJ[-1, :] = wj

    for i in range(0, x.shape[1]):
        wj, model = get_result(times=3, mask_variable=i, x=x, y=y,K=k,classification=False,opt='sgd',epochs=EPOCH)
        WJ[i, :] = wj

    return WJ

