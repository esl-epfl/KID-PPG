#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 10:47:18 2023

@author: kechris
"""

import tensorflow as tf
import tensorflow_probability as tfp
import scipy
from scipy import stats
tfd = tfp.distributions

class KID_PPG:

    def __init__(self, input_shape = (256, 1),
                 weights_file = None):
        self.input_shape = input_shape

        self.model = self.build_model_probabilistic()

        if weights_file != None:
            self.model.load_weights(weights_file)
        
        self.submodel = tf.keras.models.Model(inputs = self.model.inputs, 
                                              outputs = self.model.layers[-2].output)


    def predict(self, x, threshold = None):
        y_pred = self.submodel.predict(x)
        
        y_pred_m = y_pred[:, 0]
        y_pred_std = (1 + tf.math.softplus(y_pred[:,1:2])).numpy().flatten()
        
        if threshold != None:
            p = scipy.stats.norm(y_pred_m, y_pred_std).cdf(y_pred_m + threshold) \
                - scipy.stats.norm(y_pred_m, y_pred_std).cdf(y_pred_m - threshold)
            
            return y_pred_m, y_pred_std, p
        
        return y_pred_m, y_pred_std


    def convolution_block(self, input_shape, n_filters, 
                        kernel_size = 5, 
                        dilation_rate = 2,
                        pool_size = 2,
                        padding = 'causal'):
            
        mInput = tf.keras.Input(shape = input_shape)
        m = mInput
        for i in range(3):
            m = tf.keras.layers.Conv1D(filters = n_filters,
                                    kernel_size = kernel_size,
                                    dilation_rate = dilation_rate,
                                        padding = padding,
                                    activation = 'relu')(m)
        
        m = tf.keras.layers.AveragePooling1D(pool_size = pool_size)(m)
        m = tf.keras.layers.Dropout(rate = 0.5)(m, training = False)
            
        model = tf.keras.models.Model(inputs = mInput, outputs = m)
        
        return model

    def my_dist(self, params):
        return tfd.Normal(loc=params[:,0:1], 
                        scale = 1 + tf.math.softplus(params[:,1:2]))# both parameters are learnable
        

    def build_model_probabilistic(self, return_attention_weights = False):
        modal_input_shape = (self.input_shape[0], 1)
        
        mInput = tf.keras.Input(shape = self.input_shape)
        
        mInput_t_1 = mInput[..., :1]
        mInput_t = mInput[..., 1:]
        
        conv_block1 = self.convolution_block(modal_input_shape, n_filters = 32,
                                        pool_size = 4)
        conv_block2 = self.convolution_block((64, 32), n_filters = 48)
        conv_block3 = self.convolution_block((32, 48), n_filters = 64)
        
        m_ppg_t_1 = conv_block1(mInput_t_1)
        m_ppg_t_1 = conv_block2(m_ppg_t_1)
        m_ppg_t_1 = conv_block3(m_ppg_t_1)
        
        m_ppg_t = conv_block1(mInput_t)
        m_ppg_t = conv_block2(m_ppg_t)
        m_ppg_t = conv_block3(m_ppg_t)
        
        
        attention_layer = tf.keras.layers.MultiHeadAttention(num_heads = 4,
                                                            key_dim = 16,
                                                            )
        
        if return_attention_weights:
            m, attention_scores = attention_layer(query = m_ppg_t, value = m_ppg_t_1, return_attention_scores=True)
        else:
            m = attention_layer(query = m_ppg_t, value = m_ppg_t_1, return_attention_scores = False)
        
        m = m + m_ppg_t
        
        m = tf.keras.layers.LayerNormalization()(m)
        
            
        m = tf.keras.layers.Flatten()(m)
        m = tf.keras.layers.Dense(units = 256, activation = 'relu')(m)
        m = tf.keras.layers.Dropout(rate = 0.125)(m)
        m = tf.keras.layers.Dense(units = 2)(m)
        
        m = tfp.layers.DistributionLambda(my_dist)(m)
        
        if return_attention_weights:
            model = tf.keras.models.Model(inputs = mInput, outputs = [m, attention_scores])
        else:
            model = tf.keras.models.Model(inputs = mInput, outputs = m)
            
        model.summary()
        
        return model 