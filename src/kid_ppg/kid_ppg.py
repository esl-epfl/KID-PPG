import tensorflow as tf
import tensorflow_probability as tfp
import scipy
import numpy as np

from importlib import resources as impresources
from . import model_weights

tfd = tfp.distributions

weights_file = impresources.files(model_weights) / 'kid_ppg_weights.h5'
weights_filename = weights_file.resolve()

class KID_PPG:
    """KID-PPG class. Defines the KID-PPG model. 

    Attributes:
        input_shape (tuple): KID-PPG model input shape.
        model (tensorflow.keras.models.Model): Tensorflow model.
        submodel (tensorflow.keras.models.Model): Same as model but without the last
        distribution layer. 
    """
    def __init__(self, input_shape:tuple = (256, 2),
                 input_weights_file:str = None, load_weights:bool = True):
        """Initializes KID-PPG

        Args:
            input_shape (tuple): Shape of the input data given to the KID-PPG model. 
              Defaults to (256, 2).
            input_weights_file (str): String directory pointing to the location of 
              the KID-PPG model pretrained weights. Defaults to None, default location 
              with pretrained weights is used. 
            load_weights (bool): True if pretrained weights should be loaded. 
              Defaults to True.
        """
        self.input_shape = input_shape

        self.model = self.__build_model_probabilistic()

        if load_weights:

            if input_weights_file is None:
                self.model.load_weights(weights_filename)
            else:
                self.model.load_weights(input_weights_file)
        
        self.submodel = tf.keras.models.Model(inputs = self.model.inputs, 
                                              outputs = self.model.layers[-2].output)


    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Estimates HR probability given PPG input.

        PPG input should be sampled at 32Hz and converted into pairs [PPG(n), PPG(n + 1)].

        Args:
            x (numpy.ndarray): PPG signal for heart rate extraction. 
            Size should be [N_samples, 256, 2].
        Returns:
            y_pred_m (numpy.ndarray): Expected HR as estimated by KID-PPG model.
              Size is [N_samples, 1].
            y_pred_std (numpy.ndarray): Estimated standard deviation of the HR distribution. 
              Size is [N_samples, 1]
        """
        y_pred = self.submodel.predict(x)
        
        y_pred_m = y_pred[:, 0]
        y_pred_std = (1 + tf.math.softplus(y_pred[:,1:2])).numpy().flatten()
        
        return y_pred_m, y_pred_std

    def predict_threshold(self, x: np.array, threshold: np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Estimates HR probability given PPG input. It also calculates
        the probability of error: p(error < threshold).

        PPG input should be sampled at 32Hz and converted into pairs [PPG(n), PPG(n + 1)].

        Args:
            x (numpy.ndarray): PPG signal for heart rate extraction. 
              Size should be [N_samples, 256, 2].
            threshold (numpy.float32): Threshold for estimating the probability 
              p(error > threshold).
        Returns:
            y_pred_m (numpy.ndarray): Expected HR as estimated by KID-PPG model. 
              Size is [N_samples, 1].
            y_pred_std (numpy.ndarray): Estimated standard deviation of the 
              HR distribution. Size is [N_samples, 1].
            p_error (numpy.ndarray): Estimated probability of error > threshold. 
              Size is [N_samples, 1].
        """
        y_pred_m, y_pred_std = self.predict(x)

        p_error = scipy.stats.norm(y_pred_m, y_pred_std).cdf(y_pred_m + threshold) \
                - scipy.stats.norm(y_pred_m, y_pred_std).cdf(y_pred_m - threshold)
            
        return y_pred_m, y_pred_std, p_error

    def __convolution_block(self, input_shape:tuple, n_filters:int, 
                        kernel_size:int = 5, 
                        dilation_rate:int = 2,
                        pool_size:int = 2,
                        padding:str = 'causal')->tf.keras.models.Model:
        """ Creates a convolutional block containing convolutional layers
        followed by a 1D Average Pooling and a Dropout layer.

        Args:
            input_shape (tuple): the input shape of the first convolutional layer.
            n_filters (int): Number of filters on each convolutional layer.
            kernel_size (int): Size of convolution kernel. Defautls to 5.
            dilation_rate (int): Dilation rate used in the convolution. Defaults to 2.
            pool_size (int): Pool size of the average pooling layer. Defaults to 2.
            padding (str): Padding mode used by convolutional layers. Defaults to 'str'.

        Returns:
            model (tensorflow.models.Model): Tensorflow model ofthe convolutional block.
        """
            
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

    def __my_dist(self, params:tf.Tensor) -> tfp.distributions.Normal:
        """Defines the Normal distribution used as the exit of the KID-PPG model.

        Args:
            params (tf.Tensor): Parameters for the Normal distribution. Shape should be 
            (..., 2), where (..., 0) refers to the loc parameters and (..., 1) to the 
            scale. 

        Returns:
            tfd.Normal() (tfp.distributions.Normal): The normal distribution.
        """
        return tfd.Normal(loc=params[:,0:1], 
                        scale = 1 + tf.math.softplus(params[:,1:2]))
        

    def __build_model_probabilistic(self, return_attention_weights:bool = False) -> tf.keras.models.Model:
        """Create probabilistic model.
    
        Model takes as inputs PPG pairs of [X(n), X(n + 1)] are associated 
        and outputs a heart rate probability estimate.
        
        Args:
            return_attention_weights (bool): Set to true if the model should return 
            the attention weights. Defaults to false.

        Returns:
            model (tensorflow.models.Model): Tensorflow model of KID-PPG.
        """
        
        modal_input_shape = (self.input_shape[0], 1)
        
        mInput = tf.keras.Input(shape = self.input_shape)
        
        mInput_t_1 = mInput[..., :1]
        mInput_t = mInput[..., 1:]
        
        conv_block1 = self.__convolution_block(modal_input_shape, n_filters = 32,
                                        pool_size = 4)
        conv_block2 = self.__convolution_block((64, 32), n_filters = 48)
        conv_block3 = self.__convolution_block((32, 48), n_filters = 64)
        
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
        
        m = tfp.layers.DistributionLambda(self.__my_dist)(m)
        
        if return_attention_weights:
            model = tf.keras.models.Model(inputs = mInput, outputs = [m, attention_scores])
        else:
            model = tf.keras.models.Model(inputs = mInput, outputs = m)
                    
        return model 

    def summary(self):
        """ Prints model summary.
        """
        self.model.summary()