import tensorflow as tf
import keras
import numpy as np

class AdaptiveFilteringModel(keras.Model):
    """Class defining the Adaptive Linear Filtering Model.
    """
    def __init__(self, local_optimizer:tf.keras.optimizers.Optimizer, 
                 num_epochs_self_train:int = 500,
                 input_shape:tuple = (3, 256, 1), 
                 track_prediction_history:bool = False,
                 name:str = None):
        """Initializes the AdaptiveFilterModel.

        Args:
            local_optimizer (tf.keras.optimizer.Optimizer): Tensorflow optimizer to 
              train the adaptive filter.
            num_epochs_self_train (int): Number of epochs to converge the model.
              Defaults to 500.
            input_shape (tuple): Linear model input shape. Defaults to (3, 256, 1).
            track_prediction_history (bool): If True, the history of filtered signals 
              is tracked during filter training. Defaults to false.
            name (str): Name of the adaptive filter model. Defaults to None.

        Attributes:
            model (tensorflow.keras.models.Model): The adaptive linear tensorflow model.
            prediction_history (list): List of intermediate filter estimations during 
              training. Used if track_prediction_history is set to True. 
            track_prediction_history (bool): If true, prediction history is tracked.
            initial_weights (list of numpy.array): Initial random weights.
        """
        super().__init__()
        
        self.local_optimizer = local_optimizer
        self.num_epochs_self_train = num_epochs_self_train
        
        mInput = tf.keras.Input(shape = input_shape)
        
        self.conv1 = keras.layers.Conv2D(filters = 1, 
                                         kernel_size = (3, 21),
                                         padding = 'same', 
                                         activation = 'linear')
        self.conv2 = keras.layers.Conv2D(filters = 1, 
                                         kernel_size = (3, 1),
                                         padding = 'valid')
        
        m = self.conv1(mInput)
        m = self.conv2(m)
        m = m[:, 0, :, 0]
        
        self.model = keras.Model(inputs = mInput, outputs = m,
                                 name = name)
        self.initial_weights = self.model.get_weights()
        
        self.track_prediction_history = track_prediction_history
        self.prediction_history = []
        
    def reinitialize_weights(self):
        """Reset linear filter weights to initial random weights.
        """
        self.model.set_weights(self.initial_weights)
        
    def __adaptive_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Adaptive loss function for optimizing the linear filter.

        Args:
            y_true (tensorflow.Tensor): Target (desired) signal.
            y_pred (tensorflow.Tensor): Model output signal.
        Returns:
            e (tensorflow.Tensor): Mean adaptive loss.
        """
        y_true_reshaped = y_true[:, 0, :, 0]
        y_true_fft = tf.cast(y_true_reshaped, dtype = tf.complex128)
        y_true_fft = tf.signal.fft(y_true_fft)

        
        y_pred_fft = tf.cast(y_pred, dtype = tf.complex128)
        y_pred_fft = tf.signal.fft(y_pred_fft)

        e = tf.math.abs(y_true_fft - y_pred_fft)
        e = tf.cast(e, dtype = tf.float64)
        e = tf.math.reduce_sum(tf.math.square(e), axis = -1)
                
        return tf.reduce_mean(e)
    
    def __grad(self, inputs: tf.Tensor, targets: tf.Tensor)->tuple[tf.Tensor, tf.Tensor]:
        """Calculates the required gradients for optimizing linear filter.

        Uses tensorflow's GradientTape to estimate the gradients.

        Args:
            inputs (tensorflow.Tensor): Inputs for calculating the gradient.
            targets (tensorflow.Tensor): Target tensors for calculating the gradient.
        Returns:
            loss_value (tensorflow.Tensor): Loss value based on the self.__loss() function.
            tape.gradient (tensorflow.Tensor): The calculated gradients.
        """
        with tf.GradientTape() as tape:
            loss_value = self.__loss(inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, 
                                         self.model.trainable_variables)
    
    def __loss(self, x: tf.Tensor, y: tf.Tensor, training: bool) -> tf.Tensor:
        """Helper function to calculate loss

        Args:
            x (tensorflow.Tensor): The input to the model (self.model).
            y (tensorflow.Tensor): Corresponding ground-truth values.
        
        Returns:
            loss (tensorflow.Tensor): Adaptive loss as estimated by __adaptive_loss()
              function.
        """
        y_ = self.model(x, training = training)
        
        return self.__adaptive_loss(y_true = y, y_pred = y_)        
    
    def call(self, inputs: np.ndarray, train_weights:bool = True) -> tf.Tensor:
        """Filter input PPG by using ACC signals as motion reference.

        Args:
            inputs (numpy.ndarray): Input signal composing of PPG and acceleration 
              reference signals. Input size is [N_samples, 4, 256].
            train_weights (bool): True if model weights should be adapted/trained 
              on the given input data or not. Defaults to True.
        Returns:
            x_out (tensorflow.Tensor): PPG after adaptively filtering motion artifacts 
              linearly correlated to the acceleration.
        """
        x = inputs[:, 1:, ...]
        y = inputs[:, :1, ...]
        
        self.model.trainable = train_weights

        if train_weights:
            self.reinitialize_weights()
            for epoch in range(self.num_epochs_self_train):
                loss_value, grads = self.__grad(x, y)
                self.local_optimizer.apply_gradients(zip(grads, 
                                                        self.model.trainable_variables))
                
                if self.track_prediction_history:
                    x_out = y[:, 0, :, 0] - self.model(x)
                    self.prediction_history.append(x_out)
            self.model.trainable = False

        x_out = y[:, 0, :, 0] - self.model(x)

        return x_out
    