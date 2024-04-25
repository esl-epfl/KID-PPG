import tensorflow as tf
import keras

class AdaptiveFilteringModel(keras.Model):
    def __init__(self, local_optimizer, num_epochs_self_train = 500,
                 input_shape = (3, 256, 1), track_prediction_history = False,
                 name = None):
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
        self.model.set_weights(self.initial_weights)
        
    def adaptive_loss(self, y_true, y_pred):
        y_true_reshaped = y_true[:, 0, :, 0]
        y_true_fft = tf.cast(y_true_reshaped, dtype = tf.complex128)
        y_true_fft = tf.signal.fft(y_true_fft)

        
        y_pred_fft = tf.cast(y_pred, dtype = tf.complex128)
        y_pred_fft = tf.signal.fft(y_pred_fft)

        e = tf.math.abs(y_true_fft - y_pred_fft)
        e = tf.cast(e, dtype = tf.float64)
        e = tf.math.reduce_sum(tf.math.square(e), axis = -1)
                
        return tf.reduce_mean(e)
    
    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, 
                                         self.model.trainable_variables)
    
    def loss(self, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = self.model(x, training = training)
        
        return self.adaptive_loss(y_true = y, y_pred = y_)        
    
    def call(self, inputs, train_weights = True):
        x = inputs[:, 1:, ...]
        y = inputs[:, :1, ...]
        
        self.model.trainable = train_weights

        if train_weights:
            self.reinitialize_weights()
            for epoch in range(self.num_epochs_self_train):
                loss_value, grads = self.grad(x, y)
                self.local_optimizer.apply_gradients(zip(grads, 
                                                        self.model.trainable_variables))
                
                if self.track_prediction_history:
                    x_out = y[:, 0, :, 0] - self.model(x)
                    self.prediction_history.append(x_out)
            self.model.trainable = False

        x_out = y[:, 0, :, 0] - self.model(x)

        return x_out
    