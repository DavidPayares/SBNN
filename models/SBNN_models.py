# Neurodegenerative Diseases Classification
# Construct 3D CNN.
# Author: David Payares
# Copyleft: MIT Licience

from tensorflow.keras.layers import Input, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model

import tensorflow as tf
import tensorflow_probability as tfp

class SBNNModels(object):
    
    def __init__(self, 
                 model_name = 'reparametrization', 
                 input_shape = (91,109,91,1),
                 pooling = "max",
                 drop_rate = 0.5,
                 training_data_length = 10,
                 classes = 5):
        
        """Spatially Informed Bayesian Neural Network.
        
           Please see link-of-thesis for more information.
            
            Parameters:
            -----------
            
            model_name: string, select one bayesian model among
                          "reparametrization", "flipout-kl" and  "flipout-prior".
                          default model "reparametrization".
            input_shape: list or tuple of four ints, the shape of the input data.
                         (91, 109, 91, 1) by default.
            pooling: string, pooling methods. "max" for max pooling,
                       "avg" for average pooling. Default is "max".
            drop_rate: float, dropout rate, default is 0.5.
            training_data_length : int, numer of samples in the training data.
            classes: number of classes.
            
            
            Returns
            -------
            
            Model object.
            
            
            Raises
            ------
            ValueError if model is not an allowable value.
            
            
        """
        
        # Set model parameters
        self.model_name = model_name
        self.input_shape = input_shape
        self.pooling = pooling
        self.drop_rate = drop_rate
        self.training_data_length = training_data_length
        self.classes = classes
        
        
        # Set bayesian model parameters
        
        ### Gaussian priors
        # Layers weights and bias distribution priors
        self.kernel_prior = tfp.layers.default_multivariate_normal_fn
        self.bias_prior = tfp.layers.default_multivariate_normal_fn
        
        ### Normal posteriors
        # Layers weights and bias distribution posteriors
        self.kernel_posterior = tfp.layers.default_mean_field_normal_fn(is_singular = False)
        self.bias_posterior = tfp.layers.default_mean_field_normal_fn(is_singular = False)
        
        ### Kullback-Leibler divergence
        self.divergence_fn = lambda q,p,_: tfp.distributions.kl_divergence(q,p)/ self.training_data_length
        
        
        # Build SBNN models
        if model_name in {'reparametrization', 'flipout-kl', 'flipout-prior'}:
            self.model = self.bayesianModel(self.input_shape)
        else:
            raise ValueError("Unknown model name. Allowed models are 'reparametrization', 'flipout-kl' and  'flipout-prior'")
            
        return
    
    def conv3D(self, inputs, filter_size, strides):
        
        """Build a 3D bayesian convolution layer .
        
            
            Parameters:
            -----------
            
            inputs:  input tensor, it should be the original input,
                      or the output from a previous layer.
            filter_size: int, the number of filters.
            strides: int tuple with length 3, the stride step in
                       each dimension.
            
            
            Returns
            -------
            
            Tensor for convolutional layer.
            
        """
        
        
        # Decide convolutional layer architecture based on selected model
        if self.model_name == 'reparametrization':
            conv_layer = tfp.layers.Convolution3DReparameterization(filters = 64,
                                                                    kernel_size= filter_size,
                                                                    strides= strides,
                                                                    activation='relu',
                                                                    kernel_prior_fn = self.kernel_prior,
                                                                    kernel_posterior_fn = self.kernel_posterior,
                                                                    kernel_divergence_fn = self.divergence_fn,
                                                                    bias_prior_fn = self.bias_prior,
                                                                    bias_posterior_fn= self.bias_posterior,
                                                                    bias_divergence_fn = self.divergence_fn)(inputs)
        elif self.model_name == 'flipout-prior':
            conv_layer = tfp.layers.Convolution3DFlipout(filters=64, 
                                                         kernel_size= filter_size, 
                                                         strides= strides, 
                                                         activation= 'relu',
                                                         kernel_prior_fn= self.kernel_prior)(inputs)
        elif self.model_name == 'flipout-kl':
            conv_layer = tfp.layers.Convolution3DFlipout(filters=64, 
                                                         kernel_size= filter_size , 
                                                         strides= strides, 
                                                         activation= 'relu', 
                                                         kernel_divergence_fn= self.divergence_fn)(inputs)
        
        return conv_layer
    
    def dense3D(self, inputs, units, activation = 'relu'):
        
        """Build a 3D bayesian convolution layer .
        
            
            Parameters:
            -----------
            
            inputs:  input tensor, it should be the original input,
                      or the output from a previous layer.
            units: int, Integer or Long, dimensionality of the output space.
            
            Returns
            -------
            
            Tensor for convolutional layer.
            
        """
        
        
        # Decide convolutional layer architecture based on selected model
        if self.model_name == 'reparametrization':
            dense_layer = tfp.layers.DenseReparameterization(units= units,
                                                            activation= activation,
                                                            kernel_prior_fn = self.kernel_prior,
                                                            kernel_posterior_fn = self.kernel_posterior,
                                                            kernel_divergence_fn = self.divergence_fn,
                                                            bias_prior_fn = self.bias_prior,
                                                            bias_posterior_fn= self.bias_posterior,
                                                            bias_divergence_fn = self.divergence_fn)(inputs)
        elif self.model_name == 'flipout-prior':
            dense_layer = tfp.layers.DenseFlipout(units= units, 
                                                  activation= activation,
                                                  kernel_prior_fn= self.kernel_prior)(inputs)
        elif self.model_name == 'flipout-kl':
            dense_layer = tfp.layers.DenseFlipout(units= units, 
                                                  activation= activation, 
                                                  kernel_divergence_fn= self.divergence_fn)(inputs)
        
        return dense_layer
    
    
    
    def covLayer(self, inputs, filter_size, conv_strides, pool_strides):
        
        
        """Build a convolutional layer.
        
            
            Parameters:
            -----------
            
            inputs:  input tensor, it should be the original input,
                      or the output from a previous layer.
            
            Returns
            -------
            Tensor for convolutional layer.
        
        """
            
        # Decide type of pooling (max or avg)
        if self.pooling == "max":
            pool = MaxPooling3D
        elif self.pooling == "avg":
            pool = AveragePooling3D
        
        # Define the convolutional layer block
        conv_layer = self.conv3D(inputs, filter_size, conv_strides)
        conv_pool = pool(pool_size=2, strides= pool_strides)(conv_layer)
        conv_batch = BatchNormalization()(conv_pool)
        
        return conv_batch
    
    
    def bayesianModel(self, inputs):
        
        """Build the bayesian convolutional network.
        
            
            Parameters:
            -----------
            
            inputs:  input tensor, it should be the original input.
            
            Returns
            -------
            model: Keras Model instance, SBNN model.
        
        """
        
        #### Intensities Branch (preprocessed MRI scans)
        
        ## Input layer
        input_int = Input(shape= self.input_shape)
        
        # Convolutional Layers
        conv_int = self.covLayer(input_int, (3,3,3), (1,1,1), (1,1,1))
        conv_int = self.covLayer(conv_int, (5,5,5), (1,1,1), (2,2,2))
        conv_int = self.covLayer(conv_int, (7,7,7), (1,1,1), (3,3,3))
        conv_int = self.covLayer(conv_int, (9,9,9), (1,1,1), (4,4,4))
        
        # Fully Connected Layers
        conv_int = Flatten()(conv_int)
        conv_int= self.dense3D(conv_int, units = 512)
        conv_int = Dropout(self.drop_rate)(conv_int)
        conv_int= self.dense3D(conv_int, units = 128)
        conv_int = Dropout(self.drop_rate)(conv_int)
        
        #### Spatially Informed Branch (Segmented MRI scans)
        
        ## Input layers
        input_si = Input(shape= self.input_shape)
        
        # Convolutional Layers
        conv_si = self.covLayer(input_si, (3,3,3), (1,1,1), (1,1,1))
        conv_si = self.covLayer(conv_si, (5,5,5), (1,1,1), (2,2,2))
        conv_si = self.covLayer(conv_si, (7,7,7), (1,1,1), (3,3,3))
        conv_si = self.covLayer(conv_si, (9,9,9), (1,1,1), (4,4,4))
        
        # Fully Connected Layer
        conv_si = Flatten()(conv_si)
        conv_si= self.dense3D(conv_si, units = 512)
        conv_si = Dropout(self.drop_rate)(conv_si)
        conv_si= self.dense3D(conv_si, units = 128)
        conv_si = Dropout(self.drop_rate)(conv_si)
        
        # Fusion layer
        concat = Concatenate(axis = -1)([conv_int, conv_si])
        concat = self.dense3D(concat, units = 512)
        concat = Dropout(self.drop_rate)(concat)
        
        # Classification Layer
        if self.model_name == 'reparametrization':
            outputs = self.dense3D(concat, units = tfp.layers.OneHotCategorical.params_size(self.classes), activation = 'softmax')
            outputs = tfp.layers.OneHotCategorical(self.classes)(outputs)
        else:
            outputs = self.dense3D(concat, units = 5, activation = 'softmax')
            
        model = Model(inputs = [input_int, input_si], outputs = outputs)
        
        ## Output Model
        return model

if __name__ == "__main__":
    
    # A test to print model's architecture.
    model = SBNNModels().model

    initial_learning_rate = 0.0001
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate, 
                                                              decay_steps=100000, 
                                                              decay_rate=0.96, 
                                                              staircase=True)
    
    checkpoint_cb = ModelCheckpoint("Spatially-informed-Bayesian-Neural-Network.h5", 
                                                    save_best_only=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer = Adam(learning_rate = lr_schedule),
                  metrics=["acc"]
                  )
    
    model.summary()

    
        
        
        
        
        
        
        
        
        
        