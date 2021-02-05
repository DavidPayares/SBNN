# Neurodegenerative Diseases Classification
# Train 3D Spatially Informed Bayesian CNN.
# Author: David Payares.
# Copyleft: MIT Licience.


import os
import shutil

from keras import backend as K
from SBNN_models import SBNNModels
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

class SBNNTrain(object):
    
    def __init__(self, name = 'reparametrization', weights_dir, logs_dir, save_best_weights = True):
        
        
        """ Training the Spatially Informed Bayesian Neural Network.
        
            
            Parameters:
            -----------
            weights_dir: string, directory path to save the
                                trained model weights.
            logs_dir: string, directory path to save the
                             logs of training process.
            save_best_weights: boolean, to whether save or not the model with best
                                 validation accuracy. Default is True.
            
            
            Returns
            -------
            Weights of the train model.
        
        """
        
        # Name for saving weights and logs
        self.name = name
        
        # Dataset: training, validation and test
        self.data = None
        
        # Best model
        self.save_best_weights = save_best_weights
        
        # Create folder for saving weights and logs
        self.weights_dir = os.path.join(weights_dir, 'SBNN-' ,name)
        self.createDir(self.weights_dir)
        
        self.logs_dir = os.path.join(logs_dir, 'SBNN-' ,name)
        self.createDir(self.logs_dir)
            
        # Initialize files' names for weights in the last and best epoch
        self.last_weights_path = os.path.join(self.weights_dir, "last.h5")
        self.best_weights_path = os.path.join(self.weights_dir, "best.h5")

        # CSV file path for writing learning curves
        self.curves_path = os.path.join(self.logs_dir, "curves.csv")
        
        # Load parameters and hyperparameters
        self.paras = self.loadModelParameters()
        
        
        return
    
        def loadModelParameters(self):
            
            """
                Load default parameters and hyperparameters.
            """

            # Parameters to build the  model
            self.model_name = self.name

            # Parameters to train model
            self.optimizer = "adam"
            self.lr_init_rate = 0.0001
            self.epochs_num = 100
        
        return
        
        
        def loadModel(self):
            
            """ Load Spatially informed Bayesian Neural Network model.
            
                Returns
                -------
                Model.
        
            """
            
            self.model = SBNNModels(model_name = self.model_name).model
            
            return
        
        
        def setLRScheduler(self):
            
            """ Set Learning rate scheduler for training process.
        
                Returns
                -------
                learning rate.
        
            """
            
            lr_schedule = schedules.ExponentialDecay(self.lr_init_rate, 
                                                     decay_steps=100000, 
                                                     decay_rate=0.96, 
                                                     staircase=True)
            return lr_schedule
            
        def loadOptimizer(self):
            
            """ Set Optimizer for training process.
                Default is Adam
        
                Returns
                -------
                Optimizer.
        
    
            """
            
            if self.optimizer == "adam":
                self.opt = Adam(learning_rate = setLRScheduler())
            
            return
        
        def loadCallBacks(self):
        
            """ Set callback functions while training model.
            
            Returns
            -------
            Callbacks.
            
            """
            
            # Save learning curves
            csv_logger = CSVLogger(self.curves_path,
                                   append=True, 
                                   separator=",")
            
            self.callbacks = [csv_logger]
            
            # Save model
            if self.save_best_only:
                checkpoint = ModelCheckpoint(filepath = self.best_weights_path, 
                                             save_best_only=True)
                self.callbacks += [checkpoint]
            
            return
        
        def modelEvaluation(self):
            
            """ Prints out metrics of acurracy and loss for the 
                training, validation and test dataset.
            
            Returns
            -------
            Metrics.
            
            """
            
            # Compute metrics
            def computeMetrics(data_set, name_dataset):
                score = self.model.evaluate(data_set)
                print(name_dataset + " Dataset: Loss: {0:.4f}, Accuracy: {1:.4f}".format(
                  score[0], score[1]))
                
                return
            
            datasets_names = ["Training", "Validation", "Test"]
            datasets_data = [self.data.training, self.data.val , self.data.test]
            
            for n,d in zip(datasets_names, datasets_data):
                computeMetrics(n,d)
                
            return
                        
        def trainModel(self, data):
            
            """ Train the Spatially Informed Bayesian Neural Network model.
            
            Parameters:
            -----------
            data: a SBNNDataset instance for training, validation and test dataset.
            
            
            Returns
            -------
            A tranined model.
        
            """
            
            print("\nTraining the model.\n")
            
            self.data = data
            
            # Configurate the model
            self.loadModel()
            self.loadOptimizer()
            
            # loss likelihood function
            def negative_log_likelihood(y_true, y_pred):
                return -y_pred.log_prob(y_true)
            
            # Compile model
            self.model.compile(loss= negative_log_likelihood, 
                               optimizer = self.opt,
                               metrics = ["accuracy"])
            
            # Display model summary
            self.model.summary()
            
            # Load Callbacks
            self.loadCallBacks()
            
            ### Train the model
            self.model.fit(self.data.training, 
                           validation_data = self.data.val, 
                           epochs=  self.epochs_num, 
                           shuffle = True , 
                           verbose = 2 , 
                           callbacks = self.callbacks)
            
            # Save last epoch model
            self.model.save(self.last_weights_path)
            
            # print metrics
            self.modelEvaluation()
            
            #Clear TF session
            K.clear_session()
            
            return
            
            
        @staticmethod
        def createDir(path, rm=True):
            
            """ Create a new directory.
        
                Parameters:
                -----------
                dir_path: string, path of new directory.
                rm: boolean, remove existing directory or not.
            
                Returns
                -------
                new folder.
        
            """
        
        # check if dicrectory exists
        
            if os.path.isdir(path):
                if rm:
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path)
            else:
                os.makedirs(dir_path)
            return