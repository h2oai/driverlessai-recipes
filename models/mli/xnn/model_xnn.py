""" Explainable neural net """
import uuid
import os
import datatable as dt
import numpy as np
from h2oaicore.models import CustomTensorFlowModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count, loggerdata
from h2oaicore.systemutils import user_dir, remove, config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdebug
import functools

import numpy as np
import pandas as pd
import tensorflow as tf


class SplitLayer(tf.keras.layers.Layer):
    def __init__(self, splits, **kwargs):
        super(SplitLayer, self).__init__(**kwargs)
        self.splits = splits

    def build(self, input_shape):
        pass

    def call(self, input, **kwargs):
        return tf.split(input, self.splits, 1)

    def get_config(self):
        config = {'splits': self.splits}
        base_config = super(SplitLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CustomXNNModel(CustomTensorFlowModel):
    """
        TensorFlow-based Custom Model
    """
    _tensorflow = True
    _parallel_task = True
    _can_use_gpu = True
    _can_use_multi_gpu = True  
    
    _regression = True
    _binary = True
    _multiclass = False
    _display_name = "Explainable Neural Network"
    _description = "xnn"
    _modules_needed_by_name = ['shap', 'scipy', 'pandas', 'matplotlib']

    _is_reproducible = False

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def acceptance_test_timeout():
        return 45.0

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        self.params = dict(random_state=kwargs.get("random_state", 1234),
                           n_estimators=min(kwargs.get("n_estimators", 100), 1000),
                           n_jobs=self.params_base.get('n_jobs', max(1, physical_cores_count)),
                           batch_size=1024, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True,
                           arch = [10, 6], lr = 0.01)

    def mutate_params(self, accuracy=10, **kwargs):
        
        if accuracy > 8:
            estimators_list = [10000, 20000]
            batch_size = [32, 128, 1024]
            arch = [[4, 4], [8, 8], [12, 8], [20,12], [40,24]]
            lr = [0.01, 0.001]
        elif accuracy >= 5:
            estimators_list = [10000]
            batch_size = [32, 128, 1024]
            arch = [[8, 8], [20,12], [40,24]]
            lr = [0.01, 0.001]
        else:
            estimators_list = [10000]
            batch_size = [128, 1024]
            arch = [[8, 6], [20,12]]
            lr = [0.1, 0.01]
            
        # Modify certain parameters for tuning
        self.params["n_estimators"] = int(np.random.choice(estimators_list))
        self.params["batch_size"] =int(np.random.choice(batch_size))
        self.params["arch"] = arch[np.random.choice(range(len(arch)))]
        self.params["lr"] = int(np.random.choice(lr))


    def _create_tmp_folder(self, logger):
        # Create a temp folder to store xnn files 
        # Set the default value without context available (required to pass acceptance test)
        tmp_folder = os.path.join(user_dir(), "%s_xnn_model_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_xnn_model_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "XNN was denied temp folder creation rights")
            tmp_folder = os.path.join(user_dir(), "%s_xnn_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "XNN temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_xnn_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(user_dir(), "%s_xnn_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "XNN temp folder {}".format(tmp_folder))
        return tmp_folder

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        
        # Get column names
        orig_cols = list(X.names)
        
        
        from h2oaicore.tensorflow_dynamic import got_cpu_tf, got_gpu_tf
        import tensorflow as tf
        import shap
        import scipy
        import pandas as pd

        self.setup_keras_session()

        import h2oaicore.keras as keras
        import matplotlib.pyplot as plt

        if not hasattr(self, 'save_model_path'):
            model_id = str(uuid.uuid4())[:8]
            self.save_model_path = os.path.join(user_dir(), "custom_xnn_model.hdf5")
        
        np.random.seed(self.random_state)

        my_init = keras.initializers.RandomUniform(seed=self.random_state)
        
        
        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id,
                                            tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)

        # Set up temp folter
        tmp_folder = self._create_tmp_folder(logger)

        # define base model
        def xnn_initialize(features, ridge_functions=3, arch=[20,12], learning_rate=0.01, bg_samples=100, beta1=0.9, beta2=0.999, dec=0.0, ams=True, bseed=None, is_categorical=False):
            
            #
            # Prepare model architecture
            #
            # Input to the network, our observation containing all the features
            input = keras.layers.Input(shape=(features,), name='main_input')
                      
            # Record current column names
            loggerinfo(logger, "XNN LOG")
            loggerdata(logger, "Feature list:")
            loggerdata(logger, str(orig_cols))
            
            # Input to ridge function number i is the dot product of our original input vector times coefficients
            ridge_input = keras.layers.Dense(ridge_functions, name="projection_layer",
                                             activation='linear')(input)
            
            ridge_networks = []
            # Each subnetwork uses only 1 neuron from the projection layer as input so we need to split it
            ridge_inputs = SplitLayer(ridge_functions)(ridge_input)
            for i, ridge_input in enumerate(ridge_inputs):
                # Generate subnetwork i
                mlp = _mlp(ridge_input, i, arch)
                ridge_networks.append(mlp)
                        
            added = keras.layers.Concatenate(name='concatenate_1')(ridge_networks)
            
            # Add the correct output layer for the problem
            if is_categorical:
                out = keras.layers.Dense(1, activation='sigmoid', input_shape= (ridge_functions, ), name='main_output')(added)
            else:
                out = keras.layers.Dense(1, activation='linear', input_shape= (ridge_functions, ), name='main_output')(added)
                
            model = keras.models.Model(inputs=input, outputs=out)
                      
            optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=dec, amsgrad=ams)
            
            # Use the correct loss for the problem
            if is_categorical:
                model.compile(loss={'main_output': 'binary_crossentropy'}, optimizer=optimizer)
            else:
                model.compile(loss={'main_output': 'mean_squared_error'}, optimizer=optimizer)
                
            return model
                      
            
        def _mlp(input, idx, arch=[20,12], activation='relu'):
            # Set up a submetwork
            
            # Hidden layers
            mlp = keras.layers.Dense(arch[0], activation=activation, name='mlp_{}_dense_0'.format(idx), kernel_initializer=my_init)(input)
            for i, layer in enumerate(arch[1:]):
                mlp = keras.layers.Dense(layer, activation=activation, name='mlp_{}_dense_{}'.format(idx, i+1), kernel_initializer=my_init)(mlp)
             
            # Output of the MLP
            mlp = keras.layers.Dense(1, 
                                     activation='linear', 
                                     name='mlp_{}_dense_last'.format(idx), 
                                     kernel_regularizer=keras.regularizers.l1(1e-3),
                                     kernel_initializer=my_init)(mlp)
            return mlp
        
        def get_shap(X, model):
            # Calculate the Shap values
            np.random.seed(24)
            bg_samples = min(X.shape[0], 1000)
            
            if isinstance(X, pd.DataFrame):
                background = X.iloc[np.random.choice(X.shape[0], bg_samples, replace=False)]
            else:
                background = X[np.random.choice(X.shape[0], bg_samples, replace=False)]
    
            # Explain predictions of the model on the subset
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X)
            
        
            # Return the mean absolute value of each shap value for each dataset
            xnn_shap = np.abs(shap_values[0]).mean(axis=0)

            return xnn_shap
               
        # Initialize the xnn's
        features = X.shape[1]
        orig_cols = list(X.names)
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            
            self.is_cat = True
            xnn1 = xnn_initialize(features=features, ridge_functions=features,arch=self.params["arch"], 
                                  learning_rate=self.params["lr"], beta1=self.params["beta_1"], beta2=self.params["beta_1"], 
                                  dec=self.params["decay"], ams=self.params["amsgrad"], is_categorical= self.is_cat)
            xnn = xnn_initialize(features=features, ridge_functions=features,arch=self.params["arch"], learning_rate=self.params["lr"], 
                                  beta1=self.params["beta_1"], beta2=self.params["beta_1"], 
                                  dec=self.params["decay"], ams=self.params["amsgrad"], is_categorical= self.is_cat)
        else:
            self.is_cat = False
            xnn1 = xnn_initialize(features=features, ridge_functions=features, arch=self.params["arch"], learning_rate=self.params["lr"],
                                   beta1=self.params["beta_1"], beta2=self.params["beta_1"], 
                                  dec=self.params["decay"], ams=self.params["amsgrad"], is_categorical= self.is_cat)
            xnn = xnn_initialize(features=features, ridge_functions=features, arch=self.params["arch"], learning_rate=self.params["lr"], 
                                  beta1=self.params["beta_1"], beta2=self.params["beta_1"], 
                                  dec=self.params["decay"], ams=self.params["amsgrad"], is_categorical= self.is_cat)

        # Replace missing values with a value smaller than all observed values
        self.min = dict()
        for col in X.names:
            XX = X[:, col]
            self.min[col] = XX.min1()
            if self.min[col] is None or np.isnan(self.min[col]):
                self.min[col] = -1e10
            else:
                self.min[col] -= 1
            XX.replace(None, self.min[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_numpy()

        inputs = {'main_input': X}
        validation_set=0
        verbose = 0
        
        # Train the neural network once with early stopping and a validation set
        history = keras.callbacks.History()
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')

        history=xnn1.fit(inputs, y, epochs=self.params["n_estimators"], batch_size=self.params["batch_size"], 
                         validation_split=0.3, verbose=verbose, callbacks=[history, es])
        
        # Train again on the full data
        number_of_epochs_it_ran = len(history.history['loss'])
        
        xnn.fit(inputs, y, epochs=number_of_epochs_it_ran, batch_size=self.params["batch_size"], 
                        validation_split=0.0, verbose=verbose)

        # Get the mean absolute Shapley values
        importances = np.array(get_shap(X, xnn))
        
        int_output = {}
        int_weights = {}
        int_bias = {}
        int_input = {}
        
        original_activations = {}
        
        x_labels = list(map(lambda x: 'x' + str(x), range(features)))
        
        intermediate_output = []
        
        # Record and plot the projection weights
        # 
        weight_list = []
        for layer in xnn.layers:
        
            layer_name = layer.get_config()['name']
            if layer_name != "main_input":
                print(layer_name)
                weights = layer.get_weights()
                
                
                # Record the biases
                try:
                    bias = layer.get_weights()[1]
                    int_bias[layer_name] = bias
                except:
                    print("No Bias")
                    
                               
                # Record outputs for the test set
                intermediate_layer_model = keras.models.Model(inputs=xnn.input, outputs=xnn.get_layer(layer_name).output)
                
                # Record the outputs from the training set
                if self.is_cat and (layer_name == 'main_output'):
                    original_activations[layer_name] = scipy.special.logit(intermediate_layer_model.predict(X))   
                    original_activations[layer_name + "_p"] = intermediate_layer_model.predict(X)
                else:
                    original_activations[layer_name] = intermediate_layer_model.predict(X)        
            
            
                # Record other weights, inputs, and outputs
                int_weights[layer_name] = weights
                int_input[layer_name] = layer.input
                int_output[layer_name] = layer.output

                       
            # Plot the projection layers    
            if "projection_layer" in layer.get_config()['name']:
                
                #print(layer.get_config()['name'])
                
                # Record the weights for each projection layer
                weights = [np.transpose(layer.get_weights()[0])]
            
                weight_list2=[]
                for i, weight in enumerate(weights[0]):
                    weight_list.append(weight)
                    weight_list2.append(list(np.reshape(weight, (1,features))[0]))
            
                    # Plot weights
                    plt.bar(orig_cols, abs(np.reshape(weight, (1,features))[0]), 1, color="blue")
                    plt.ylabel("Coefficient value")
                    plt.title("Projection Layer Weights {}".format(i), fontdict={'fontsize': 10})
                    plt.xticks(rotation=90)
                    plt.show()
                    plt.savefig(os.path.join(tmp_folder, 'projection_layer_' + str(i)+'.png'), bbox_inches = "tight")
                    plt.clf()
            
            if "main_output" in layer.get_config()['name']:
                weights_main = layer.get_weights()
                print(weights_main)
            
            
        pd.DataFrame(weight_list2).to_csv(os.path.join(tmp_folder, "projection_data.csv"), index=False)
        
        intermediate_output = []
        
        for feature_num in range(features):
            intermediate_layer_model = keras.models.Model(inputs=xnn.input,
                                                          outputs=xnn.get_layer('mlp_'+str(feature_num)+'_dense_last').output)
            intermediate_output.append(intermediate_layer_model.predict(X))
        
        
        # Record and plot the ridge functions
        ridge_x = []
        ridge_y = []
        for weight_number in range(len(weight_list)):
            
            ridge_x.append(list(sum(X[:, ii]*weight_list[weight_number][ii] for ii in range(features))))
            ridge_y.append(list(intermediate_output[weight_number]))
        
            plt.plot(sum(X[:, ii]*weight_list[weight_number][ii] for ii in range(features)), intermediate_output[weight_number], 'o')
            plt.xlabel("Input")
            plt.ylabel("Subnetwork " + str(weight_number))
            plt.title("Ridge Function {}".format(i), fontdict={'fontsize': 10})
            plt.show() 
            plt.savefig(os.path.join(tmp_folder, 'ridge_'+str(weight_number)+'.png'))
            plt.clf()
            
        
        # Output the ridge function importance    
        weights2 = np.array([item[0] for item in list(weights)[0]])
        
        output_activations = np.abs(np.array([item*weights2 for item in list(original_activations["concatenate_1"])])).mean(axis=0)
        loggerinfo(logger, str(output_activations))
        pd.DataFrame(output_activations).to_csv(os.path.join(tmp_folder, "ridge_weights.csv"), index=False)


        plt.bar(x_labels, output_activations , 1, color="blue")
        plt.xlabel("Ridge function number")
        plt.ylabel("Feature importance")
        plt.title("Ridge function importance", fontdict={'fontsize': 10})
        plt.show()
        plt.savefig(os.path.join(tmp_folder, 'Ridge_function_importance.png')) 
        
        
        pd.DataFrame(ridge_y).applymap(lambda x:x[0]).to_csv(os.path.join(tmp_folder, "ridge_y.csv"), index=False)
        pd.DataFrame(ridge_x).to_csv(os.path.join(tmp_folder, "ridge_x.csv"), index=False)           
        
        pd.DataFrame(orig_cols).to_csv(os.path.join(tmp_folder, "input_columns.csv"), index=False)
        

        self.set_model_properties(model=xnn,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=self.params['n_estimators'])
        
    def predict(self, X, **kwargs):
        np.random.seed(self.random_state)
        
        X = dt.Frame(X)
        for col in X.names:
            XX = X[:, col]
            XX.replace(None, self.min[col])
            X[:, col] = XX
        model, _, _, _ = self.get_model_properties()
        X = X.to_numpy()

        preds = model.predict(X)
        
        if self.is_cat == 1:
            preds[preds < 0.0] = 0
            preds[preds  > 1.0] = 1.0        

        return preds

    def get_model(self):
        import h2oaicore.keras as keras
        with keras.utils.CustomObjectScope({'SplitLayer': SplitLayer}):
            if self.model is None:
                self.did_get_model = True
                assert self.model_bytes is not None
                self.pre_get_model()
                self.model = self.picfunc().loads(self.model_bytes)
            return self.model

