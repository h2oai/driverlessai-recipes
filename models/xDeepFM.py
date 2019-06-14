from h2oaicore.models import BaseCustomModel
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal, glorot_uniform
from tensorflow.python.keras.layers import Layer, Activation
from tensorflow.python.keras.regularizers import l2
import random
import gc
import numpy as np


class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix and bias.
    """

    def __init__(self, hidden_units, activation, l2_reg=0.0, **kwargs):
        assert len(hidden_units) > 0
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = [keras.layers.Dense(units, activation=self.activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                                   bias_regularizer=keras.regularizers.l2(self.l2_reg)) for units in self.hidden_units]
        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for layer in self.layers:
            deep_input = layer(deep_input)

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


class CIN(Layer):
    """Compressed Interaction Network used in xDeepFM. Original code https://github.com/Leavingseason/xDeepFM.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .

      Arguments
        - **layer_size** : list of int.Feature maps in each layer.

        - **activation** : activation function used on feature maps.

        - **seed** : A Python integer to use as random seed.

      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, layer_size, activation, split_half=True, l2_reg=1e-5, seed=1024, **kwargs):
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed
        self.split_half = split_half
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        self.field_nums = [input_shape[1].value]
        self.filters = []
        self.bias = []
        for i, size in enumerate(self.layer_size):

            self.filters.append(self.add_weight(name='filter' + str(i),
                                                shape=[1, self.field_nums[-1]
                                                       * self.field_nums[0], size],
                                                dtype=tf.float32, initializer=glorot_uniform(seed=self.seed + i),
                                                regularizer=l2(self.l2_reg)))

            self.bias.append(self.add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                             initializer=tf.keras.initializers.Zeros()))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        self.activation_layers = [self.activation for _ in self.layer_size]

        super(CIN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        dim = inputs.get_shape()[-1].value
        hidden_nn_layers = [inputs]
        final_result = []

        for idx, layer_size in enumerate(self.layer_size):
            dot_result = tf.einsum('imj,inj->imnj', hidden_nn_layers[0], hidden_nn_layers[-1])
            dot_result = tf.reshape(
                    dot_result, shape=[-1, self.field_nums[0] * self.field_nums[idx], dim])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID', data_format='NCW')


            curr_out = tf.nn.bias_add(tf.expand_dims(curr_out, axis=-1), self.bias[idx], data_format='NCHW')
            curr_out = tf.squeeze(curr_out, axis=-1)

            curr_out = self.activation_layers[idx](curr_out)

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)

        result = tf.reduce_sum(result, -1, keep_dims=False)

        return result

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(
                self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        return (None, featuremap_num)

    def get_config(self, ):

        config = {'layer_size': self.layer_size, 'split_half': self.split_half, 'activation': self.activation,
                  'seed': self.seed}
        base_config = super(CIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class XDeepFM(BaseCustomModel):
    _regression = False
    _binary = True
    _multiclass = False

    _boosters = ['xDeepFM']
    _display_name = "eXtreme Deep Factorization Machine"
    _description = "eXtreme Deep Factorization Machine Model. Not adviced if the data is smaller than 1M rows"

    def set_default_params(self, **kwargs):
        embedding = kwargs['embedding']
        l2_embedding = kwargs['l2_embedding']
        dnn_size = kwargs['dnn_size']
        l2_dnn = kwargs['l2_dnn']
        cin_size = kwargs['cin_size']
        l2_cin = kwargs['l2_cin']

        self.params = {
            'embedding': embedding,
            'l2_embedding': l2_embedding,
            'dnn_size': dnn_size,
            'l2_dnn': l2_dnn,
            'cin_size': cin_size,
            'l2_cin': l2_cin
        }

    def mutate_params(self, **kwargs):
        list_embedding = [4, 8, 12, 16]
        list_embedding_l2 = [5e-6, 1e-6, 5e-7, 1e-7]
        list_dnn_l2 = [0.0, 1e-7, 1e-6]
        list_dnn_size = [(64, 64), (128, 128), (256, 256), (128, 128, 128)]
        list_cin_size = [(256, 128), (128, 64, 32), (128, 64, 32, 16), (64, 32, 16)]
        list_cin_l2 = [5e-6, 1e-6, 5e-7]

        self.params = {
            'embedding': random.choice(list_embedding),
            'l2_embedding': random.choice(list_embedding_l2),
            'dnn_size': random.choice(list_dnn_size),
            'l2_dnn': random.choice(list_dnn_l2),
            'cin_size': random.choice(list_cin_size),
            'l2_cin': random.choice(list_cin_l2)
        }


    def _build_model(self, numerical_features, cat_features, embedding_size, cin_size, dnn_size, dnn_activation=keras.activations.relu, l2_reg_dnn=0.0, l2_reg_cin=1e-5,
                 l2_embedding=1e-4, seed=0, init_std=0.01):

        numerical_input = list(map(lambda x: keras.layers.Input(shape=(1,), name='numerical_{0}'.format(x), dtype=tf.float32), numerical_features))
        cat_input = list(map(lambda x: keras.layers.Input(shape=(1,), name="cat_{0}".format(x[0]), dtype=tf.int32), cat_features))

        embeding_input = []
        for idx, [name, size] in enumerate(cat_features):
            embedding_layer = tf.keras.layers.Embedding(size, embedding_size,
                                                  name='emb_' + name,
                                                        embeddings_initializer=keras.initializers.RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=keras.regularizers.l2(l2_embedding),
                                                        )
            embeding = embedding_layer(cat_input[idx])

            embeding_input.append(embeding)


        for idx, name in enumerate(numerical_features):
            x = keras.layers.Dense(embedding_size, kernel_regularizer=keras.regularizers.l2(l2_embedding),
                                   bias_regularizer=keras.regularizers.l2(l2_embedding))(numerical_input[idx])
            x = keras.layers.Reshape([1, embedding_size])(x)
            embeding_input.append(x)

        concat = keras.layers.Concatenate(axis=1)(embeding_input)

        cin_out = CIN(cin_size, keras.activations.relu, l2_reg=l2_reg_cin)(concat)
        deep_input = tf.keras.layers.Flatten()(concat)

        dnn = DNN(dnn_size, dnn_activation, l2_reg_dnn)

        deep_out = dnn(deep_input)
        deep_logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(deep_out)

        logit = keras.layers.add([deep_logit, cin_out])

        predictions = keras.layers.Dense(1, activation='sigmoid', use_bias=False)(logit)

        return tf.keras.models.Model(cat_input + numerical_input, outputs=predictions)

    def fit(self, X, y, **kwargs):
        self.cat_feature_names = []
        self.num_feature_names = []
        self.mm_scalers = {}
        self.label_encoders = {}
        cat_feature_size = []


        for col in X.columns:
            if X.dtypes[col] in [np.float32, np.float64]:
                mms = MinMaxScaler(feature_range=(0, 1))
                #TODO: add extra field indicating field is nan
                X[col] = X[col].fillna(0)
                X[col] = mms.fit_transform(X[col].values.reshape(-1, 1)).astype(np.float32)
                self.num_feature_names.append(col)
                self.mm_scalers[col] = mms
            else:
                label_encoder = LabelEncoder()
                X[col] = label_encoder.fit_transform(X[col]).astype(np.int32)
                self.num_feature_names.append(col)
                self.label_encoders[col] = label_encoder
                cat_feature_size.append((col, len(label_encoder.classes_)))
            gc.collect()

        self.model = self._build_model(self.num_feature_names, cat_feature_size, embedding_size=self.params['embedding'], l2_embedding=self.params['l2_embedding'],
                                  dnn_size=self.params['dnn_size'], l2_reg_dnn=self.params['l2_dnn'], cin_size=self.params['cin_size'], l2_reg_cin=self.params['l2_cin'])

        self.model.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=['binary_crossentropy'])


        train_model_cat_input = [X[feature].values for feature in self.cat_feature_names]
        train_model_numerical_input = [X[feature].values for feature in self.num_feature_names]
        train_target = y

        input = train_model_cat_input + train_model_numerical_input

        self.model.fit(input, train_target, batch_size=4*1024, epochs=10, verbose=2)

    def predict(self, X, **kwargs):
        for col in X.columns:
            if X.dtypes[col] in [np.float32, np.float64]:
                X[col] = X[col].fillna(0)
                X[col] = self.mm_scalers[col].transform(X[col].values.reshape(-1, 1)).astype(np.float32)
            else:
                X[col] = self.label_encoders[col].transform(X[col]).astype(np.int32)
            gc.collect()

        return self.model.predict(X, batch_size=4*1024)
