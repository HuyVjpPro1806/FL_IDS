"""Shared DNN, LSTM, CNN builders and common TFF Adam optimizer."""
from common import config

MODEL_KINDS = ('dnn', 'lstm', 'cnn')
MODEL_BACKENDS = {'dnn':'Dense/ReLU on GPU',
    'lstm':'CuDNNLSTM (GPU-only, tanh/sigmoid)',
    'cnn':'Conv1D/ReLU CNN on GPU'}


def make_model(kind,params):
    from runtime import tf
    if kind == 'cnn':
        return _cnn_model(kind, params)
    if kind not in MODEL_KINDS:
        raise ValueError(kind)
    widths=[int(v) for v in params['architecture'].split('-')]
    layers=[tf.keras.layers.InputLayer(input_shape=(39,))]
    if kind=='lstm': layers.append(tf.keras.layers.Reshape((1,39)))
    for i,units in enumerate(widths):
        args=dict(kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42+i))
        if kind=='dnn': layers.append(tf.keras.layers.Dense(units,activation='relu',**args))
        elif kind=='lstm':
            # TF 2.14 GPU-only layer: emits CudnnRNN ops even in TFF graphs.
            layers.append(tf.compat.v1.keras.layers.CuDNNLSTM(units,return_sequences=i<2,
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=52+i),**args))
        else: raise ValueError(kind)
        if i<2: layers.append(tf.keras.layers.Dropout(params['dropout'],seed=62+i))
    layers.append(tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=72)))
    return tf.keras.Sequential(layers,name=kind)


def _cnn_model(kind, params):
    from runtime import tf

    if kind != 'cnn':
        raise ValueError('CNN workflow accepts only cnn.')
    widths = [int(value) for value in params['architecture'].split('-')]
    if len(widths) != 3 or min(widths) <= 0:
        raise ValueError('CNN architecture is conv1_filters-conv2_filters-dense_units.')
    kernel = int(params['kernel_size'])
    if kernel not in (3, 5):
        raise ValueError('CNN kernel_size must be 3 or 5.')
    dropout = float(params['dropout'])
    if not 0 <= dropout < 1:
        raise ValueError('Invalid CNN dropout.')
    first, second, dense = widths
    # Feature order stays exactly as prepared for DNN/LSTM. These are features,
    # not consecutive packets or time steps. Flatten retains feature positions.
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(39,)),
        tf.keras.layers.Reshape((39, 1)),
        tf.keras.layers.Conv1D(first, kernel, padding='same', activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=42)),
        tf.keras.layers.MaxPooling1D(pool_size=2, padding='same'),
        tf.keras.layers.Dropout(dropout, seed=62),
        tf.keras.layers.Conv1D(second, kernel, padding='same', activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=43)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense, activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=44)),
        tf.keras.layers.Dropout(dropout, seed=63),
        tf.keras.layers.Dense(1, activation='sigmoid',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=72)),
    ], name='cnn')


def adam(rate):
    from runtime import tff
    choices = config()['implementation_choices_not_specified_in_paper']
    return tff.learning.optimizers.build_adam(rate, beta_1=choices['adam_beta1'],
        beta_2=choices['adam_beta2'], epsilon=choices['adam_epsilon'])

