"""Binary classifiers; client Adam and direct server averaging."""
MODEL_KINDS = ('dnn', 'cnn')
MODEL_BACKENDS = {'dnn': 'Dense/ReLU', 'cnn': 'Conv1D on the fixed feature axis'}


def make_model(kind, params, task='binary', seed=42):
    if task != 'binary': raise ValueError('Only binary classification is enabled.')
    from runtime import tf
    widths = [int(v) for v in params['architecture'].split('-')]
    if not widths or min(widths) < 1 or not 0 <= params['dropout'] < 1:
        raise ValueError('Invalid model parameters.')
    layers = [tf.keras.layers.InputLayer(input_shape=(39,))]
    if kind == 'dnn':
        for i, width in enumerate(widths):
            layers.append(tf.keras.layers.Dense(width, activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+i)))
            if i < len(widths)-1:
                layers.append(tf.keras.layers.Dropout(params['dropout'], seed=seed+20+i))
    elif kind == 'cnn':
        if len(widths) != 3 or params['kernel_size'] not in (3, 5): raise ValueError('Invalid CNN shape.')
        layers.extend([
            tf.keras.layers.Reshape((39, 1)),
            tf.keras.layers.Conv1D(widths[0], params['kernel_size'], padding='same', activation='relu',
                kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
            tf.keras.layers.MaxPooling1D(2, padding='same'),
            tf.keras.layers.Dropout(params['dropout'], seed=seed+20),
            tf.keras.layers.Conv1D(widths[1], params['kernel_size'], padding='same', activation='relu',
                kernel_initializer=tf.keras.initializers.HeUniform(seed=seed+1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(widths[2], activation='relu',
                kernel_initializer=tf.keras.initializers.HeUniform(seed=seed+2)),
            tf.keras.layers.Dropout(params['dropout'], seed=seed+21)])
    else: raise ValueError(kind)
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+30)))
    return tf.keras.Sequential(layers, name=f'{kind}_{task}')


def adam(rate, settings):
    from runtime import tff
    return tff.learning.optimizers.build_adam(rate, beta_1=settings['adam_beta1'],
        beta_2=settings['adam_beta2'], epsilon=settings['adam_epsilon'])
