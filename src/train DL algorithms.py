def create_dl_model(model_name):

    inputs = layers.Input(shape=(INPUT_DIM,))

    # ========================================================
    # MLP
    # ========================================================
    if model_name == "MLP":

        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)

    # ========================================================
    # CNN
    # ========================================================
    elif model_name == "CNN":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.Conv1D(32,2,activation='relu')(x)
        x = layers.MaxPooling1D()(x)
        x = layers.Flatten()(x)

    # ========================================================
    # RNN
    # ========================================================
    elif model_name == "RNN":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.SimpleRNN(64)(x)

    # ========================================================
    # LSTM
    # ========================================================
    elif model_name == "LSTM":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.LSTM(64)(x)

    # ========================================================
    # GRU
    # ========================================================
    elif model_name == "GRU":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.GRU(64)(x)

    # ========================================================
    # CNN_LSTM
    # ========================================================
    elif model_name == "CNN_LSTM":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.Conv1D(32,2,activation='relu')(x)
        x = layers.MaxPooling1D()(x)
        x = layers.LSTM(64)(x)

    # ========================================================
    # CNN_GRU
    # ========================================================
    elif model_name == "CNN_GRU":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.Conv1D(32,2,activation='relu')(x)
        x = layers.MaxPooling1D()(x)
        x = layers.GRU(64)(x)

    # ========================================================
    # CNN_RNN
    # ========================================================
    elif model_name == "CNN_RNN":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.Conv1D(32,2,activation='relu')(x)
        x = layers.MaxPooling1D()(x)
        x = layers.SimpleRNN(64)(x)

    # ========================================================
    # LSTM_RNN
    # ========================================================
    elif model_name == "LSTM_RNN":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.SimpleRNN(32)(x)

    # ========================================================
    # LSTM_GRU
    # ========================================================
    elif model_name == "LSTM_GRU":

        x = layers.Reshape((INPUT_DIM,1))(inputs)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.GRU(32)(x)

    else:
        raise ValueError(model_name)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
