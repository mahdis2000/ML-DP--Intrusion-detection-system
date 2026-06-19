#Main code for data preprocessing
for key, filename in attack_files.items():

    path = os.path.join(BASE_PATH, filename)

    print("\nLoading:", filename)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_cols = feature_cols + ['Label']
    df = df[required_cols]

    # clean
    df = df.replace([np.inf,-np.inf], np.nan)
    df = df.dropna()

    # binary label
    df['Binary_Label'] = np.where(
        df['Label'].str.contains(
            'BENIGN',
            case=False
        ),
        0,
        1
    )

    attack_datasets[key] = df

    print(
        attack_names[key],
        "| shape:", df.shape,
        "| attack:", df['Binary_Label'].sum(),
        "| benign:", (df['Binary_Label']==0).sum()
    )

# ============================================================
# SCALE + SPLIT
# ============================================================

processed_data = {}

for key, df in attack_datasets.items():

    print("\nProcessing:", attack_names[key])
    X = df[feature_cols].values
    y = df['Binary_Label'].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # 70 train
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )
    # 15 val / 15 test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )
    processed_data[key] = {

        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }
    print(
        "Train:", X_train.shape,
        "| Val:", X_val.shape,
        "| Test:", X_test.shape
    )
