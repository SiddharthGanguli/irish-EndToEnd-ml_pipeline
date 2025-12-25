from dataspliting import TrainTestSplit  # whatever your filename is

RAW_DATA_PATH = "/Users/siddharthaganguli/Desktop/irish/data/preprocessed_data/iris_preprocessed.csv"
SPLIT_OUTPUT_PATH = "/Users/siddharthaganguli/Desktop/irish/data/split_data"
TARGET_COLUMN = "species" 

def main():
    splitter = TrainTestSplit(
        input_path=RAW_DATA_PATH,
        output_path=SPLIT_OUTPUT_PATH,
        target_column=TARGET_COLUMN
    )

    splitter.load_data()
    X_train, X_test, y_train, y_test = splitter.split()
    paths = splitter.save_splits(X_train, X_test, y_train, y_test)

    print("Train/Test split completed. Saved files:")
    for path in paths:
        print(path)

if __name__ == "__main__":
    main()
