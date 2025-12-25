from datapreprocessing import Preprocessing

RAW_DATA_PATH = "/Users/siddharthaganguli/Desktop/irish/data/raw/Iris_Data.csv"
PROCESSED_DATA_PATH = "/Users/siddharthaganguli/Desktop/irish/data/preprocessed_data"

def main():
    prep = Preprocessing(
        input_path=RAW_DATA_PATH,
        output_path=PROCESSED_DATA_PATH
    )

    prep.load_data()
    prep.encode_features()
    output_file = prep.save_data()

    print(f"Preprocessing completed. Saved to: {output_file}")

if __name__ == "__main__":
    main()
