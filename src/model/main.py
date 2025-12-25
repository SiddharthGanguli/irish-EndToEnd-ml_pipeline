from training import Trainer

SPLIT_DATA_PATH = "/Users/siddharthaganguli/Desktop/irish/data/split_data"
MODEL_OUTPUT_PATH = "/Users/siddharthaganguli/Desktop/irish/irish_models"

def main_training():
    trainer = Trainer(data_path=SPLIT_DATA_PATH, output_path=MODEL_OUTPUT_PATH)
    trainer.train_with_mlflow()

if __name__ == "__main__":
    main_training()
