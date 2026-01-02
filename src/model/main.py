from training import Trainer
from decision_tree_train import DecisionTreeModel

SPLIT_DATA_PATH = "/Users/siddharthaganguli/Desktop/irish/data/split_data"
MODEL_OUTPUT_PATH = "/Users/siddharthaganguli/Desktop/irish/irish_models"

def main_training():
    trainer = Trainer(data_path=SPLIT_DATA_PATH, output_path=MODEL_OUTPUT_PATH)
    trainer.train_with_mlflow()

def decision_tree():
    decision_tree_model=DecisionTreeModel(data_path=SPLIT_DATA_PATH,output_path=MODEL_OUTPUT_PATH)
    decision_tree_model.train_with_mlflow()

if __name__ == "__main__":
    main_training()
    decision_tree()
