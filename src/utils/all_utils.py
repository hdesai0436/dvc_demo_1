import yaml
import os
import numpy as np
import joblib


def read_yaml(path_to_yaml: str):
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content


def create_dir(dirs:list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True)
        print(f"dir is created at the {dir_path}")

def save_local_df(data,data_path):
    data.to_csv(data_path,index=False)

def save_model( model, model_filename):
    joblib.dump(model,model_filename)

def load_model(model_file_path):
    model = joblib.load(model_file_path)
    return model
