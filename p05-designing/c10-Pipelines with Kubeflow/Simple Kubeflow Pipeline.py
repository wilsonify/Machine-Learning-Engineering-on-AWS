# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import kfp
from kfp import dsl
from kfp.components import InputPath, OutputPath
from kfp.components import create_component_from_func


def download_dataset(
    df_all_data_path: OutputPath(str)):
    
    import pandas as pd
    
    url="https://bit.ly/3POP8CI"
    
    df_all_data = pd.read_csv(url)
    print(df_all_data)
    df_all_data.to_csv(df_all_data_path, header=True, index=False)


def process_data(
    df_all_data_path: InputPath(str), 
    df_training_data_path: OutputPath(str), 
    df_test_data_path: OutputPath(str)):
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df_all_data = pd.read_csv(df_all_data_path)
    print(df_all_data)
    
    X = df_all_data['management_experience_months'].values 
    y = df_all_data['monthly_salary'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    df_training_data = pd.DataFrame({ 'monthly_salary': y_train, 'management_experience_months': X_train})
    df_training_data.to_csv(df_training_data_path, header=True, index=False)
    df_test_data = pd.DataFrame({ 'monthly_salary': y_test, 'management_experience_months': X_test})
    df_test_data.to_csv(df_test_data_path, header=True, index=False)


def train_model(
    df_training_data_path: InputPath(str),
    model_path: OutputPath(str)):
    
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from joblib import dump
    
    df_training_data = pd.read_csv(df_training_data_path)
    print(df_training_data)
    
    X_train = df_training_data['management_experience_months'].values
    y_train = df_training_data['monthly_salary'].values
    
    model = LinearRegression().fit(X_train.reshape(-1, 1), y_train)
    print(model)
    dump(model, model_path)


def evaluate_model(
    model_path: InputPath(str),
    df_test_data_path: InputPath(str)):
    
    import pandas as pd
    from joblib import load
    
    df_test_data = pd.read_csv(df_test_data_path)
    
    X_test = df_test_data['management_experience_months'].values
    y_test = df_test_data['monthly_salary'].values
    
    model = load(model_path)
    print(model.score(X_test.reshape(-1, 1), y_test))


def perform_sample_prediction(
    model_path: InputPath(str)):
    from joblib import load
    
    model = load(model_path)
    print(model.predict([[42]])[0])


# +
download_dataset_op = create_component_from_func(
    download_dataset, 
    packages_to_install=['pandas']
)

process_data_op = create_component_from_func(
    process_data, 
    packages_to_install=['pandas', 'sklearn']
)

train_model_op = create_component_from_func(
    train_model, 
    packages_to_install=['pandas', 'sklearn', 'joblib']
)

evaluate_model_op = create_component_from_func(
    evaluate_model, 
    packages_to_install=['pandas', 'joblib', 'sklearn']
)

perform_sample_prediction_op = create_component_from_func(
    perform_sample_prediction, 
    packages_to_install=['joblib', 'sklearn']
)


# -

@dsl.pipeline(
    name='Basic pipeline',
    description='Basic pipeline'
)
def basic_pipeline():
    DOWNLOAD_DATASET = download_dataset_op()
    PROCESS_DATA = process_data_op(DOWNLOAD_DATASET.output)
    TRAIN_MODEL = train_model_op(PROCESS_DATA.outputs['df_training_data'])
    EVALUATE_MODEL = evaluate_model_op(TRAIN_MODEL.outputs['model'], PROCESS_DATA.outputs['df_test_data'])
    PERFORM_SAMPLE_PREDICTION = perform_sample_prediction_op(TRAIN_MODEL.outputs['model'])
    PERFORM_SAMPLE_PREDICTION.after(EVALUATE_MODEL)


kfp.compiler.Compiler().compile(basic_pipeline, 'basic_pipeline.yaml')