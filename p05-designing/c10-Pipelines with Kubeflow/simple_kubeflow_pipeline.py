import pandas as pd
from joblib import dump
from joblib import load
from kfp import dsl
from kfp.deprecated.compiler import Compiler
from kfp.deprecated.components import InputPath, OutputPath
from kfp.deprecated.components import create_component_from_func
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def download_dataset(url: InputPath(str), df_all_data_path: OutputPath(str)):
    df_all_data = pd.read_csv(url)
    df_all_data.to_csv(df_all_data_path, header=True, index=False)


def process_data(
        df_all_data_path: InputPath(str),
        df_training_data_path: OutputPath(str),
        df_test_data_path: OutputPath(str)
):
    df_all_data = pd.read_csv(df_all_data_path)
    X = df_all_data['management_experience_months'].values
    y = df_all_data['monthly_salary'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    df_training_data = pd.DataFrame({'monthly_salary': y_train, 'management_experience_months': x_train})
    df_training_data.to_csv(df_training_data_path, header=True, index=False)
    df_test_data = pd.DataFrame({'monthly_salary': y_test, 'management_experience_months': x_test})
    df_test_data.to_csv(df_test_data_path, header=True, index=False)


def train_model(df_training_data_path: InputPath(str), model_path: OutputPath(str)):
    df_training_data = pd.read_csv(df_training_data_path)
    print(f"df_training_data.shape = {df_training_data.shape}")
    x_train = df_training_data['management_experience_months'].values
    y_train = df_training_data['monthly_salary'].values
    model = LinearRegression()
    model.fit(x_train.reshape(-1, 1), y_train)
    dump(model, model_path)
    print(f"model.__dict__ = {model.__dict__}")
    print(f"model_path = {model_path}")


def evaluate_model(model_path: InputPath(str), df_test_data_path: InputPath(str)):
    df_test_data = pd.read_csv(df_test_data_path)
    x_test = df_test_data['management_experience_months'].values
    y_test = df_test_data['monthly_salary'].values
    model = load(model_path)
    print(model.score(x_test.reshape(-1, 1), y_test))


def perform_sample_prediction(model_path: InputPath(str)):
    model = load(model_path)
    print(model.predict([[42]])[0])


download_dataset_op = create_component_from_func(
    func=download_dataset,
    packages_to_install=['pandas']
)

process_data_op = create_component_from_func(
    func=process_data,
    packages_to_install=['pandas', 'scikit-learn']
)

train_model_op = create_component_from_func(
    func=train_model,
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)

evaluate_model_op = create_component_from_func(
    func=evaluate_model,
    packages_to_install=['pandas', 'joblib', 'scikit-learn']
)

perform_sample_prediction_op = create_component_from_func(
    func=perform_sample_prediction,
    packages_to_install=['joblib', 'scikit-learn']
)


@dsl.pipeline(
    name='Basic pipeline',
    description='Basic pipeline'
)
def basic_pipeline():
    url = "https://github.com/wilsonify/Machine-Learning-Engineering-on-AWS/raw/main/p05-designing/c10-Pipelines%20with%20Kubeflow/management_experience_and_salary.csv"
    download_dataset_step = download_dataset_op(url)
    process_data_step = process_data_op(download_dataset_step.output)
    train_model_step = train_model_op(process_data_step.outputs['df_training_data'])
    evaluate_model_step = evaluate_model_op(
        train_model_step.outputs['model_path'],
        process_data_step.outputs['df_test_data']
    )
    perform_sample_prediction_step = perform_sample_prediction_op(train_model_step.outputs['model'])
    perform_sample_prediction_step.after(evaluate_model_step)


Compiler().compile(basic_pipeline, 'basic_pipeline.yaml')
