from autogluon.tabular import (
    TabularDataset,
    TabularPredictor
)

train_loc = 'tmp/bookings.train.csv'
test_loc = 'tmp/bookings.test.csv'

train_data = TabularDataset(train_loc)
test_data = TabularDataset(test_loc)

label = 'is_cancelled'
save_path = 'tmp'
tp = TabularPredictor(label=label, path=save_path)
predictor = tp.fit(train_data)

y_test = test_data[label]
test_data_no_label = test_data.drop(columns=[label])
y_pred = predictor.predict(test_data_no_label)

predictor.evaluate_predictions(
    y_true=y_test,
    y_pred=y_pred,
    auxiliary_metrics=True
)
