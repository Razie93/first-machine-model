"""Small educational exam-score prediction example."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

FEATURES = ['hours_studied', 'attendance_rate', 'midterm_score']
TARGET = 'final_exam_score'

def load_data():
    return pd.DataFrame({'hours_studied': [2, 5, 8, 3, 10, 7, 1, 9, 4, 6],
                         'attendance_rate': [60, 80, 95, 70, 98, 88, 50, 92, 75, 85],
                         'midterm_score': [55, 72, 88, 65, 94, 81, 48, 90, 68, 78],
                         'final_exam_score': [58, 75, 90, 62, 96, 84, 50, 91, 70, 80]})

def train_model(data):
    model = LinearRegression()
    model.fit(data[FEATURES], data[TARGET])
    return model

def evaluate_model(data):
    if len(data) < 3:
        raise ValueError('At least three students are required.')
    predictions = cross_val_predict(LinearRegression(), data[FEATURES], data[TARGET], cv=LeaveOneOut())
    return {'mae': mean_absolute_error(data[TARGET], predictions), 'predictions': predictions}

def predict_score(model, student):
    row = pd.DataFrame([student], columns=FEATURES)
    if row.isna().any().any() or not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError('All inputs must be finite numbers.')
    if not (0 <= row.at[0, 'hours_studied'] <= 24 and
            0 <= row.at[0, 'attendance_rate'] <= 100 and
            0 <= row.at[0, 'midterm_score'] <= 100):
        raise ValueError('Hours must be 0-24; attendance and midterm must be 0-100.')
    return float(np.clip(model.predict(row)[0], 0, 100))

def main():
    data = load_data()
    result = evaluate_model(data)
    student = {'hours_studied': 6.5, 'attendance_rate': 85, 'midterm_score': 80}
    print(f"Leave-one-out average absolute error: {result['mae']:.2f} points")
    print(f"Predicted final exam score: {predict_score(train_model(data), student):.2f} points")
    print('Caution: 10 illustrative records are not a validated real-world dataset.')

if __name__ == '__main__':
    main()
