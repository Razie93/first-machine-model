# First Machine Model

An educational Python project that predicts a student's final exam score from hours studied, attendance rate, and midterm score using linear regression.

## Run

Requires Python, pandas, NumPy, and scikit-learn. With [uv](https://docs.astral.sh/uv/):

```bash
uv run --no-project --with pandas --with scikit-learn python student_scorePredector.py
uv run --no-project --with pandas --with scikit-learn python -m unittest test_student_scorePredector -v
```

Alternatively, install the dependencies in a virtual environment and run the same Python commands without `uv run --no-project --with ...`.

The example student (6.5 hours studied, 85% attendance, 80 midterm score) receives a predicted final score of about 80.32. Leave-one-out evaluation on the 10 examples gives an average absolute error of about 1.05 points.

## Limitations

The 10 records are illustrative, not real historical student data. The reported error is on these examples only and does not establish real-world predictive accuracy. The model does not account for other factors affecting exam results; do not use its predictions to make decisions about students. Replace the sample data and evaluate on a sufficiently large, representative dataset before considering practical use.
