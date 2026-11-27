# AI Handwritten Digit Guesser — KNN vs SVM

A Streamlit coursework app comparing K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) on scikit-learn's Digits dataset.

## Files
- `app.py`: upload interface, photo preprocessing, predictions, and pixel preview.
- `knn_model.joblib`, `svm_model.joblib`: trained estimators.
- `metrics.json`: held-out test metrics and confusion matrices.
- `train_models.py`: reproducible training/evaluation script.
- `requirements.txt`: app dependencies.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Test-set metrics are from the held-out dataset, not guarantees for uploaded photos.
- The photo pipeline estimates local paper background, isolates a connected stroke component, normalizes contrast, and maps it to an 8x8 input.
- The app displays the detected crop and processed model inputs to make failures easier to inspect.
- Model probability scores are estimates and are not measured accuracy.
