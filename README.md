# AI Handwritten Digit Guesser — KNN vs SVM

Streamlit coursework app comparing K-Nearest Neighbors and Support Vector Machine on scikit-learn's Digits dataset.

## Files
- `app.py`: upload interface, image preprocessing, predictions, and the single shared 8x8 pixel preview.
- `knn_model.joblib`, `svm_model.joblib`: trained model pipelines.
- `metrics.json`: held-out test metrics.
- `train_models.py`: reproducible training/evaluation.
- `requirements.txt`: dependencies.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Both classifiers receive the exact same preprocessed 64-value array. Preprocessing uses adaptive thresholding, component isolation, stroke thickening, resizing, and scaling to 0–16. The gray pencil 8 photo is a known validation example; success on this one image does not establish general photo accuracy.

Test-set metrics are not guarantees for uploaded photos, and model confidence is not measured accuracy.
