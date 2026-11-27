# AI Handwritten Digit Guesser (KNN vs SVM)

## Files
- `app.py`: Streamlit app for image upload, preprocessing and predictions.
- `train_models.py`: trains both models and produces evaluation metrics.
- `knn_model.joblib`, `svm_model.joblib`: trained model pipelines.
- `metrics.json`: held-out evaluation metrics.
- `requirements.txt`: dependencies.

## Retrain
```bash
python train_models.py
```
Run the app locally:
```bash
streamlit run app.py
```

The dataset is `sklearn.datasets.load_digits` (1797 grayscale 8x8 images). The test split is stratified, 80/20, with random state 42. Test-set metrics are distinct from real-photo performance.
