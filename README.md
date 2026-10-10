# AgriVision
An Intelligent System for Leaf Disease Diagnosis and Yield Loss Prediction

Upload wheat leaf photos and get the disease, severity, estimated yield loss,
a recommended action and a Grad-CAM explanation.

## Results
- Classifier (MobileNetV2, 5 classes): 66.0% test accuracy
  (fine-tuning gave 64.7%, so the baseline is used)
- Yield-loss regression (simulated labels): Random Forest best, MAE 1.114, R2 0.964

## Run
pip install -r requirements.txt
streamlit run app/app.py

## Notebooks
01 EDA/split, 02 weather, 03 severity, 04 classifier, 05 yield-loss formula, 06 regression

## Limitations
Small imbalanced dataset, colour-based severity, simulated yield-loss labels.