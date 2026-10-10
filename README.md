# AgriVision
An Intelligent System for Leaf Disease Diagnosis and Yield Loss Prediction

Upload one or more wheat leaf photos and AgriVision returns:
1. **Disease diagnosis** (BrownRust, YellowRust, Septoria, Mildew or Healthy) with confidence
2. **Severity** of the infection (% of leaf area, a rough colour-based index)
3. **Estimated yield loss** (%) from disease, severity and weather
4. **Recommended action** for the diagnosed disease
5. **Grad-CAM heatmap** showing which parts of the leaf influenced the diagnosis

Weather comes live from the NASA POWER API for a chosen location (default Islamabad), with manual sliders as a fallback.

## Results
- Classifier (MobileNetV2, 5 classes): 66.0% test accuracy
  (fine-tuning gave 64.7%, so the baseline is used)
- Model comparison (test accuracy): MobileNetV2 66.0% > EfficientNetB0 51.3% > MobileNetV3Small 50.7% > CNN from scratch 34.0%
- Yield-loss regression (simulated labels): Random Forest best, MAE 1.114, RMSE 1.531, R2 0.964
  (Ridge 0.876, Lasso 0.869, Linear 0.878)
- Severity vs informal manual estimates (15 leaves): correlation 0.70, but the method overestimates (mean absolute difference 21 points)

## Project structure
- `notebooks/01_eda_preprocessing.ipynb`: data exploration, stratified split (70/15/15), class weights
- `notebooks/02_weather_data.ipynb`: NASA POWER weather download and seasonal summary
- `notebooks/03_severity.ipynb`: HSV colour-based severity estimation and validation
- `notebooks/04_model_training.ipynb`: MobileNetV2 transfer learning, fine-tuning, evaluation, confusion matrix
- `notebooks/05_yield_loss.ipynb`: rule-based yield-loss formula
- `notebooks/06_yield_regression.ipynb`: Linear, Ridge, Lasso and Random Forest comparison
- `notebooks/07_model_comparison.ipynb`: CNN from scratch, MobileNetV3Small and EfficientNetB0 vs MobileNetV2
- `app/app.py`: Streamlit web app
- `app/utils.py`: Grad-CAM

## Dataset
999 wheat leaf photos in 5 classes: Septoria (349), YellowRust (239), Mildew (161), BrownRust (128), Healthy (122). The dataset is imbalanced, so class weights are used during training. The images are not included in this repository.

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
1. Put the dataset in `data/raw/`.
2. Run notebooks 01 to 06 in order. Notebook 04 creates `models/leaf_model.keras` and notebook 06 creates `models/yield_model.joblib`. Trained `.keras` files are not stored in the repository.
3. Start the app:
```
streamlit run app/app.py
```

## Limitations
- Accuracy is about 66%, and YellowRust is often missed.
- Yield-loss labels are simulated, and the disease and weather factors are assumptions, not measured values. Treat yield loss as indicative only.
- Severity is colour-based and overestimates (soil, straw and leaf tips can count as disease). Read it as a relative index.
- The dataset is small and imbalanced, and covers wheat only.
- Not tested on separate field photos, and the app is not deployed online.

## Future work
- More images (especially YellowRust and Healthy) and testing on new field photos
- A segmentation model for severity, validated against expert annotations
- Real yield records to train and validate the regression model
- More careful tuning of other backbones, and deployment (for example on Hugging Face Spaces)