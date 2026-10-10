# AgriVision
An Intelligent System for Leaf Disease Diagnosis and Yield Loss Prediction

Upload a photo of a wheat leaf and AgriVision returns:
1. **Disease diagnosis** (BrownRust, YellowRust, Septoria, Mildew or Healthy)
2. **Severity** of the infection (% of leaf area affected)
3. **Estimated yield loss** (%) based on disease, severity and seasonal weather

## Project structure
- `notebooks/01_eda_preprocessing.ipynb`: data exploration, stratified split (70/15/15), class weights
- `notebooks/02_weather_data.ipynb`: seasonal weather data (2021-22 to 2025-26)
- `notebooks/03_severity.ipynb`: color-based (HSV) severity estimation
- `notebooks/04_model_training.ipynb`: MobileNetV2 transfer learning classifier
- `notebooks/05_yield_loss.ipynb`: rule-based yield loss estimation
- `app/app.py`: Streamlit web app

## Dataset
999 wheat leaf photos in 5 classes: Septoria (349), YellowRust (239), Mildew (161), BrownRust (128), Healthy (122). The dataset is imbalanced, so class weights are used during training. The images are not included in this repository.

## Results
- Baseline (frozen MobileNetV2): 66% test accuracy
- After fine-tuning: XX% test accuracy

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Put the dataset in `data/raw/`, run the notebooks in order, then start the app:
```
streamlit run app/app.py
```

## Limitations
- Severity is a color-based estimate. Dry yellow straw can be counted as disease.
- Yield-loss factors are assumed values, not learned from real yield data.
- The dataset is small and imbalanced.