# Gender Detection From Names

**Live Demo**: [https://task-1-classification-project-gender-505q.onrender.com/](https://task-1-classification-project-gender-505q.onrender.com/)

This project uses a simple machine-learning classification model to predict gender from Andhra Pradesh names.


## Files

- `generate_dataset.py` creates `data/names_gender.csv` with Andhra Pradesh `name,gender` rows.
- `train_model.py` trains a character n-gram classifier and saves it to `models/name_gender_model.joblib`.
- `predict_gender.py` predicts gender for a single name.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Generate the dataset:

```bash
python generate_dataset.py
```

Train the model:

```bash
python train_model.py
```

Predict gender:

```bash
python predict_gender.py Manikanta
```
## output images
<img width="1920" height="1459" alt="image" src="https://github.com/user-attachments/assets/9cb1e571-659b-46ea-b10a-4698db7da47a" />

<img width="1920" height="2235" alt="image" src="https://github.com/user-attachments/assets/5ecea2eb-41ce-4d45-8141-380952d04817" />
<img width="1920" height="1112" alt="image" src="https://github.com/user-attachments/assets/a0ec3e77-b8ea-4d77-b44a-7379a718fbd4" />




## Notes

- The classifier is a baseline model trained only on names.
- Predictions are approximate and should not be treated as ground truth.
