# An Efficient BERT Aided Pipeline to Detect Aggression and Misogyny
<!-- Insert Abstract here -->

## Paper Link

## File Structure
```
.
|
├── core
│   ├── config.py
│   ├── __init__.py
│   ├── models.py
│   ├── run.sh
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
├── input
│   ├── eng_gold_A.csv
│   ├── eng_gold_B.csv
│   ├── Final_AUG_Sub-task A_ENGLISH.csv
│   ├── Final_AUG_Sub-task B_ENGLISH.csv
│   ├── trac2_eng_dev.csv
│   ├── trac2_eng_test.csv
│   └── trac2_eng_train.csv
├── LICENSE.md
├── models
│   ├── TASK_A_model.pkl
│   └── TASK_B_model.pkl
├── README.md
└── requirements.txt

```
## Training and Inference
* Create a virtual environment. [See here](https://docs.python.org/3/library/venv.html)
* Install requirements as `pip install -r requirements.txt`
* Navigate to `/core` directory and set it as your current working directory
* run `bash ./run.sh` for train, validation and inference

## Results (Weighted F1 Score)
|Team Name(Cited in paper)|Score Sub Task A|Score Sub Task B|
|--|--|--|
|hello|jj|kk|



