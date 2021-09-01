# Aggression and Misogyny Detection in YouTube Comments


## File Structure
```
.
├── core
│   ├── config.py
│   ├── custom_vectoriser.py
│   ├── Data_Augmentation_Aggression_Detection.ipynb
│   ├── __init__.py
│   ├── models.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
├── input
│   ├── Final_AUG_Sub-task A_ENGLISH.csv
│   ├── Final_AUG_Sub-task B_ENGLISH.csv
│   ├── trac2_eng_dev.csv
│   └── trac2_eng_train.csv
├── LICENSE.md
├── models
│   ├── TASK_A_model_final.pkl
│   └── TASK_B_model_final.pkl
├── README.md
├── reports
└── requirements.txt
```

## Installing
`pip install -e requirements.txt`

## Training
`python ./core/train.py`

## Validation
`python ./core/validate.py`