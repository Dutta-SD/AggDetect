# An Efficient BERT Aided Pipeline to Detect Aggression and Misogyny

Social media is bustling with ever growing cases of trolling, aggression and hate. A huge amount of data is generated each day which is insurmountable for manual inspection. 

In this work, we propose an efficient and fast pipeline to detect aggression and misogyny in social media texts. We use data from the Second Workshop on Trolling, Aggression and Cyber Bullying for our task. 

We employ a BERT based pipeline to augment our data. Next we employ Tf-Idf and XGBoost based pipeline for detecting aggression and misogyny. 

Our model achieves 0.73 and 0.85 (both Weighted F1 Score) on the 2 prediction tasks, which ranks very close to the state of the art. 

However, the training time, model size and resource requirements are drastically reduced compared to state of the art models, making our proposed pipeline useful for fast inference. We describe the pipeline, examine the results and conduct error analysis to understand the shortcomings of our model.

## Paper Link

[ACL Anthology Link](https://aclanthology.org/2021.icon-main.60.pdf)

## Model Pipeline

![Model_pipeline](https://ik.imagekit.io/oj8f972s8/NN_FINAL.png)

## Training and Inference

* Create a virtual environment. [See here](https://docs.python.org/3/library/venv.html)
* Clone the repository [See here](https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone)
* Navigate to the cloned repository
* Install requirements as `pip install -r requirements.txt`
* Navigate to `/core` directory and set it as your _current working directory_
* run `bash ./run.sh` for train, validation and inference

## Results (Weighted F1 Score)

|Team Name(Cited in paper)|Score Sub Task A|Score Sub Task B|
|--|--|--|
|Julian|0.802|0.851|
|abaruah|0.728|0.870|
|sdhanshu|0.759|0.857|
|**Our Model**|**0.735**|**0.852**|

## Analysis

<img src='https://ik.imagekit.io/oj8f972s8/heatmap_task_A.png' width = 300> 

**Task A (Aggression Detection) Confusion Matrix** 

Classes are (left to right and top to bottom) 
- **OAG (Overtly Aggressive)**: Explicitly Aggressive Terms
- **CAG (Covertly Aggressive)**: Covertly Aggressive Terms like sarcasm
- **NAG (Non Aggressive)**: Non Aggressive texts

<img src='https://ik.imagekit.io/oj8f972s8/heatmap_task_B.png' width = 300>

**Task B (Misogyny Detection) Confusion Matrix** 

Classes are (left to right and top to bottom) 
- **NGEN**: Neutral Texts
- **GEN**: Contains misogynistic connotations

## Repository Details

- `assets` - Images for report
- `core` - Code related to training and testing after augmentation
- `input` - Data Input. Contains train, test and gold data
- `models` - Serialized Model Files
- `notebooks` - Notebooks done in Google Colab. `notebooks/Data_Augmentation_Aggression_Detection.ipynb` contains detailed code regarding the augmentation process
- `reports` - .tex files
- `test_results` - Test CSV file

## Try it Out!
``` Due to python version issues in huggingface spaces, the link is broken. Will be updated. Redirects to top of README.md now ```

[Click Here](#an-efficient-bert-aided-pipeline-to-detect-aggression-and-misogyny)