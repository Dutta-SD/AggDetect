# An Efficient BERT Aided Pipeline to Detect Aggression and Misogyny
<!-- Insert Abstract here -->
Social media is bustling with ever growing cases of trolling, aggression and hate. A huge amount of data is generated each day which is insurmountable for manual inspection. 

In this work, we propose an efficient and fast pipeline to detect aggression and misogyny in social media texts. We use data from the Second Workshop on Trolling, Aggression and Cyber Bullying for our task. 

We employ a BERT based pipeline to augment our data. Next we employ Tf-Idf and XGBoost based pipeline for detecting aggression and misogyny. 

Our model achieves 0.73 and 0.85 (both Weighted F1 Score) on the 2 prediction tasks, which ranks very close to the state of the art. 

However, the training time, model size and resource requirements are drastically reduced compared to state of the art models, making our proposed pipeline useful for fast inference. We describe the pipeline, examine the results and conduct error analysis to understand the shortcomings of our model.

## Paper Link

## Model Pipeline
<img src='./reports/assets/nn-1.png' width = 500>

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
<!-- Insert Analysis, Confusion Matrix -->
<img src='./reports/assets/heatmap_task_A.png' width = 300><img src='./reports/assets/heatmap_task_B.png' width = 300>