# Validation
import config
import utils
import joblib
from sklearn import metrics
import xgboost
import pandas as pd

def validate(
    clf_pipe,
    task_name,
    target_map,
    string_cleaner,
    seed=0,
    verbose=True,
    flatten=True,
):
    # Returns Score
    if verbose:
        print("Validating on Non trained data...")

    val_data = config.VAL_TASK
    df = pd.read_csv(val_data)

    X_val, y_val = utils.get_clean_dataset(
        df,
        target_map,
        train=False,
        task_name=task_name,
        string_cleaner=string_cleaner,
        seed=seed,
    )
    print("VECTORIZING")
    # Vectorize

    X_val = utils.reshape_training_data(X_val, flatten=flatten)

    y_preds = clf_pipe.predict(X_val)
    y_true = y_val

    # Return classification report
    return {
        "f1_weighted": round(metrics.f1_score(y_true, y_preds, average="weighted"), 3)
    }

# Script Run
if __name__ == '__main__':
    # Load model for task 1
    utils.seed_all(config.RANDOM_SEED)

    # Task 1
    task_1_pipe = joblib.load(config.MODEL_SAVE_PATH + 'TASK_A_model_final.pkl')
    print(
        validate(
            task_1_pipe,
            'A',
            config.TASK_A_MAP,
            utils.clean_one_text,
            seed = config.RANDOM_SEED,
            verbose=True,
        )
    )

    # Task 2
    task_1_pipe = joblib.load(config.MODEL_SAVE_PATH + 'TASK_B_model_final.pkl')
    print(
        validate(
            task_1_pipe,
            'B',
            config.TASK_B_MAP,
            utils.clean_one_text,
            seed = config.RANDOM_SEED,
            verbose=True,
        )
    )