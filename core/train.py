import config
import utils
import models
import pandas as pd
import numpy as np
from sklearn import model_selection
import joblib


def get_data_and_train_model(
    task_name,
    target_map,
    string_cleaner,
    seed=0,
    verbose=True,
    flatten=True,
):
    """
    Returns the trained model
    """
    if task_name == "A":
        train_data = config.TRAIN_TASK_A
    else:
        train_data = config.TRAIN_TASK_B

    if verbose:
        print("FETCHING")

    # Get data
    df = pd.read_csv(train_data)

    if verbose:
        print(f"FETCHED DATA, NOW CLEANING")

    # Clean data
    X_train, y_train = utils.get_clean_dataset(
        df, target_map, task_name=task_name, string_cleaner=string_cleaner, seed=seed
    )

    if verbose:
        print("CLEANING DONE")

    # Get class weights
    weights = np.bincount(y_train)
    weights = {i: weights.sum() / weights[i] for i in range(len(weights))}

    # Otherwise XGBOOST Throws error
    if len(weights) == 2:
        weights = weights[1] / weights[0]

    print("VECTORIZING")
    # Vectorize
    # X_train = get_glove_vec_data(X_train, enable=enable)

    print("GETTING PIPE...")
    cf_pipe = models.get_cf_pipe(weights, seed=seed)

    # Fit the model
    if verbose:
        print("FITTING...")

    X_train = utils.reshape_training_data(X_train, flatten=flatten)

    cf_pipe.fit(X_train, y_train)

    if verbose:
        print("TRAINING DONE")

    # Save model
    joblib.dump(
        cf_pipe, f"{config.MODEL_SAVE_PATH}TASK_{task_name}_model_final.pkl", compress=1
    )

    return {
        "model": cf_pipe,
        "cv_f1_weighted": model_selection.cross_val_score(
            cf_pipe,
            X_train,
            y_train,
            n_jobs=-1,
            verbose=1,
            scoring="f1_weighted",
        ),
    }


if __name__ == "__main__":

    # Seed all
    utils.seed_all(config.RANDOM_SEED)

    # Train the stuff
    # TASK A

    model_1_dict = get_data_and_train_model(
        "A",
        config.TASK_A_MAP,
        utils.clean_one_text,
        seed=config.RANDOM_SEED,
        verbose=True,
    )


    # print params
    print(model_1_dict['cv_f1_weighted'])

    model_2_dict = get_data_and_train_model(
        "B",
        config.TASK_B_MAP,
        utils.clean_one_text,
        seed=config.RANDOM_SEED,
        verbose=True,
    )

    # print params
    print(model_2_dict['cv_f1_weighted'])
