# Model definitions
import xgboost
from sklearn import (
    pipeline,
    feature_extraction,
    preprocessing,
    svm,
    ensemble,
    neural_network
)


def get_cf_pipe(c_weights, seed=0):
    # Models
    cf_model = xgboost.XGBClassifier(
        # max_depth=4,
        gamma=0.1,
        # scale_pos_weight=c_weights,
        random_state=seed,
        n_jobs=-1,
    )

    cf_model_2 = svm.LinearSVC(
        random_state=seed,
    )

    cf_model_3 = ensemble.RandomForestClassifier(
        random_state=seed,
    )

    cf_model_4 = neural_network.MLPClassifier(
        random_state=seed,
        max_iter = 50,
        verbose=True,
    )

    cf_pipe = pipeline.Pipeline(
        [
            ("feature-extractor", feature_extraction.text.TfidfVectorizer()),
            ("classifier", cf_model),
        ],
        verbose=True,
    )
    return cf_pipe
