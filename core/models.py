# Model definitions
import xgboost
from sklearn import pipeline, feature_extraction, preprocessing


def get_cf_pipe(c_weights, seed=0):
    cf_model = xgboost.XGBClassifier(
        max_depth=4, gamma=0.1, scale_pos_weight=c_weights, random_state=seed, n_jobs=-1
    )
    cf_pipe = pipeline.Pipeline(
        [
            ("feature-extractor", feature_extraction.text.TfidfVectorizer()),
            ("norm", preprocessing.Normalizer()),
            ('pow-trans', preprocessing.QuantileTransformer(output_distribution='normal')),
            ("classifier", cf_model),
        ],
        verbose=True,
    )
    return cf_pipe
