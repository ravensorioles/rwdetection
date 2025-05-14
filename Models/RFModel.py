from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class RFModel:

    def __init__(self, config):

        self.tree_n_estimators = config['RF']['num_trees']
        self.tree_max_depth = config['RF']['max_depth']

    def get_pipeline(self):
        rf = RandomForestClassifier(
            n_estimators=self.tree_n_estimators,
            max_depth=self.tree_max_depth,
            random_state=42
        )
        steps = [("rf", rf)]
        return Pipeline(steps=steps)
