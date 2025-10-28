"""
Training and prediction logic for PyCANUPO models.
"""

def cmd_train(args):
    """Trains a binary classifier (SVM or Logistic Regression) on labeled point clouds."""
    # TODO: Load training clouds, compute features, train classifier, and save model.
    raise NotImplementedError


def cmd_predict(args):
    """Applies a trained model to classify new point clouds and write scalar fields."""
    # TODO: Load model and input cloud, compute features, run inference, and save output.
    raise NotImplementedError
