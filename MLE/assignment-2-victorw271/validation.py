from sklearn.preprocessing import StandardScaler, OneHotEncoder
import string
import inspect
import pandas as pd


def is_legal_answer(q: str, n_options: int = 4):
    assert isinstance(q, str), f"The answer should be a string, was {type(q)}."
    q = q.replace(" ", "").replace(",", "")
    for c in q:
        assert c.lower() in string.ascii_lowercase[:n_options], f"Invalid character in answer: '{c}'"
    print("Answer formatted correctly.")


def _format_parameters(fn):
    def _format_parameter(param):
        return param.name + (f"={param.default}" if not param.default == inspect._empty else "")

    return "(" + ", ".join(_format_parameter(p) for p in inspect.signature(fn).parameters.values()) + ")"


def _check_signature(expected, provided):
    assert str(inspect.signature(provided)) == str(inspect.signature(
        expected)), f"Function signature modified! Expected {_format_parameters(expected)} but got {_format_parameters(provided)}."


def signature_unchanged(fn, *args, **kwargs):

    if fn.__name__ == "compute_edges":
        def compute_edges(X):
            pass

        _check_signature(compute_edges, fn)

    if fn.__name__ == "pipeline_builder":
        def pipeline_builder(X, model, scaler=StandardScaler(), encoder=OneHotEncoder()):
            """ Returns a pipeline that imputes missing values and scales numeric features
            Keyword arguments:
              X -- The input data. Only used to identify features types (eg. numeric/categorical), not for training the pipeline.
              model -- any scikit-learn model (e.g. a classifier or regressor)
              scaler -- any scikit-learn feature scaling method (Optional)
              encoder -- any scikit-learn category encoding method (Optional)
              Returns: a scikit-learn pipeline which preprocesses the data and then runs the classifier
            """
            pass
        
        _check_signature(pipeline_builder, fn)
    
    if fn.__name__ == "feature_count":
        def feature_count(X, y, pipeline):
            """ Counts the number of features created in the preprocessing steps of the 
            given pipeline.
                X -- The input data.
                y -- The labels
                pipeline -- The pipeline that will transform the data
            Returns: The feature count (an integer)
            """
            pass
        
        _check_signature(feature_count, fn)
    
    if fn.__name__ == "evaluate_pipe":
        def evaluate_pipe(X, y, pipeline, scoring, subsample_size=None):
            """ Evaluates the given pipeline using cross-validation on the given data
            Keyword arguments:
              X -- The input data.
              y -- The labels
              pipeline -- any scikit-learn pipeline (including a classifier or regressor)
              scoring -- the scoring function for the evaluation
              subsample_size -- Fraction of the input data to use for the cross-validation. 
                                If None, no subsample is made. If float in (0.0, 1.0), 
                                it takes that proportion of the dataset .
            Returns: a dictionary containing `score` and `fit_time` key-value pairs
            """
            pass
        
        _check_signature(evaluate_pipe, fn)
    
    if fn.__name__ == "evaluate_tree_regression":
        def evaluate_tree_regression():
            """ Evaluates a decision trees on the regression task (price prediction).
            You can use the predefined train-test splits for this task.
            """
            pass
        
        _check_signature(evaluate_tree_regression, fn)
    
    if fn.__name__ == "evaluate_tree_classification":
        def evaluate_tree_classification():
            """ Evaluates a decision trees on the classification task (price category prediction)
                You can use the predefined train-test splits for this task.
            """
            pass
        
        _check_signature(evaluate_tree_classification, fn)
    
    if fn.__name__ == "evaluate_boosting_regression":
        def evaluate_boosting_regression():
            """ Evaluates a gradient boosting model with cross-validation on the regression task (price prediction)
            """
            pass
        
        _check_signature(evaluate_boosting_regression, fn)
    
    if fn.__name__ == "evaluate_boosting_classification":
        def evaluate_boosting_classification():
            """ Evaluates a gradient boosting model with cross-validation on the classifcation task (price category prediction)
            """
            pass
        
        _check_signature(evaluate_boosting_classification, fn)
    
    if fn.__name__ == "evaluate_boosting_classification_test":
        def evaluate_boosting_classification_test():
            """ Evaluates a gradient boosting model on the held-out test data for theclassifcation task (price category prediction)
            """
            pass
        
        _check_signature(evaluate_boosting_classification_test, fn)
    
    if fn.__name__ == "compute_tsne":
        def compute_tsne(X):
          """ Applies tSNE to build a 2D representation of the data
          Returns a dataframe with the 2D representation
          X -- The input data
          """
          pass
        
        _check_signature(compute_tsne, fn)
    
    if fn.__name__ == "plot_tsne":
        def plot_tsne(tsne_embeds, price_class):
          """ Plots the given 2D data points, color-coded by rent price classes
          tsne_embeds -- The tSNE embeddings of all neighborhoods
          scores -- The corresponding rent prices
          """
          pass
        
        _check_signature(plot_tsne, fn)
    
    if fn.__name__ == "plot_feature_importance":
        def plot_feature_importance(X, y):
            """ See detailed description above.
            """
            pass
        
        _check_signature(plot_feature_importance, fn)
    
    if fn.__name__ == "plot_encoders":
        def plot_encoders(X, y):
            """ Evaluates a range of models with different categorical encoders and 
            plots the results in a heat map.
            """
            pass
        
        _check_signature(plot_encoders, fn)
    
    if fn.__name__ == "create_doc2vec_embeddings":
        def create_doc2vec_embeddings(model, tokenized_sentences):
          """ Uses the given Doc2Vec model to infer embeddings for the given (tokenized) sentences """
          pass
        
        _check_signature(create_doc2vec_embeddings, fn)
    
    if fn.__name__ == "compute_doc2vec_tsne":
        def compute_doc2vec_tsne(original_array):
          """ Applies tSNE to build a 2D representation of the data
          Returns a dataframe with the 2D representation
          original_array -- The input data
          """
          pass
        
        _check_signature(compute_doc2vec_tsne, fn)
    
    if fn.__name__ == "plot_embeddings":
        def plot_embeddings():
          """ Uses the functions you created above to create the 2D scatter plot.
          """
          pass
        
        _check_signature(plot_embeddings, fn)
    
    if fn.__name__ == "build_final_model":
        def build_final_model(X,y):
          """ Build the best possible model (highest AUC score) for the given dataset.
          """
          pass
        
        _check_signature(build_final_model, fn)
    
    if fn.__name__ == "evaluate_final_model":
        def evaluate_final_model(X_train,y_train,X_test,y_test):
          """ Build and evaluate the model and return the AUC score
          """
          pass
        
        _check_signature(evaluate_final_model, fn)
    