import string
import inspect
import pandas as pd



class _SomeNumber:
    def __eq__(self, other):
        return isinstance(other, (float, int))


SomeNumber = _SomeNumber()


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
    assert inspect.signature(provided) == inspect.signature(
        expected), f"Function signature modified! Expected {_format_parameters(expected)} but got {_format_parameters(provided)}."


def is_legal_function(fn, *args, **kwargs):
    if fn.__name__ == "evaluate_LR":
        def evaluate_LR(X, y, C):
            pass

        _check_signature(evaluate_LR, fn)
        # For some reason the code below still issues the warning, 
        # in which case I think it does more harm than good (the students will be confused).
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=ConvergenceWarning)
#             rval = fn(*args, **kwargs)
            
#         assert rval == {"train": SomeNumber,
#                         "test": SomeNumber}, f"`evaluate_LR` should return a dictionary with train and test scores, e.g., {'train': 0.9, 'test': 0.95}, got: {rval}."

    if fn.__name__ == "get_subset":
        def get_subset(X, y, characters_list):
            pass

        _some_subset = fn(*args, **kwargs)
        assert isinstance(_some_subset,
                          pd.DataFrame), f"`get_subset(X, y, characters=['A','B']) should return a dataframe but returned a {type(_some_subset)}."

    if fn.__name__ == "plot_curve":
        def plot_curve(X, y):
            pass

        _check_signature(plot_curve, fn)

    if fn.__name__ == "plot_sign_coefficients":
        def plot_sign_coefficients(X, y, character):
            pass

        _check_signature(plot_sign_coefficients, fn)

    if fn.__name__ == "plot_confusion_matrix":
        def plot_confusion_matrix(X, y):
            pass

        _check_signature(plot_confusion_matrix, fn)

    if fn.__name__ == "plot_mistakes":
        def plot_mistakes(X, y, character):
            pass

        _check_signature(plot_mistakes, fn)

    if fn.__name__ == "plot_hog_feature":
        def plot_hog_feature(original_image, hog_image, cell_size=2):
            pass

        _check_signature(plot_hog_feature, fn)

    if fn.__name__ == "plot_hog_features":
        def plot_hog_features(X, y, character):
            pass

        _check_signature(plot_hog_features, fn)

    if fn.__name__ == "plot_edges":
        def plot_edges(X, y):
            pass

        _check_signature(plot_edges, fn)

    if fn.__name__ == "final_evaluator":
        def final_evaluator(X_features, y, X_eval_features, y_eval):
            pass

        _check_signature(final_evaluator, fn)

    if fn.__name__ == "compute_hog_feats":
        def compute_hog_feats(X, cell_size):
            pass

        _check_signature(compute_hog_feats, fn)

    if fn.__name__ == "compute_edges":
        def compute_edges(X):
            pass

        _check_signature(compute_edges, fn)
