from abc import ABC, abstractmethod


class ExplainerBase(ABC):
    """
    Base interface for model explainers.
    Implementations must provide `explain(X)` which should generate
    explanations for the given samples and plot them as appropriate.
    The method may also return the underlying SHAP values or other
    explanation artifacts for further processing by the caller.
    """
    @abstractmethod
    def explain(self, X):
        """Generate and (optionally) plot explanations for X."""
        raise NotImplementedError