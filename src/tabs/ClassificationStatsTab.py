from dash import html
import dash_bootstrap_components as dbc

from explainerdashboard.custom import *


class ClassificationStatsTab(ExplainerComponent):
    """
    A class for creating a 'Classification Stats' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="Classification Stats", name=None,
                 hide_selector=True, pos_label=None, **kwargs):
        """
        Initialize a ClassificationStatsTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - hide_selector (bool): Whether to display a selector or hide it.
            - pos_label (int, str): Initial positive label.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Setting attributes
        self.summary = ClassifierModelSummaryComponent(explainer, name=self.name+"0",
                                                       hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.rocauc = RocAucComponent(explainer, name=self.name+"1",
                                      hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.confusionmatrix = ConfusionMatrixComponent(explainer, name=self.name+"2",
                                                        hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.classification = ClassificationComponent(explainer, name=self.name+"3",
                                                      hide_selector=hide_selector, pos_label=pos_label, **kwargs)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        return dbc.Container([
            dbc.Row([
                # Display ClassifierModelSummaryComponent
                dbc.Col(
                    self.summary.layout()),

                # Display ConfusionMatrixComponent
                dbc.Col(
                    self.confusionmatrix.layout()),

            ], class_name="mt-4 gx-4"),
            dbc.Row([
                # Display RocAucComponent
                dbc.Col(
                    self.rocauc.layout()),

                # Display ClassificationComponent
                dbc.Col(
                    self.classification.layout()),

            ], class_name="mt-4 gx-4")
        ], fluid=True)
