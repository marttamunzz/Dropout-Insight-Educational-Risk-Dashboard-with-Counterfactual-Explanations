import dash_bootstrap_components as dbc

from tabs.components.components import AutoMLReportComponent
from explainerdashboard.custom import *


class AutoMLReportTab(ExplainerComponent):
    """
    A class for creating a 'AutoML Report' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="AutoML Report", name=None,
                 ML_report=None, **kwargs):
        """
        Initialize a AutoMLReportTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - ML_report (any): AutoML Report instance of the model.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Setting attributes
        self.ML_report = ML_report
        self.report = AutoMLReportComponent(
            explainer, name=self.name+"0", ML_report=self.ML_report)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap container
        return dbc.Container([
            dbc.Row([
                # Display AutoMLReportComponent
                dbc.Col([
                    self.report.layout(),
                ]),
            ], class_name="mt-4"),
        ], fluid=True)
