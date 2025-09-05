import dash_bootstrap_components as dbc
from explainerdashboard.custom import *

from tabs.components.components import CounterfactualsComponent


class CounterfactualsTab(ExplainerComponent):
    """
    A class for creating a 'Counterfactuals' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="Counterfactuals scenarios", name="Counterfactuals",
                 dataframe=None, trained_model=None, **kwargs):
        """
        Initialize a CounterfactualsTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        name = name or "Counterfactuals"
        super().__init__(explainer, title=title, name=name)

        # Setting attributes
        self.dataframe = dataframe
        self.trained_model = trained_model
        self.counterfactual = CounterfactualsComponent(explainer=explainer, name=self.name+"1",
                                                       dataframe=self.dataframe,
                                                       trained_model=self.trained_model, **kwargs)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap container
        return dbc.Container([
            dbc.Row([
                # Display CounterfactualsComponent
                dbc.Col([
                    self.counterfactual.layout(),
                ]),
            ], class_name="mt-4"),
        ], fluid=True)
