from dash import html
import dash_bootstrap_components as dbc

from explainerdashboard.custom import *
from tabs.components.components import SelectStudentComponent
from tabs.components.components import TimerComponent


class WhatIfBasicTab(ExplainerComponent):
    """
    A class for creating a 'What If' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="What if...", name=None,
                 hide_selector=True, index_check=True,
                 n_input_cols=4, **kwargs):
        """
        Initialize a WhatIfTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - hide_selector (bool): Whether to display a selector or hide it.
            - n_input_cols (int): Number of columns to split features inputs in.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Setting attributes
        self.input = FeatureInputComponent(explainer, name=self.name+"0",
                                           hide_selector=hide_selector, n_input_cols=n_input_cols,
                                           **update_params(kwargs, hide_index=False))
        self.index = SelectStudentComponent(explainer, name=self.name+"1",
                                            hide_selector=hide_selector, index_dropdown=False, **kwargs)
        self.prediction = ClassifierPredictionSummaryComponent(explainer, name=self.name+"2",
                                                               feature_input_component=self.input,
                                                               hide_star_explanation=True,
                                                               hide_selector=hide_selector, **kwargs)
        self.contribution = ShapContributionsGraphComponent(explainer, name=self.name+"3",
                                                            hide_selector=hide_selector, **kwargs)
        self.timer = TimerComponent(explainer, name=self.name+"4", **kwargs)
        self.index_connector = IndexConnector(self.index, [self.input, self.contribution],
                                              explainer=explainer if index_check else None)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap container
        return dbc.Container([

            # Timer 
            dbc.Row([
                dbc.Col([ self.timer.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

            # Selector
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("Select the student you want to check via the dropdown box or select it randomly."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([ self.index.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

            # Contributions 
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("This plot shows the contribution that each individual feature has had on the prediction for a specific observation."),
                    html.Div("The contributions (starting from the population average) add up to the final prediction."),
                    html.Div("This allows you to explain exactly how each individual prediction has been built up from all the individual ingredients in the model."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([ self.contribution.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

            # Prediction Summary 
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("The graph shows the probability of dropout and no dropout of the selected student above."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([ self.prediction.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

            # Feature Input
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("With the Feature Input module, you can change the different values of the variables of a student to see how the probability of dropout vary. You can check the value range under each variable."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([ self.input.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

        ], fluid=True)


class WhatIfExpertTab(ExplainerComponent):
    """
    A class for creating a 'What If' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="What if...", name=None,
                 hide_selector=True, index_check=True,
                 n_input_cols=4, **kwargs):
        """
        Initialize a WhatIfTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - hide_selector (bool): Whether to display a selector or hide it.
            - n_input_cols (int): Number of columns to split features inputs in.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Setting attributes
        self.input = FeatureInputComponent(explainer, name=self.name+"0",
                                           hide_selector=hide_selector, n_input_cols=n_input_cols,
                                           **update_params(kwargs, hide_index=False))
        self.index = SelectStudentComponent(explainer, name=self.name+"1",
                                            hide_selector=hide_selector, **kwargs)
        self.prediction = ClassifierPredictionSummaryComponent(explainer, name=self.name+"2",
                                                               feature_input_component=self.input,
                                                               hide_star_explanation=True,
                                                               hide_selector=hide_selector, **kwargs)
        self.contribution = ShapContributionsGraphComponent(explainer, name=self.name+"3",
                                                            hide_selector=hide_selector, **kwargs)
        self.timer = TimerComponent(explainer, name=self.name+"4", **kwargs)
        self.index_connector = IndexConnector(self.index, [self.input, self.contribution],
                                              explainer=explainer if index_check else None)

    def layout(self):
        return dbc.Container([

            # Timer
            dbc.Row([
                dbc.Col(self.timer.layout())
            ], class_name="mt-4 gx-4"),

            # Selector arriba
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("Select the student you want to check via the dropdown box or select it randomly."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([ self.index.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

            # Contributions
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("This plot shows the contribution that each individual feature has had on the prediction for a specific observation."),
                    html.Div("The contributions (starting from the population average) add up to the final prediction."),
                    html.Div("This allows you to explain exactly how each individual prediction has been built up from all the individual ingredients in the model."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([ self.contribution.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

            # Prediction Summary en el hueco donde antes estaba el selector
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("The graph shows the probability of dropout and no dropout of the selected student above."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([ self.prediction.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

            # Feature Input
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("With the Feature Input module, you can change the different values of the features of a student to see how the probability of dropout vary. You can check the value range under each variable."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([ self.input.layout() ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

        ], fluid=True)