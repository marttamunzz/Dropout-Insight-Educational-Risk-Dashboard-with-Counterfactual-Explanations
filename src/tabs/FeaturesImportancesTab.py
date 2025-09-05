from dash import html
import dash_bootstrap_components as dbc

from explainerdashboard.custom import *


class FeaturesImportanceBasicTab(ExplainerComponent):
    """
    A class for creating a 'Predictions' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="Predictions", name=None,
                 pos_label=None, hide_descriptions=True, hide_selector=True,
                 hide_type=True, hide_depth=True, hide_popout=True, **kwargs):
        """
        Initialize a FeaturesImportanceTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - hide_descriptions (bool): Whether to display descriptions of the variables.
            - hide_selector (bool): Whether to display a selector or hide it.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Setting attributes
        self.importances = ImportancesComponent(explainer, name=self.name+"0", title="Feature Impact", subtitle="Average impact on predicted dropout",
                                                hide_selector=hide_selector, hide_type=hide_type, hide_depth=hide_depth, hide_popout=hide_popout, hide_descriptions=hide_descriptions)
        self.confusionmatrix = ConfusionMatrixComponent(explainer, name=self.name+"2",
                                                        hide_selector=hide_selector, hide_subtitle=True, pos_label=pos_label, **kwargs)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap container
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div(
                        "This matrix represents the visualization of the performance of the algorithm used for prediction. Each square has a different meaning: "),
                    html.Div(""),
                    html.Div(
                        "- True positive (top left): It is predicted to be postive and it's true."),
                    html.Div(
                        "- False negative (top right): It is predicted to be negative and it's false."),
                    html.Div(
                        "- False positive (bottom left): It is predicted to be positive and it's false."),
                    html.Div(
                        "- True negative (bottom right): It is predicted to be negative and it's true."),
                    html.Div(""),
                    html.Div(
                        "The top squares show the dropout percentage and the down squares the non-dropout percentage."),
                ], width=4, style=dict(margin=30)),
                # Display ConfusionMatrixComponent
                dbc.Col([
                    self.confusionmatrix.layout(),
                ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),

            dbc.Row([
                dbc.Col([
                    html.H3("Description"),
                    html.Div("The bar graph indicates how much each feature contributes to the failure/model prediction, determining the degree of usefulness of a specific variable for a current model and prediction."),
                    html.Div(
                        "For this purpose, it is calculated the mean absolute SHAP value that shows how much a single feature impacts on the prediction."),
                    html.Div(
                        "You can check the feature label on the left side of the graph and the description on the right side."),
                    html.Div(f"{self.explainer.columns_ranked_by_shap()[0]} was contributing the most"
                             f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                             f" and {self.explainer.columns_ranked_by_shap()[2]}."),
                ], width=4, style=dict(margin=30)),
                # Display ImportancesComponent
                dbc.Col([
                    self.importances.layout(),
                ], width=7, style=dict(margin=30)),
            ], class_name="mt-4"),
        ], fluid=True)


class FeaturesImportanceExpertTab(ExplainerComponent):
    """
    A class for creating a 'Feature Impact' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="Feature Impact", name=None,
                 hide_descriptions=True, hide_selector=True, **kwargs):
        """
        Initialize a FeaturesImportanceTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - hide_descriptions (bool): Whether to display descriptions of the variables.
            - hide_selector (bool): Whether to display a selector or hide it.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Setting attributes
        self.importances = ImportancesComponent(explainer, name=self.name+"0", title="Feature Impact", subtitle="Average impact on predicted dropout",
                                                hide_selector=True, hide_type=True, hide_descriptions=hide_descriptions)
        
        # SHAP Summary (beeswarm)
        self.shap_summary = ShapSummaryComponent(
            explainer, name=self.name+"1",
            hide_selector=True, hide_depth=False,
            hide_summary_type = True,
            summary_type="detailed"
        )

        # SHAP Dependence of the top-1 feature for SHAP
        top1 = explainer.columns_ranked_by_shap()[0]
        self.shap_dependence = ShapDependenceComponent(
            explainer, name=self.name+"2",
            feature=top1, color_col=None,   
            hide_selector=hide_selector
        )

        if not self.explainer.descriptions:
            self.hide_descriptions = True

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap container
        return dbc.Container([
            dbc.Row([
                # Display ImportancesComponent
                dbc.Col([
                    html.H5("Which features have the greatest average impact on the model?", className="mb-2 text-muted"),
                    self.importances.layout(),
                ]),
            ], class_name="mt-4 mb-5"),

            dbc.Row([
                    dbc.Col([
                        html.H5("How is the influence of each feature distributed and in which direction does it values go upper/lower?", className="mb-2 text-muted"),
                        self.shap_summary.layout(),
                    ])
                ], class_name="mt-4 mb-5"),

            dbc.Row([
                dbc.Col([
                    html.H5("How does it specifically affect the prediction depending on its values?", className="mb-2 text-muted"),
                    self.shap_dependence.layout(),
                ])
            ], class_name="mt-4 mb-5"),   

        ], fluid=True)
