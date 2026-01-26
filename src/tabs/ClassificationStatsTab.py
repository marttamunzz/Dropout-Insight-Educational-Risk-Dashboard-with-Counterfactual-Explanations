from dash import html
import dash_bootstrap_components as dbc
from explainerdashboard.custom import *
import base64


def _classification_benchmarks_img():
    # Colores filas
    RED = "#FF0000"
    YELLOW = "#D4BF01"
    GREEN = "#02B842"

    # Estilo tarjeta
    CARD_BG = "#EEF2FF"
    CARD_BORDER = "#C7D2FE"
    GRID = "#E5E7EB"
    HEADER_BG = "#4F6EF7"
    METRIC_BG = "#5B76F8"

    blocks = [
        ("Precision", [
            ("≤ 0.50", "Weak: Many “false alarms” (many flagged students are not actually at risk).", RED),
            ("0.50 – 0.80", "Acceptable: Most alerts are correct.", YELLOW),
            ("> 0.80", "Good: Very few false positives.", GREEN),
        ]),
        ("Recall", [
            ("≤ 0.50", "Weak: Misses more than half of truly at-risk students.", RED),
            ("0.50 – 0.80", "Acceptable: Catches most at-risk students.", YELLOW),
            ("> 0.80", "Good: Rarely misses an at-risk student.", GREEN),
        ]),
        ("F1-Score", [
            ("≤ 0.50", "Weak: Poor balance between catching risk and avoiding false alarms.", RED),
            ("0.50 – 0.80", "Acceptable: Reasonable balance for most uses.", YELLOW),
            ("> 0.80", "Good: Strong balance (few misses and few false alarms).", GREEN),
        ]),
        ("ROC-AUC Score", [
            ("< 0.70", "Weak: Limited ability to separate at-risk vs. not-at-risk students.", RED),
            ("0.70 – 0.80", "Acceptable: Good discrimination in general.", YELLOW),
            ("> 0.80", "Good: Strong discrimination (risk ranking is reliable).", GREEN),
        ]),
        ("Log Loss", [
            ("> 0.50", "Weak: Probabilities are not very reliable (often overconfident mistakes).", RED),
            ("0.30 – 0.50", "Acceptable: Probabilities are reasonably reliable.", YELLOW),
            ("< 0.30", "Good: Probabilities are trustworthy (few confident mistakes).", GREEN),
        ]),
    ]

    # Layout
    W = 980
    margin = 20
    title_h = 70
    header_h = 38
    row_h = 36

    total_rows = sum(len(r) for _, r in blocks)
    H = margin * 2 + title_h + header_h + total_rows * row_h

    col_metric = 220
    col_range = 160
    col_label = W - margin * 2 - col_metric - col_range

    x0 = margin
    y = margin

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')

    svg_parts.append(f"""
    <defs>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#000000" flood-opacity="0.18"/>
    </filter>
    <clipPath id="tableClip">
        <rect x="{margin}" y="{margin}" width="{W - 2*margin}" height="{H - 2*margin}" rx="18" ry="18"/>
    </clipPath>
    </defs>
    """)

    svg_parts.append("""
    <style>
    .title { font: 700 20px Arial, Helvetica, sans-serif; fill: #ffffff; }
    .sub   { font: 13px Arial, Helvetica, sans-serif; fill: #ffffff; }
    .hdr   { font: 700 14px Arial, Helvetica, sans-serif; fill: #ffffff; }
    .txt   { font: 14px Arial, Helvetica, sans-serif; fill: #ffffff; }
    .metric{ font: 700 14px Arial, Helvetica, sans-serif; fill: #ffffff; }
    </style>
    """)

    svg_parts.append('<g clip-path="url(#tableClip)">')

    svg_parts.append(
        f'<rect x="{margin}" y="{margin}" width="{W - 2 * margin}" height="{H - 2 * margin}" '
        f'rx="18" fill="{CARD_BG}" stroke="{CARD_BORDER}" stroke-width="1.5" filter="url(#softShadow)"/>'
    )

    svg_parts.append(
        f'<rect x="{x0}" y="{margin}" width="{W - 2 * margin}" height="{title_h}" fill="{HEADER_BG}"/>'
    )

    svg_parts.append(f'<text x="{x0 + 16}" y="{margin + 32}" class="title">Classification Performance Benchmarks</text>')
    svg_parts.append(f'<text x="{x0 + 16}" y="{margin + 54}" class="sub">Green = good, yellow = acceptable, Red = weak</text>')
    y += title_h

    svg_parts.append(f'<rect x="{x0}" y="{y}" width="{col_metric}" height="{header_h}" fill="{HEADER_BG}" stroke="{GRID}"/>')
    svg_parts.append(f'<rect x="{x0 + col_metric}" y="{y}" width="{col_range}" height="{header_h}" fill="{HEADER_BG}" stroke="{GRID}"/>')
    svg_parts.append(f'<rect x="{x0 + col_metric + col_range}" y="{y}" width="{col_label}" height="{header_h}" fill="{HEADER_BG}" stroke="{GRID}"/>')

    svg_parts.append(f'<text x="{x0 + 12}" y="{y + 25}" class="hdr">metric</text>')
    svg_parts.append(f'<text x="{x0 + col_metric + 12}" y="{y + 25}" class="hdr">Range (%)</text>')
    svg_parts.append(f'<text x="{x0 + col_metric + col_range + 12}" y="{y + 25}" class="hdr">Performance Label</text>')
    y += header_h

    for metric, rows in blocks:
        block_h = len(rows) * row_h

        svg_parts.append(f'<rect x="{x0}" y="{y}" width="{col_metric}" height="{block_h}" fill="{METRIC_BG}" stroke="{GRID}"/>')
        svg_parts.append(f'<text x="{x0 + 12}" y="{y + 22}" class="metric">{esc(metric)}</text>')

        for i, (rng, label, bg) in enumerate(rows):
            yy = y + i * row_h
            svg_parts.append(f'<rect x="{x0 + col_metric}" y="{yy}" width="{col_range}" height="{row_h}" fill="{bg}" stroke="{GRID}"/>')
            svg_parts.append(f'<rect x="{x0 + col_metric + col_range}" y="{yy}" width="{col_label}" height="{row_h}" fill="{bg}" stroke="{GRID}"/>')
            svg_parts.append(f'<text x="{x0 + col_metric + 12}" y="{yy + 23}" class="txt">{esc(rng)}</text>')
            svg_parts.append(f'<text x="{x0 + col_metric + col_range + 12}" y="{yy + 23}" class="txt">{esc(label)}</text>')

        y += block_h

    svg_parts.append("</g>")
    svg_parts.append("</svg>")

    svg = "".join(svg_parts)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return html.Img(
        src=f"data:image/svg+xml;base64,{b64}",
        style={"width": "100%", "maxWidth": f"{W}px", "height": "auto"},
    )


def _perf_color(metric: str, score: float) -> str:
    m = metric.lower()

    def hi_better(th_red, th_yellow):
        if score < th_red:
            return "#FF0000"
        if score < th_yellow:
            return "#D4BF01"
        return "#02B842"

    def lo_better(th_green, th_yellow):
        if score <= th_green:
            return "#02B842"
        if score <= th_yellow:
            return "#D4BF01"
        return "#FF0000"

    if m == "precision":
        return hi_better(0.50, 0.80)
    if m == "recall":
        return hi_better(0.50, 0.80)
    if m in ("f1", "f1-score", "f1_score"):
        return hi_better(0.50, 0.80)
    if m in ("roc_auc_score", "roc_auc"):
        return hi_better(0.65, 0.85)
    if m in ("log_loss", "logloss"):
        return lo_better(0.20, 0.60)

    return "#9CA3AF"


def _dot(color: str):
    return html.Span(
        style={
            "display": "inline-block",
            "width": "10px",
            "height": "10px",
            "borderRadius": "50%",
            "backgroundColor": color,
            "marginRight": "8px",
            "boxShadow": "0 0 0 2px rgba(0,0,0,0.06)",
            "verticalAlign": "middle",
        }
    )


def _badge(metric: str, score: float):
    c = _perf_color(metric, score)
    return dbc.Badge(
        [
            _dot(c),
            html.Span(f"{metric}: {score:.3f}", style={"color": "#111827"}),
        ],
        color="light",
        pill=True,
        class_name="me-2 mb-2",
        style={
            "padding": "10px 12px",
            "border": "1px solid #E5E7EB",
            "backgroundColor": "#FFFFFF",
            "color": "#111827",
        },
    )



def _get_metrics_dict(explainer, pos_label=None):
    # Compatible con distintas versiones
    for fn_name in ("metrics", "model_performance"):
        fn = getattr(explainer, fn_name, None)
        if callable(fn):
            try:
                out = fn(pos_label=pos_label)
                if hasattr(out, "to_dict"):
                    out = out.to_dict()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    return None

def metrics_color_panel(explainer, pos_label=None):
    metrics = _get_metrics_dict(explainer, pos_label=pos_label)
    if not metrics:
        return dbc.Alert("Imposible to read the results of the features.", color="warning")

    ordered = ["precision", "recall", "f1", "roc_auc_score", "log_loss"]
    chips = []
    for k in ordered:
        if k in metrics:
            chips.append(_badge(k, float(metrics[k])))

    return html.Div(
        [
            html.Div(
                "Quick and easy interpretation",
                style={"fontWeight": "700", "marginBottom": "8px", "color": "#111827"},
            ),
            html.Div(chips, style={"display": "flex", "flexWrap": "wrap"}),

            html.Div(
                [
                    html.Span("● ", style={"color": "#02B842", "fontWeight": "900"}),
                    html.Span("good", style={"color": "#111827"}),

                    html.Span("  ● ", style={"color": "#D4BF01", "fontWeight": "900", "marginLeft": "12px"}),
                    html.Span("acceptable", style={"color": "#111827"}),

                    html.Span("  ● ", style={"color": "#FF0000", "fontWeight": "900", "marginLeft": "12px"}),
                    html.Span("weak", style={"color": "#111827"}),
                ],
                style={"marginTop": "8px", "fontSize": "13px"},
            ),
        ]
    )



class ClassificationStatsTab(ExplainerComponent):
    def __init__(self, explainer, title="Classification Stats", name=None,
                 hide_selector=True, pos_label=None, **kwargs):
        super().__init__(explainer, title, name)

        self.pos_label = pos_label
        self.bench_info_btn_id = f"{self.name}-bench-info-btn"

        # Esta es la tabla ORIGINAL de explainerdashboard (no la tocamos)
        self.summary = ClassifierModelSummaryComponent(
            explainer, name=self.name + "0",
            hide_selector=hide_selector, pos_label=pos_label, 
            show_metrics=["precision", "recall", "f1", "roc_auc_score", "log_loss"],
            **kwargs
        )

        self.rocauc = RocAucComponent(
            explainer, name=self.name + "1",
            hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.confusionmatrix = ConfusionMatrixComponent(
            explainer, name=self.name + "2",
            hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )
        self.classification = ClassificationComponent(
            explainer, name=self.name + "3",
            hide_selector=hide_selector, pos_label=pos_label, **kwargs
        )

    def layout(self):
        info_btn = dbc.Button(
            "+ info",
            id=self.bench_info_btn_id,
            color="primary",
            size="sm",
            style={"borderRadius": "10px", "fontWeight": "600"},
        )

        pop = dbc.Popover(
            [
                dbc.PopoverHeader("Classification Performance Benchmarks"),
                dbc.PopoverBody(_classification_benchmarks_img()),
            ],
            target=self.bench_info_btn_id,
            trigger="hover",
            placement="bottom",
            offset=12,
            style={"maxWidth": "1020px"},
        )

        # Card wrapper para poner header propio con +info, sin tocar la tabla del componente
        metrics_wrapper = dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(html.H4("Model performance metrics", className="mb-0"), width="auto"),
                            dbc.Col(info_btn, width="auto", class_name="ms-auto"),
                        ],
                        align="center",
                        class_name="g-2",
                    )
                ),
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(self.summary.layout(), width=8),
                            dbc.Col(metrics_color_panel(self.explainer, pos_label=self.pos_label), width=4),
                        ],
                        class_name="g-3",
                    )
                ),
                pop,
            ],
            class_name="shadow-sm",
        )

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(metrics_wrapper, width=6),
                        dbc.Col(self.confusionmatrix.layout(), width=6),
                    ],
                    class_name="mt-4 gx-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(self.rocauc.layout(), width=6),
                        dbc.Col(self.classification.layout(), width=6),
                    ],
                    class_name="mt-4 gx-4",
                ),
            ],
            fluid=True,
        )
