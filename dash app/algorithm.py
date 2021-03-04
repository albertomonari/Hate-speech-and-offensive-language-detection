import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import base64

colors = {
    'background': '#111111',
    'text': '#FF603F',
    'footer': '#B8B8B8'
}

INFOGRAPHIC_ALGO = "infographic.jpg"
test_base64 = base64.b64encode(open(INFOGRAPHIC_ALGO, 'rb').read()).decode('ascii')

from navbar import Navbar
navbar = Navbar()

from nav import Nav
nav = Nav()

infographic = dbc.Card(
    [
        dbc.CardHeader(html.H2("How the algorithm is built"),style={'textAlign': 'center', 'color': colors['text'], 'margin-top':"10px", 'margin-down':"15px"}),
        dbc.CardBody(
            html.Img(src='data:image/png;base64,{}'.format(test_base64), width="100%"),
        )
    ]
)

body = dbc.Container(
    [
        dbc.Row([dbc.Col(nav)]),
        dbc.Row([dbc.Col(infographic)])
    ])

algorithm_page_layout = html.Div(children=[
                            navbar,
                            body
                            ])
