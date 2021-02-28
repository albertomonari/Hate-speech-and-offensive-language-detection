import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

colors = {
    'background': '#111111',
    'text': '#FF603F',
    'footer': '#B8B8B8'
}

from navbar import Navbar
navbar = Navbar()

from nav import Nav
nav = Nav()

user_app = dbc.Card(
    [
        dbc.CardHeader(html.H2("Hate Speech Automatic Detector"),style={'textAlign': 'center', 'color': colors['text'], 'margin-top':"10px", 'margin-down':"15px"}),
        dbc.CardBody(
            [
            html.P('''Hate speech is any form of expression through which speakers intend to vilify, humiliate, or incite hatred against a group or a class of persons
            on the basis of race, religion, skin color sexual identity, gender identity, ethnicity, disability, or national origin.
            '''),
            html.P('''This application will allow you to check whether a certain type of text in English can be considered as hate speech, as an offensive language or neither.'''),
            dbc.Input(id='input_text', className="mb-3", placeholder="Type your speech here", style={'width': '100%', 'height': '100%'}, bs_size="md"),
            ]
        ),
        dcc.Loading([
        dbc.Button("Classify your sentence", id='classify', n_clicks=0, style={"marginLeft": 20}, outline=True, color="secondary", className="mr-1"),
        html.Div(id='container-button-basic', children='Write a sentence and press the button', style={"marginLeft": 20, "marginTop": 10})
        ]),
        dbc.CardFooter(children="Developped by Alberto Monari, Maéva Mecker, Louise Dietrich, Marina Serrano Diego, Kevin Da Silva and Ivan Ingaud-Jaubert",
        style={"marginTop": 70, "color":colors['footer']})
    ]
)


body = dbc.Container(
    [
        dbc.Row([dbc.Col(nav)]),
        dbc.Row([dbc.Col(user_app)])
    ])

user_page_layout = html.Div(children=[
                            navbar,
                            body
                    ])
