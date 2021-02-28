import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

colors = {
    'background': '#111111',
    'text': '#FF603F',
    'footer': '#B8B8B8'
}

HATE_LOGO = "https://townsquare.media/site/656/files/2019/03/Hate-Speech-Getty-Images-by-iarti.jpg?w=980&q=75"
TBS_LOGO = "https://www.tbs-education.fr/content/uploads/sites/4/2020/02/1-logo-tbs-2019-rond-cmjn-fond-rouge.png"

github_link = dbc.Row(
            [
                html.A(
                        children=[
                            'View on GitHub'
                        ],
                        href="https://github.com/albertomonari/Hate-speech-and-offensive-language-detection",
                        style={'color': 'white'}
                    ),
            ],
             #no_gutters=True,
             className="ml-auto flex-nowrap mt-3 mt-md-0",
             align="center",
)

def Navbar():
    navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=HATE_LOGO, height="40px")),
                    dbc.Col(html.Img(src=TBS_LOGO, height="40px")),
                    dbc.Col(dbc.NavbarBrand("Hate Speech Detection", className="ml-2")),
                ],
            ),
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(github_link, id="navbar-collapse", navbar=True),
    ],
    color=colors['background'],
    dark=True,
)
    return navbar
