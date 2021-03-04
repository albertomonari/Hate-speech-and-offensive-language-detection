import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

colors = {
    'background': '#111111',
    'text': '#FF603F',
    'footer': '#B8B8B8'
}

def Nav():
    nav = dbc.Nav(
        [
            dbc.Row(
                [
                dbc.NavItem(dbc.NavLink("Hate Speech Automatic Detector", active=True, href="/user",style={'color': colors['text']})),
                dbc.NavItem(dbc.NavLink("How the algorithm is built", href="/algorithm", style={'color': colors['text']})),
                ],
                style={"marginTop": 10, "marginBottom":20}
            )
        ]
    )
    return nav
