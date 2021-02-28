import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from textblob import TextBlob
import pycountry

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

import user
import algorithm
import model
from model import get_predictions

colors = {
    'background': '#111111',
    'text': '#FF603F',
    'footer': '#B8B8B8'
}

app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# index page callback
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/user' or pathname == '/':
        return user.user_page_layout
    elif pathname == '/algorithm':
        return algorithm.algorithm_page_layout
    else:
        return '404'

app.config['suppress_callback_exceptions'] = True

# page 1 callback
@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('classify', 'n_clicks')],
    [dash.dependencies.State('input_text', 'value')])

def update_output(n_clicks, value):
    if value is not None:
        value.encode('utf8')
        sentence = TextBlob(value)
        iso_code = sentence.detect_language()
        language = pycountry.languages.get(alpha_2=iso_code)
        if iso_code=="en":
            language_sentence = html.P("Your sentence is in English")
            tweet_class = get_predictions(value)
            return (html.P([html.P('Your sentence is classified as {}.'.format(tweet_class), style={'textAlign': 'center', 'font-size': 20}),html.Br(),'The sentence to classify was "{}".'.format(value),html.Br(),\
            language_sentence,html.Br(),'You tried to classify {} sentence(s) since you started using this app.'.format(n_clicks)]))
        else:
            language_sentence = html.P("Your sentence is in "+language.name+", please write a sentence in English")
    else:
        return
    return (html.P([html.P(language_sentence,style={'textAlign': 'center', 'font-size': 20, 'color': colors['text']}),html.Br(),'The sentence to classify was "{}".'.format(value),html.Br(),'You tried to classify {} sentence(s) since you started using this app.'\
    .format(n_clicks)]))

if __name__ == '__main__':
    app.run_server(debug=True)
