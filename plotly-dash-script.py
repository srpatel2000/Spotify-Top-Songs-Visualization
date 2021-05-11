# import statements

import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# -------------------------------------------------------------------------------------
# APP LAYOUT 

app.layout = html.Div([

    html.H1("Spotify Top Songs Dashboard (DSC 106)", style={'text-align': 'center'}),

])

# -------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)