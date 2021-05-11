# IMPORT STATEMENTS

import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# -------------------------------------------------------------------------------------
# READ IN DATA
top_songs = pd.read_csv('top_songs.csv')

# -------------------------------------------------------------------------------------
# APP LAYOUT 

app.layout = html.Div([

    html.H1("Spotify Top Songs Dashboard (DSC 106)", style={'text-align': 'center'}),

    dcc.Checklist(
        id='my_checklist',
        options = [
            {'label':'Danceability', 'value':'danceability'},
            {'label':'Energy', 'value':'energy'},
            {'label':'Key', 'value':'key'},
            {'label':'Loudness', 'value':'loudness'},
            {'label':'Mode', 'value':'mode'},
            {'label':'Speechiness', 'value':'speechiness'},
            {'label':'Acousticness', 'value':'acousticness'},
            {'label':'Instrumentalness', 'value':'instrumentalness'},
            {'label':'Liveness', 'value':'liveness'},
            {'label':'Valence', 'value':'valence'},
            {'label':'Tempo', 'value':'tempo'},
            {'label':'Duration', 'value':'duration'},
            {'label':'Time Signature', 'value':'time_signature'},
        ],  value =['danceability']),

         html.Div([
            dcc.Graph(id='line_graph')
        ]),

])

# -------------------------------------------------------------------------------------
# CONNECT THE PLOTLY GRAPHS WITH DASH COMPONENTS 
@app.callback(
    Output(component_id='line_graph', component_property='figure'),
    [Input(component_id='my_checklist', component_property='value')]
)

def update_line_graph(selected_columns):
    line_copy = top_songs.copy()

    columns = selected_columns +  ['year']

    line_copy = line_copy[columns]
    line_copy = top_songs.groupby('year', as_index=False).mean()

    fig = px.line(
        data_frame = line_copy,
        x = 'year', 
        y = selected_columns
    )

    return fig

# -------------------------------------------------------------------------------------
# RUN THE APP 
if __name__ == '__main__':
    app.run_server(debug=True)