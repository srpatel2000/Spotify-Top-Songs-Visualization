# IMPORT STATEMENTS

import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc #pip install dash-bootstrap-components
from dash.dependencies import Input, Output

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server

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


    # dcc.Dropdown(
    #     id='year_slider',
    #     options = [
    #         {'label':'2000', 'value':2000},
    #         {'label':'2001', 'value':2001},
    #         {'label':'2002', 'value':2002},
    #         {'label':'2003', 'value':2003},
    #         {'label':'2004', 'value':2004},
    #         {'label':'2005', 'value':2005},  
    #         {'label':'2006', 'value':2006},
    #         {'label':'2007', 'value':2007},
    #         {'label':'2008', 'value':2008},
    #         {'label':'2009', 'value':2009},
    #         {'label':'2010', 'value':2010},
    #         {'label':'2011', 'value':2011},
    #         {'label':'2012', 'value':2012},
    #         {'label':'2013', 'value':2013},
    #         {'label':'2014', 'value':2014},
    #         {'label':'2015', 'value':2015},
    #         {'label':'2016', 'value':2016},
    #         {'label':'2017', 'value':2017},
    #         {'label':'2018', 'value':2018},
    #         {'label':'2019', 'value':2019}
    #     ],
    #     value = ['2019'],
    #     multi = True),

    html.Div([
        dcc.Graph(id='genres_graph')
    ]),

    dcc.Slider(
    id='genres_year_slider',
    min=top_songs['year'].min(),
    max=top_songs['year'].max(),
    value=top_songs['year'].min(),
    marks={str(year): str(year) for year in top_songs['year'].unique()},
    step=None
    ),

    html.Div([
        dcc.Graph(id='artists_graph')
    ]),

    dcc.Slider(
    id='artists_year_slider',
    min=top_songs['year'].min(),
    max=top_songs['year'].max(),
    value=top_songs['year'].min(),
    marks={str(year): str(year) for year in top_songs['year'].unique()},
    step=None
    ),

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

#Genres graph callback
@app.callback(
    Output(component_id='genres_graph', component_property='figure'),
    [Input(component_id='genres_year_slider', component_property='value')]
)
def update_genres_graph(year_val):
    line_copy = top_songs.copy()

    #Preprocessing genre column
    line_copy['artist_genre'] = line_copy['artist_genre'].apply(lambda x : x.replace('[', ''))
    line_copy['artist_genre'] = line_copy['artist_genre'].apply(lambda x : x.replace(']', ''))
    line_copy['artist_genre'] = line_copy['artist_genre'].apply(lambda x : x.replace("'", ''))
    line_copy['artist_genre'] = line_copy['artist_genre'].apply(lambda x : x.split(','))

    columns = ['artist_genre', 'year']

    line_copy = line_copy[columns]

    # Filter only the year input
    line_copy = line_copy.loc[line_copy['year'] == year_val]

    # Count the song genres
    genres = dict()
    for lst in line_copy['artist_genre']:
    
        for genre in lst:
            if genre in genres:
                genres[genre] += 1
            else:
                genres[genre] = 1

    # Get top artist genres and the song counts
    top_genres_keys = sorted(genres, key=genres.get, reverse=True)[1:10]
    top_genres_values = [genres[key] for key in top_genres_keys]
    top_genres_values = [x/sum(top_genres_values) * 100 for x in top_genres_values] # Percentize the values

    # Plot the graph
    fig = px.bar(
        x=top_genres_keys, 
        y=top_genres_values,
        color = top_genres_keys,
        title = 'Top Artist Genres in {0}'.format(str(year_val)),
        labels={'x':'Artist Genre', 'y':'Percent of Songs (%)'}
    )
    
    return fig

@app.callback(
    Output(component_id='artists_graph', component_property='figure'),
    [Input(component_id='artists_year_slider', component_property='value')]
)
def update_artists_graph(year_val):
    line_copy = top_songs.copy()

    #Preprocessing artists column
    line_copy['artist_name'] = line_copy['artist_name'].apply(lambda x : x.replace('[', ''))
    line_copy['artist_name'] = line_copy['artist_name'].apply(lambda x : x.replace(']', ''))
    line_copy['artist_name'] = line_copy['artist_name'].apply(lambda x : x.replace("'", ''))
    line_copy['artist_name'] = line_copy['artist_name'].apply(lambda x : x.split(','))

    columns = ['artist_name', 'year']

    #Filter out dataframe according to input year
    line_copy = line_copy[columns]
    line_copy = line_copy.loc[line_copy['year'] == year_val]

    #Count the artists' song counts
    artists_dict = dict()
    for artists in line_copy['artist_name']:
        for artist in artists:
            if artist in artists_dict:
                artists_dict[artist] += 1
            else:
                artists_dict[artist] = 1

    # Get top artists names and the songs
    top_artists_keys = sorted(artists_dict, key=artists_dict.get, reverse=True)[1:15]
    top_artists_values = [artists_dict[key] for key in top_artists_keys]
    #top_artists_values  = [x/sum(top_artists_values) * 100 for x in top_artists_values] # Percentize the values
    
    fig = px.bar(x=top_artists_keys,
                 y=top_artists_values,
                 color = top_artists_keys,
                 title = 'Top Artists in {0}'.format(str(year_val)),
                 labels={'x':'Artist Name','y':'Song Count'})
    return fig
# -------------------------------------------------------------------------------------
# RUN THE APP 
if __name__ == '__main__':
    app.run_server(debug=True)