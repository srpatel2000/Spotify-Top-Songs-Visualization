# IMPORT STATEMENTS

# data processing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# daat visualizing
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

sidebar = html.Div(
    [
        html.H3("Spotify Top Songs Dashboard", className="display-4", style = {"font-size": "30px", "font-weight": "600"}),
        # button
        html.A(html.Button("Renaldy Herlim", style={'background-color': '#212121', 'border': '2px solid', 'border-radius': '10px', 'font-family':"Montserrat", 'text-align': 'center', "font-weight": "600"}), href='https://www.linkedin.com/in/renaldy-herlim/', target='_blank'),
        html.A(html.Button("Siddhi Patel", style={'background-color': '#212121', 'border': '2px solid', 'border-radius': '10px', 'font-family':"Montserrat", 'text-align': 'center', "font-weight": "600"}), href='https://www.linkedin.com/in/siddhipatel-stu/', target='_blank'),
        html.Hr(),
        html.P(
            "This dashboard was created for the SP'21 iteration of DSC106. It conveys different information about top songs throughout the years on Spotify.", className="lead", style = {"font-size": "15px"}
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Goals", href="/", active="exact"),
                dbc.NavLink("Audio Features Over the Years", href="/page-1", active="exact"),
                dbc.NavLink("Top Genres Throughout the Years", href="/page-2", active="exact"),
                dbc.NavLink("Top Artists Throughout the Years", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Iframe(src = "https://open.spotify.com/embed/playlist/37i9dQZF1DWVRSukIED0e9", style={"width":"320px", "height":"80px", "shadow":"none"}, allow="encrypted-media")
    ],
    style={
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "black",
    },
)

content = html.Div(id="page-content", children=[], style={"margin-left": "22rem", "margin-right": "2rem", "margin-top":"3rem", "padding": "5rem 1rem"})

app.layout = html.Div([

    dcc.Location(id="url"),

    # # line graph checklist
    # dcc.Checklist(
    #     style={'color': 'white', 'margin-top': '10px'},
    #     id='my_checklist',
    #     options = [
    #         {'label':'Danceability', 'value':'danceability'},
    #         {'label':'Energy', 'value':'energy'},
    #         {'label':'Key', 'value':'key'},
    #         {'label':'Loudness', 'value':'loudness'},
    #         {'label':'Mode', 'value':'mode'},
    #         {'label':'Speechiness', 'value':'speechiness'},
    #         {'label':'Acousticness', 'value':'acousticness'},
    #         {'label':'Instrumentalness', 'value':'instrumentalness'},
    #         {'label':'Liveness', 'value':'liveness'},
    #         {'label':'Valence', 'value':'valence'},
    #         {'label':'Tempo', 'value':'tempo'},
    #         {'label':'Duration', 'value':'duration'},
    #         {'label':'Time Signature', 'value':'time_signature'},
    #     ],  value =['danceability']),

    #     # line graph
    #     html.Div([
    #         dcc.Graph(id='line_graph')
    #     ]),

        # # genres bar chart title
        # html.Div([
        #  dcc.Graph(id='genres_graph')
        # ]),

        # # genres bar chart slider
        # dcc.Slider(
        # id='genres_year_slider',
        # min=top_songs['year'].min(),
        # max=top_songs['year'].max(),
        # value=top_songs['year'].min(),
        # marks={str(year): str(year) for year in top_songs['year'].unique()},
        # step=None
        # ),


        # # artists bar chart title
        # html.Div([
        #     dcc.Graph(id='artists_graph')
        # ]),

        # # artists bar chart slider
        # dcc.Slider(
        # id='artists_year_slider',
        # min=top_songs['year'].min(),
        # max=top_songs['year'].max(),
        # value=top_songs['year'].min(),
        # marks={str(year): str(year) for year in top_songs['year'].unique()},
        # step=None
        # ),

        sidebar,
        content

])

# -------------------------------------------------------------------------------------
# CONNECT THE PLOTLY GRAPHS WITH DASH COMPONENTS 

# Line graph callback
@app.callback(
    Output(component_id='line_graph', component_property='figure'),
    [Input(component_id='my_checklist', component_property='value')]
)

def update_line_graph(selected_columns):
    '''This function scales the columns in our dataset and creates a line graph for the different audio features.'''
    line_copy = top_songs.copy() # always work with a copy of the original dataset

    columns = selected_columns +  ['year']

    line_copy = line_copy[columns] # filter dataset for relevant columns
    line_copy = top_songs.groupby('year', as_index=False).mean() # retrieve the mean values of audio features for every year

    # scale the qualitative features to be between 0 and 1 to compare different values
    scaler = MinMaxScaler()
    line_copy_rel_cols = line_copy[line_copy.columns[2:]]
    line_copy_scaled = pd.DataFrame(scaler.fit_transform(line_copy_rel_cols), columns=line_copy_rel_cols.columns)

    # add scaled columns to original dataset
    for cols in line_copy_scaled.columns:
        line_copy[cols] = line_copy_scaled[cols]

    # create line chart
    fig = px.line(
        data_frame = line_copy,
        x = 'year', 
        y = selected_columns
    )

    fig.update_layout({
    'plot_bgcolor': ' #212121',
    'paper_bgcolor': ' #212121',
    'font_color': 'white',
    })

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

    fig.update_layout({
    'plot_bgcolor': ' #212121',
    'paper_bgcolor': ' #212121',
    'font_color': 'white',
    })
    
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

    fig.update_layout({
    'plot_bgcolor': ' #212121',
    'paper_bgcolor': ' #212121',
    'font_color': 'white',
    })

    return fig

# CONTENT CALLBACK
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)

def render_page_content(pathname):
    if pathname == "/":
        return [
                # html.H1('Kindergarten in Iran',
                #         style={'textAlign':'center'}),
                # line graph checklist
            dcc.Checklist(
                style={'color': 'white', 'margin-top': '10px'},
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

                # line graph
                html.Div([
                    dcc.Graph(id='line_graph')
                ])
        ]

    elif pathname == "/page-1":
        return [
                # html.H1('Grad School in Iran',
                #         style={'textAlign':'center'}),
                # genres bar chart title
                html.Div([
                dcc.Graph(id='genres_graph')
                ]),

                # genres bar chart slider
                dcc.Slider(
                id='genres_year_slider',
                min=top_songs['year'].min(),
                max=top_songs['year'].max(),
                value=top_songs['year'].min(),
                marks={str(year): str(year) for year in top_songs['year'].unique()},
                step=None
                )
                ]

    elif pathname == "/page-2":
        return [
                # html.H1('High School in Iran',
                #         style={'textAlign':'center'}),
                # artists bar chart title
                html.Div([
                    dcc.Graph(id='artists_graph')
                ]),

                # artists bar chart slider
                dcc.Slider(
                id='artists_year_slider',
                min=top_songs['year'].min(),
                max=top_songs['year'].max(),
                value=top_songs['year'].min(),
                marks={str(year): str(year) for year in top_songs['year'].unique()},
                step=None
                ),
        ]

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
# -------------------------------------------------------------------------------------
# RUN THE APP 
if __name__ == '__main__':
    app.run_server(debug=True)