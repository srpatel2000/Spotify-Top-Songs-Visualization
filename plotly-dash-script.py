# IMPORT STATEMENTS

# data processing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# daat visualizing
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc #pip install dash-bootstrap-components
from dash.dependencies import Input, Output
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP], suppress_callback_exceptions = True)
server=app.server

# -------------------------------------------------------------------------------------
# READ IN DATA
top_songs = pd.read_csv('top_songs.csv')
main_genres_options = ['pop', 'r&b', 'hip hop', 'rap', 'country', 'rock', 'metal', 'house', 'edm', 'indie', 'punk', 'trap', 'reggae', 'latin', 'alternative']

# -------------------------------------------------------------------------------------
# APP LAYOUT 

sidebar = html.Div(
    [
        html.H3("Spotify Top Songs Dashboard", className="display-4", style = {"font-size": "30px", "font-weight": "600", 'color': '#1DB954'}),
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
                dbc.NavLink("Goals", href="/", active="exact", style={'text-align':'left'}),
                dbc.NavLink("Audio Features Over the Years", href="/page-1", active="exact", style={'text-align':'left'}),
                dbc.NavLink("Sub-Genre Audio Features Over the Years", href="/page-2", active="exact", style={'text-align':'left'}),
                dbc.NavLink("Top Artists Over the Years", href="/page-3", active="exact", style={'text-align':'left'}),
                dbc.NavLink("Top Genres Over the Years", href="/page-4", active="exact", style={'text-align':'left'}),
                dbc.NavLink("Correlation Matrix of Audio Features", href="/page-5", active="exact", style={'text-align':'left'}),
                dbc.NavLink("Features and Genres Classification", href="/page-6", active="exact", style={'text-align':'left'}),

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

content = html.Div(id="page-content", children=[], style={"margin-left": "22rem", "margin-right": "2rem", "margin-top":"0rem", "padding": "2rem 1rem"})

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

# -------------------------------------------------------------------------------------
# CONNECT THE PLOTLY GRAPHS WITH DASH COMPONENTS 

# CONTENT CALLBACK
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)

def render_page_content(pathname):
    if pathname == "/":
        return [
            html.H1('Goals', style={'textAlign':'left', "color":"white", "border-bottom": "1px solid #535353", "line-height": "80px"}),
            html.Hr(),
            html.H6('The goal of this project was for us to be able to learn new concepts while also visualizing interesting information about the "Top Songs" playlists on Spotify.', style={'textAlign':'left', "color":"white"}),
            html.Hr(),
            html.H6('Through this project we were able refine our skills in: UI/UX, data retrieval, and data visualization.', style={'textAlign':'left', "color":"white"})
        ]
    elif pathname == "/page-5":
        return [
            html.H1('Correlation Matrix of Audio Features', style={'textAlign':'left', "color":"white", "border-bottom": "1px solid #535353", "line-height": "80px"}),

            html.Div([
                dcc.Dropdown(
                    id='graph-type',
                    options=[{'label': 'Correlation Matrix', 'value': 'correlation'}
                        ],
                    value='correlation',
                    clearable=False
                )
                ]  
            ),

            html.Div([
                dcc.Graph(id='matrix'),
            ]),
            
        ]
    elif pathname == "/page-6":
        available_indicators = ['r&b', 'hip hop', 'country', 'rock', 'metal', 'edm', 'indie', 'pop']
        return [
            html.H1('Features and Genres Classification', style={'textAlign':'left', "color":"white", "border-bottom": "1px solid #535353", "line-height": "80px"}),
            html.Hr(),

            html.Div([

                html.Div([
                    html.P('1st Genre', style = {'color': 'white', 'font-family':"Montserrat", 'text-decoration': 'underline'}),
                    dcc.Dropdown(
                        id='first-genre',
                        options=[{'label': i, 'value': i} for i in available_indicators],
                        value='pop',
                        clearable=False
                    ),
                ],
                style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.P('2nd Genre', style = {'color': 'white', 'font-family':"Montserrat", 'text-decoration': 'underline'}),
                    dcc.Dropdown(
                        id='second-genre',
                        options=[{'label': i, 'value': i} for i in available_indicators],
                        value='rock',
                        clearable=False
                    ),
                ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            
            html.Hr(),

            html.Div(
                [html.P('2nd Audio Feature', style = {'color': 'white', 'font-family':"Montserrat", 'text-decoration': 'underline'}),

                # line graph checklist
                dcc.Dropdown(
                    id='first-feature',
                    options = [
                        {'label':' Danceability', 'value':'danceability'},
                        {'label':' Energy', 'value':'energy'},
                        {'label':' Key', 'value':'key'},
                        {'label':' Loudness', 'value':'loudness'},
                        {'label':' Mode', 'value':'mode'},
                        {'label':' Speechiness', 'value':'speechiness'},
                        {'label':' Acousticness', 'value':'acousticness'},
                        {'label':' Instrumentalness', 'value':'instrumentalness'},
                        {'label':' Liveness', 'value':'liveness'},
                        {'label':' Valence', 'value':'valence'},
                        {'label':' Tempo', 'value':'tempo'},
                        {'label':' Duration', 'value':'duration'},
                        {'label':' Time Signature', 'value':'time_signature'},],
                    value ='danceability', 
                    clearable=False),
                ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'}
            ),
            html.Div(
                [html.P('1st Audio Feature', style = {'color': 'white', 'font-family':"Montserrat", 'text-decoration': 'underline'}),

                # line graph checklist
                dcc.Dropdown(
                    id='second-feature',
                    options = [
                        {'label':' Danceability', 'value':'danceability'},
                        {'label':' Energy', 'value':'energy'},
                        {'label':' Key', 'value':'key'},
                        {'label':' Loudness', 'value':'loudness'},
                        {'label':' Mode', 'value':'mode'},
                        {'label':' Speechiness', 'value':'speechiness'},
                        {'label':' Acousticness', 'value':'acousticness'},
                        {'label':' Instrumentalness', 'value':'instrumentalness'},
                        {'label':' Liveness', 'value':'liveness'},
                        {'label':' Valence', 'value':'valence'},
                        {'label':' Tempo', 'value':'tempo'},
                        {'label':' Duration', 'value':'duration'},
                        {'label':' Time Signature', 'value':'time_signature'},],
                    value ='speechiness', 
                    clearable=False)
                    ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'}
            ),
            html.Hr(),
            
            ### Graph for Classification Scatterplot ###
            html.Div([
                dcc.Graph(id='classification-scatterplot'),
            ], style={'display':'inline-block', "margin-left":"220px", "margin-top": "-500px", "width": "800px", "height": "490px"}),    
            
        ]
    elif pathname == "/page-1":
        return [

            html.H1('Audio Features Over the Years', style={'textAlign':'left', "color":"white", "border-bottom": "1px solid #535353", "line-height": "80px"}),
            html.Hr(),

            html.Div(
                [html.P('Select Audio Feature', style = {'color': 'black', 'font-family':"Montserrat", 'text-decoration': 'underline'}),

                # line graph checklist
                dcc.Checklist(
                    style={'color': 'black', 'font-family':"Montserrat", 'font-weight': "300", "font-size": "small"},
                    id='my_checklist',
                    options = [
                        {'label':' Danceability', 'value':'danceability'},
                        {'label':' Energy', 'value':'energy'},
                        {'label':' Key', 'value':'key'},
                        {'label':' Loudness', 'value':'loudness'},
                        {'label':' Mode', 'value':'mode'},
                        {'label':' Speechiness', 'value':'speechiness'},
                        {'label':' Acousticness', 'value':'acousticness'},
                        {'label':' Instrumentalness', 'value':'instrumentalness'},
                        {'label':' Liveness', 'value':'liveness'},
                        {'label':' Valence', 'value':'valence'},
                        {'label':' Tempo', 'value':'tempo'},
                        {'label':' Duration', 'value':'duration'},
                        {'label':' Time Signature', 'value':'time_signature'},
                ],  value =['danceability'], labelStyle = dict(display='block'))], style={'display':'inline-block', 'border-radius': '10px', 'background-color': 'white', 'border': 'solid white', "padding": "20px", "margin": "0"}),

                # line graph
                html.Div([
                    dcc.Graph(id='line_graph')
                ], style={'display':'inline-block', "margin-left":"500px", "margin-top": "-500px", "width": "800px"})
        ]

    elif pathname == "/page-4":
        return [

                html.H1('Top Genres Throughout the Years', style={'textAlign':'left', "color":"white", "border-bottom": "1px solid #535353", "line-height": "80px"}),
                html.Hr(),

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

    elif pathname == "/page-3":
        return [

                html.H1('Top Artists Throughout the Years', style={'textAlign':'left', "color":"white", "border-bottom": "1px solid #535353", "line-height": "80px"}),
                html.Hr(),

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

    elif pathname == "/page-2":
        return [

                html.H1('Sub-Genre Audio Features Over the Years', style={'textAlign':'left', "color":"white", "border-bottom": "1px solid #535353", "line-height": "80px"}),
                html.Hr(),


                #### Genres exploration graph ####
                # Main Genre Selection
        html.Label("Select Main Genre", style={'fontSize':15, 'color':'white'}),
            dcc.Dropdown(
                id = 'main_genres',
                options = [{'label': i, 'value': i} for i in main_genres_options],
                value = 'pop',
                multi=True
            ),
            html.Hr(),
    
            # Multi sub-genres selection
         html.Label("Select Sub-Genre", style={'fontSize':15, 'color':'white'}),
            dcc.Dropdown(
                id = 'sub_genres',
                options = [],
                multi = True
                )   ,
                html.Hr(),

            #Audio feature selection
            html.Div([
                html.P('Select Audio Feature', style = {'color': 'black', 'font-family':"Montserrat", 'text-decoration': 'underline'}),
                dcc.RadioItems(
                        style={'color': 'black', 'font-family':"Montserrat", 'font-weight': "300", "font-size": "small"},
                        id='feature_checklist',
                        options = [
                            {'label':' Danceability', 'value':'danceability'},
                            {'label':' Energy', 'value':'energy'},
                            {'label':' Key', 'value':'key'},
                            {'label':' Loudness', 'value':'loudness'},
                            {'label':' Mode', 'value':'mode'},
                            {'label':' Speechiness', 'value':'speechiness'},
                            {'label':' Acousticness', 'value':'acousticness'},
                            {'label':' Instrumentalness', 'value':'instrumentalness'},
                            {'label':' Liveness', 'value':'liveness'},
                            {'label':' Valence', 'value':'valence'},
                            {'label':' Tempo', 'value':'tempo'},
                            {'label':' Duration', 'value':'duration'},
                            {'label':' Time Signature', 'value':'time_signature'},
                    ],  value ='danceability', labelStyle = dict(display='block'))
            ], style={'display':'inline-block', 'border-radius': '10px', 'background-color': 'white', 'border': 'solid white', "padding": "20px", "margin": "0"}),

            html.Hr(),
            
            #Graph of genres exploration
            html.Div([
                dcc.Graph(id='genres_explore_fig'),
            ], style={'display':'inline-block', "margin-left":"220px", "margin-top": "-500px", "width": "800px", "height": "490px"})
                ]

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger", style={"color":"#212121"}),
            html.Hr(style={"color":"#212121"}),
            html.P(f"The pathname {pathname} was not recognised.", style={"color":"#212121"}),
        ]
    )

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

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='darkgray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='darkgray')

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

    #Genres Exploration Graph
#Subgenres output callback
@app.callback(
    Output('sub_genres', 'options'),
    Input('main_genres', 'value')
)
def set_subgenres_options(selected_genres):
    line_copy = top_songs.copy()

    line_copy['artist_genre'] = line_copy['artist_genre'].apply(lambda x : x.replace('[', ''))
    line_copy['artist_genre'] = line_copy['artist_genre'].apply(lambda x : x.replace(']', ''))
    line_copy['artist_genre'] = line_copy['artist_genre'].apply(lambda x : x.replace("'", ''))
    line_copy['artist_genre'] = line_copy['artist_genre'].apply(lambda x : list(map(str.strip, x.split(','))))

    
    #Create subgenres dict for this main genre    
    subgenres_options = dict()
    for selected_genre in selected_genres:
        for genres in line_copy['artist_genre']:
            for genre in genres:
                if selected_genre in genre:
                    #Special case for 'rap' selection
                    if (selected_genre == 'rap') & ('trap' in genre):
                        continue

                    #Add the subgenre to the dict
                    subgenres_options[genre] = genre
    
    return [{'label': i, 'value': i} for i in subgenres_options.keys()]

@app.callback(
    Output('sub_genres', 'value'),
    Input('sub_genres', 'options')
)
def set_subgenres_value(available_options):
    return [available_options[0]['value']]

@app.callback(
    Output('genres_explore_fig', 'figure'),
    Input('sub_genres', 'value'),
    Input('feature_checklist', 'value')
)
def graph_genre_features(selected_subgenres, genre_feature):
    #Takes in feature to graph
    output = pd.DataFrame()
    line_copy = top_songs.copy()

    for genre in selected_subgenres:
        #make line plot data
        indices = [x for x in line_copy['artist_genre'].index if genre in line_copy['artist_genre'][x]]

        df = line_copy.iloc[indices]
        df = df.groupby('year').mean().reset_index()
        df['subgenre'] = genre
        output = pd.concat([output, df])
        
        
    fig = px.line(output, x='year', y=genre_feature, color = 'subgenre')
    
    fig.update_layout({
    'plot_bgcolor': ' #212121',
    'paper_bgcolor': ' #212121',
    'font_color': 'white',
    })

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    
    return fig

@app.callback(
    Output('classification-scatterplot', 'figure'),
    Input('first-genre', 'value'),
    Input('second-genre', 'value'),
    Input('first-feature', 'value'),
    Input('second-feature', 'value')
)
def create_data_genre(first_genre, second_genre, first_feature, second_feature):
    line_copy = top_songs.copy()

    #Genres to graph
    selected_genres = [first_genre, second_genre]
    features = [first_feature, second_feature]
    output = pd.DataFrame()

    main_genres_options = ['r&b', 'hip hop', 'country', 'rock', 'metal', 'edm', 'indie', 'pop']

    for genre in main_genres_options:
        #make line plot data
        indices = [x for x in line_copy['artist_genre'].index if genre in line_copy['artist_genre'][x]]

        df = line_copy.loc[indices]
        #df = df.groupby('year').mean().reset_index()
        df['genre'] = genre
        output = pd.concat([output, df])
    
    #Filter to only use data from selected genres
    output = output.loc[(output['genre'] == selected_genres[0]) | (output['genre'] == selected_genres[1])].reset_index(drop=True)
    
    #Drop the duplicate songs
    output = output.iloc[output['songs_id'].drop_duplicates().index]

    #Select relevant columns and features chosen
    columns = ['genre', 'songs_id', 'songs_name', 'artist_name', 'year'] + features
    
    #DataFrame of the selected parameters
    df = output[columns]
    
    #Label and feature df
    X = df.iloc[:,-2:] #Get the two selected features
    y = df['genre'] #Label is the genre

    #Scale the quantitative variables to be between 0 and 1
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


    #Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y.astype(str), test_size=0.20, random_state=0)

    # Fit the model on training data, predict on test data
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]

    cols = X_test.columns
    y_cols = y_test.unique()
    fig = px.scatter(
        X_test, x=cols[0], y=cols[1],
        color=y_score, color_continuous_scale='RdBu',
        symbol=y_test, symbol_map={y_cols[0]: 'circle-dot', y_cols[1]: 'square-dot'},
        labels={'symbol': 'genre', 'color': 'Prediction Score: 0 ({0}) - 1 ({1})'.format(y_cols[0], y_cols[1])},
        title='Classifier Results {0} and {1} ({2},{3})'.format(y_cols[0], y_cols[1], cols[0], cols[1])
    )
    fig.update_traces(marker_size=11, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')

    return fig

@app.callback(
    Output('matrix', 'figure'),
    Input('graph-type', 'value') 
)
def graph_correlation_cov_matrix(graph_type):
    line_copy = top_songs.copy()

    output = pd.DataFrame()

    main_genres_options = ['r&b', 'hip hop', 'country', 'rock', 'metal', 'edm', 'indie', 'pop']

    for genre in main_genres_options:
        #make line plot data
        indices = [x for x in line_copy['artist_genre'].index if genre in line_copy['artist_genre'][x]]

        df = line_copy.loc[indices]
        #df = df.groupby('year').mean().reset_index()
        df['genre'] = genre
        output = pd.concat([output, df])
    
    #Filter to only use data from selected genres
    output = output.reset_index(drop=True)
    
    #Drop the duplicate songs
    output = output.iloc[output['songs_id'].drop_duplicates().index]
    df = output.iloc[:,6:-2] #select only the audio features

    #Correlation matrix
    corr_df = df.corr()

    if graph_type == 'correlation':
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        rLT = corr_df.mask(mask)

        heat = go.Heatmap(
            z = rLT,
            x = rLT.columns.values,
            y = rLT.columns.values,
            zmin = - 0.5, # Sets the lower bound of the color domain
            zmax = 0.7,
            xgap = 1, # Sets the horizontal gap (in pixels) between bricks
            ygap = 1,
            colorscale = 'RdBu'
        )

        title = 'Correlation Matrix of the Audio Features'

        layout = go.Layout(
            title_text=title, 
            title_x=0.5, 
            width=600, 
            height=600,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed'
        )

        fig=go.Figure(data=[heat], layout=layout)

    # elif graph_type == 'covariance':
    #     cov_matrix = np.cov(corr_df.to_numpy(), bias=True)
    #     cov_df = pd.DataFrame(cov_matrix, index = df.columns, columns = df.columns)

    #     rLT = cov_df

    #     heat = go.Heatmap(
    #         z = rLT,
    #         x = rLT.columns.values,
    #         y = rLT.columns.values,
    #         zmin = - 0.2, # Sets the lower bound of the color domain
    #         zmax = 0.2,
    #         xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    #         ygap = 1,
    #         colorscale = 'RdBu'
    #     )

    #     title = 'Covariance Matrix of Audio Features'

    #     layout = go.Layout(
    #         title_text=title, 
    #         title_x=0.5, 
    #         width=600, 
    #         height=600,
    #         xaxis_showgrid=False,
    #         yaxis_showgrid=False,
    #         yaxis_autorange='reversed'
    #     )

    #     fig=go.Figure(data=[heat], layout=layout)

    return fig
# -------------------------------------------------------------------------------------
# RUN THE APP 
if __name__ == '__main__':
    app.run_server(debug=True)