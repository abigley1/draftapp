#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:10:36 2018

@author: bigley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:29:17 2018

@author: abigley
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import nba_py 
#from nba_py import player
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.manifold import t_sne
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
import numpy as np
from sklearn.neighbors import NearestNeighbors
import base64
import dash_table
from flask import Flask
from flask_cors import CORS

def make_tsne(data, inital_player):
    tsne = t_sne.TSNE(n_components = 2, learning_rate = 750)
    X_std = StandardScaler().fit_transform(data)
    fit  = tsne.fit_transform(X_std)

    trace1 = go.Scatter(
        showlegend=False,
        x=fit[:,0],
        y=fit[:,1],
        text = data.index.get_level_values(0).values,
        hoverinfo = 'text',
        mode='markers',
        marker=dict(
            size=12,
                color='#253046',
            )

        )
    )
    
    num = data.index.get_loc(inital_player)
    trace2 = go.Scatter(
        name = inital_player,
        x=[fit[num, 0]],
        y=[fit[num, 1]],
        text = inital_player,
        hoverinfo = 'text',
        mode='markers',
        marker=dict(
            size=14,
                color='#B35E3B',
            )

        )
    )

    data1 = [trace1, trace2]
    layout = go.Layout({'hovermode':'closest', 
                        'margin':{'t':0, 'r':0, 'l':0, 'b':0},
                        'legend':{'x':0, 'y':1},
                       'paper_bgcolor':'#F8F3F1',
                       'plot_bgcolor':'#F8F3F1'})
    fig = go.Figure(data = data1, layout=layout)
    return fig

def nba_dot_plot(stats, player, col,  data_type='Percentile'):
    cols = col
    if data_type == 'Percentile':
        s = stats[cols].rank(pct=True).loc[player]
        s=round(s,2)
        s = s*100
    else:
        s = stats[cols].loc[player]
    
    if len(s.index) == 1:
        data1=[]
        trace2 = go.Bar(y= s.iloc[0], 
              x= cols,
              marker= {"color": '#bdd7e7'},  
              name = stats['Year at School'].iloc[0])
        data1 = [trace2]
    if len(s.index) == 2: 
        trace1 = {"y": s.iloc[0], 
              "x": cols, 
              "marker": {"color": "#bdd7e7"}, 
               
              "name":stats['Year at School'].loc[player].iloc[0], 
              "type": "bar", 
                 }
        trace2 = {"y": s.iloc[1], 
              "x": cols, 
              "marker": {"color": "#6baed6"}, 
               
              "name":stats['Year at School'].loc[player].iloc[1], 
              "type": "bar", 
                 }
        data1 = [trace1, trace2]
    if len(s.index) == 3: 
        trace1 = {"y": s.iloc[0], 
              "x": cols, 
              "marker": {"color": "#bdd7e7"}, 
               
              "name":stats['Year at School'].loc[player].iloc[0], 
              "type": "bar", 
                 }
        trace2 = {"y": s.iloc[1], 
              "x": cols, 
              "marker": {"color": "#6baed6"}, 
               
              "name":stats['Year at School'].loc[player].iloc[1], 
              "type": "bar", 
                 }
        trace3 = {"y": s.iloc[2], 
              "x": cols, 
              "marker": {"color": "#3182bd"}, 
               
              "name":stats['Year at School'].loc[player].iloc[2], 
              "type": "bar", 
              "hovertext":stats[cols].loc[player].values
                 }
        data1=[trace1, trace2, trace3]
    if len(s.index) == 4:
        trace1 = {"y": s.iloc[0], 
          "x": cols, 
          "marker":{"color": "#bdd7e7"}, 
           
          "name":stats['Year at School'].loc[player].iloc[0], 
          "type": "bar", 
             }
        trace2 = {"y": s.iloc[1], 
          "x": cols, 
          "marker": {"color": "#6baed6"}, 
           
          "name":stats['Year at School'].loc[player].iloc[1], 
          "type": "bar", 
             }
        trace3 = {"y": s.iloc[2], 
          "x": cols, 
          "marker": {"color": "#3182bd"}, 
           
          "name":stats['Year at School'].loc[player].iloc[2], 
          "type": "bar", 
             }
        trace4 = {"y": s.iloc[3], 
          "x": cols, 
          "marker": {"color": "#08519c"}, 
           
          "name":stats['Year at School'].loc[player].iloc[3], 
          "type": "bar", 
             }
        data1 = [trace1, trace2, trace3, trace4]
    if data_type == 'Percentile':
        
        layout = {"xaxis": {"title":"College Statsitic"},
              "yaxis": {"title": "Percentile Rank"},
              'margin': {'t':25, 'r':0},
              'hovermode':'closest',
              'barmode':'group',
              'paper_bgcolor':'#F8F3F1',
              'plot_bgcolor':'#F8F3F1'
                 }
    else:
        layout = { "xaxis": {"title":"College Statsitic"},
              "yaxis": {"title": "Count Per Game"}, 
              'margin': {'t':25, 'r':0},
              'hovermode':'closest',
              'barmode':'group',
              'paper_bgcolor':'#F8F3F1',
              'plot_bgcolor':'#F8F3F1'
                 }

    fig = go.Figure(data=data1, layout=layout )
    return fig
 

def get_surv_curv(data, player):  ##add percentile of prediction as an annottion on the graph
    cph = CoxPHFitter()
    cph.fit(data, 'NBA_Experience', event_col='active')
    X = data.loc[[player]].drop(['NBA_Experience', 'active'], axis = 1)
    league_surv = cph.baseline_survival_
    player_surv = cph.predict_survival_function(X)
    x = data.drop(['NBA_Experience', 'active'], axis = 1)
    predictions  = cph.predict_expectation(x)
    percentiles = predictions.rank(pct=True)
    player_pct = percentiles.loc[player]
    string='Career Length Prediction Percentile: ' +str(round(player_pct.values[0], 2))
    
    trace1 = go.Scatter(
        name = 'League Average',
        x=league_surv.index,
        y=league_surv['baseline survival'].values,
        marker = {'color':"#253046"}
            )
    trace2 = go.Scatter(
        name = player,
        x=player_surv.index,
        y=player_surv[player].values,
        marker={'color':'#B35E3B'}
            )

    data = [trace1, trace2]
    layout = go.Layout({ 
          "xaxis": {"title": "Years in the NBA", }, 
          "yaxis": {"title": "Probability of remaining in the NBA"},
          'paper_bgcolor':'#F8F3F1',
           'plot_bgcolor':'#F8F3F1',
          'margin': {'t':50, 'r':30},
          'annotations':[{'x':13, 'y':0.78, 'text':string, 'showarrow':False, 'font':{'size':14}}],
          'legend':{'x':.8, 'y':1, 'traceorder':'normal'} })
    
    fig = go.Figure(data=data, layout=layout)

    

    

    return fig

def get_player_sum(data, player):
    cols = ['Pos', 'Hgt', 'Wght', 'Birthday (date format)','NCAA Seasons\n(D-I)', 'Wingspan', 'RSCI Rank', 'School' ]
    summary = {}
    summary['Name'] = player
    for col in cols:
        summary[col] = data.loc[player, col]
    return summary


def get_similar_players(data, player): #must remove active column from dataset if not already

    kn = NearestNeighbors(n_neighbors=5)

    stand = StandardScaler()
    
    scaled =stand.fit_transform(data)
    
    scaled_player = stand.transform(data.loc[[player]])
    
    kn.fit(scaled)#.rank(pct=True))
    
    x = kn.kneighbors(scaled_player, 6)
    return(data.iloc[x[1][0]].index.values)
    
def get_players(data, year):
    lis=list()
    dic=list()
    for name in data.index.droplevel(1):
        comp = data['Year'].loc[name].max()
        if comp == year:
            lis.append(name)
    lis = list(set(lis))
    for name in lis:
        g= dict(label = name, value=name)
        dic.append(g)
    return dic
        
def get_years(data):
    years = list(set(data['Year'].values))
    y =list()
    for year in sorted(years):
        d = dict(label= year, value=year)
        y.append(d)
    return y
def create_summary(player):
    summ = get_player_sum(stats_for_summary, player)
    encoded_image = get_photo(player)
   
    HTML = html.Div([
                                    html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode())
                                            , style={ 'display':'block','width':'35%', 'margin-left':'auto','margin-right':'auto','margin-top':'4%' }),
                                    html.H3(summ['Name'],
                                            style={'text-align':'center'},),
                                    html.Div(
                                            [
                                    html.Summary('Birthday: ' + summ['Birthday (date format)'],
                                            style={'text-align':'center',}),                    
                                    html.Summary('Position: ' + summ['Pos'],
                                            style={'text-align':'center'}),
                                    html.Summary('Height: ' + summ['Hgt'],
                                            style={'text-align':'center'}),
                                    html.Summary('Weight: ' + str(int(summ['Wght'])),
                                            style={'text-align':'center'}),
                                    html.Summary('Wingspan: ' + str(round(summ['Wingspan'],2)),
                                            style={'text-align':'center'}),
                                    html.Summary('School: ' + str(summ['School']),
                                            style={'text-align':'center'}),
                                    html.Summary('NCAA Seasons: ' + str(summ['NCAA Seasons\n(D-I)']),
                                            style={'text-align':'center'}),
                                    html.Summary('High School Rank: ' + str(int(summ['RSCI Rank'])),
                                            style={'text-align':'center'})])    
                                                    ],style={'font-size':'medium'})

    return HTML

def create_similar(player):
     similar = get_similar_players(stats_for_similar, player)
     links = get_links(similar)
     
     HTML = html.Div([   
                            html.H2('Similar Players',
                                    style={'text-align':'center'}),
                            html.H3(links[1],
                                    style={'text-align':'center'}),
                            html.H3(links[2],
                                    style={'text-align':'center'}),
                            html.H3(links[3],
                                    style={'text-align':'center'}),
                            html.H3(links[4],
                                    style={'text-align':'center'}),
                            html.H3(links[5],
                                    style={'text-align':'center'})])
     return HTML

def get_links(similar_players):
    lis = list()
    link = 'https://basketball.realgm.com'
    stats_for_links = stats_for_summary.copy()
    stats_for_links.index=stats_for_links.index.str.replace('.', '')
    stats_for_links.index=stats_for_links.index.str.replace("'", '')
    for name in similar_players:
        suffix = stats_for_links['Player Link'].loc[name]
        url = link+suffix
        HTML = html.A(name, href=url, target='_blank')
        lis.append(HTML)
    return lis
def get_cols(data):
    drop = ['Player Link', 'RealGM Summary Page', 'Highest Level Reached',
            'Season', 'School','League', 'Conference', 'TeamID', 'Year', 
            'Year at School', 'Total S %', 'Hght\n(inches)', 'Wght',
       'RSCI Rank', 'RealGM Link', 'NCAA Seasons\n(D-I)', 'Hgt', 'Birthday (date format)'
       ,'active','NBA_Experience', 'Pos', 'Wingspan']
    data1 = data.drop(drop, axis=1)
    cols = data1.columns
    lis = list()
    for col in cols:
        d = dict(label=col, value=col)
        lis.append(d)
    return lis


def get_photo(player):
    name=''
    url = ''
    low = str.lower(player).replace('-', '').replace('.', '').split(' ')
    if len(low[1]) > 5:
        last = low[1][0:5]
    else:
        last=low[1]
    first =low[0][0:2]
    print(first, last)
    name =  last+first+'01.jpg'
    image_filename = 'img_files/'
    url = image_filename +name
    encoded_image = base64.b64encode(open(url, 'rb').read())
    return encoded_image

def create_nba_stats_table(similar, nba_stats, dtype):
    per_game_cols = ['active', 'NBA Exp', 'MIN', 'FGM', 'FGA', 'FG%',
       '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OFF', 'DEF', 'TRB', 'AST',
       'STL', 'BLK', 'PF', 'TOV', 'PTS']
    
    totals_cols = ['active', 'NBA Exp', 'MIN.1', 'FGM.1',
       'FGA.1', 'FG%.1', '3PM.1', '3PA.1', '3P%.1', 'FTM.1', 'FTA.1', 'FT%.1',
       'OFF.1', 'DEF.1', 'TRB.1', 'AST.1', 'STL.1', 'BLK.1', 'PF.1', 'TOV.1',
       'PTS.1']
    
    advanced_cols = ['active', 'NBA Exp', 'Dbl Dbl', 'Tpl Dbl', '40 Pts', '20 Reb',
       '20 Ast', 'Techs', 'HOB', 'Ast/TO', 'Stl/TO', 'FT/FGA', "W's", "L's",
       'Win %', 'OWS', 'DWS', 'WS', 'TS%', 'eFG%', 'ORB%',
       'DRB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%', 'Total S %',
       'PPR', 'PPS', 'ORtg', 'DRtg', 'PER']
    
    
    non_avg_col =[ 'RealGM Summary Page', 'Highest Level Reached' , 'Year', 'Team', 'TeamID', 
                 'Year.1', 'Abbr']
    
    
    link= nba_stats['RealGM Summary Page'].str.split('/', expand=True)
    name = link.loc[:,2]
    name1 = name.str.split('-', expand=True)
    namef = name1[0] + ' ' +name1[1]
    
    nba_stats['Name'] = namef
    nba_stats = nba_stats.set_index('Name')
    
    
    placeholder = nba_stats.loc[similar].copy()
    placeholder.index = placeholder.index.str.replace('.', '')
    placeholder.index = placeholder.index.str.replace("'", '')
    placeholder.index = placeholder.index.str.replace(' Jr', '')
    placeholder= placeholder[~placeholder.index.duplicated(keep='first')]
    placeholder = placeholder[non_avg_col]
    
    df = pd.DataFrame()
    
    
    for name in similar:
        
        name = name.replace('.', '')
        name = name.replace("'", '')
        name = name.replace(' Jr', '')
        name1=name
        if name == 'Darius Johnson Odom': #deal with werid name cuts
            name1 = name1.split('-')[0]
        if name =='Al-Farouq Aminu':
            name1 = 'Al Farouq'
        
        smaller_df = nba_stats.loc[name1] 
        if len(smaller_df) == 85:
            career_len = 1
        else:
            career_len = len(smaller_df)
        
        temp = smaller_df.copy()
        temp = temp[~temp.index.duplicated(keep='last')]
        active = temp['Year.1'] ==2018
    
        smaller_df['NBA Exp'] = career_len
        smaller_df['active'] = active
        
        if type(smaller_df) == pd.Series:
            avg= smaller_df.drop(non_avg_col)
        else:
            avg = smaller_df.drop(non_avg_col, axis=1).mean()
            avg = avg.apply(lambda x: round(x, 2))
            avg = avg.append(placeholder.loc[name])
            avg.name=name
        df =df.append(avg)
    if dtype == 'per_game':
        df = df[per_game_cols]  
    if dtype == 'Advanced':
        df = df[advanced_cols]
    if dtype =='totals':
        df = df[totals_cols] 
    df= df.reset_index()
    x =dash_table.DataTable(
            id='name',
            columns=[{'name':i, "id":i} for i in df.columns],
            data = df.to_dict('rows'))
    return x
   
    #py.iplot(fig, filename='simple-3d-scatter')
######how = 'best' takes the single highest value of a stat from all of a players seasons
##### how = 'weighted' takes a weighted average of all of the players seasons
#####  if left blank, will just take the players most recent season





stats = pd.read_csv('data/app_data.csv')
stats1 = stats.copy()
stats1['PlayerName'] = stats1['PlayerName'].str.replace('.', '')
stats= stats.set_index(['PlayerName', 'Full Year'])






nba_stats = pd.read_excel('data/NBARealGM.xlsx', header=1)



stats_for_summary = stats.copy()
stats_for_summary.index =stats_for_summary.index.droplevel(1)
stats_for_summary= stats_for_summary[~stats_for_summary.index.duplicated(keep='first')]
stats= stats[~stats.index.duplicated(keep='first')]
drop = ['Player Link', 'Adjusted BPM',
       'Estimated OBPM\n(Calculated BPM - Regressed DBPM)',
       'Estimated DBPM\n(@OdeToOden\nRegression)',
       'RealGM Summary Page', 'Highest Level Reached', 'Season', 'School',
       'League', 'Conference', 'TeamID', 'Year', 'Year at School', 'GP', 'GS',
       'MIN', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%',
       'OFF', 'DEF', 'TRB', 'AST', 'STL', 'BLK', 'PF', 'TOV', 'PTS', 'GP_t','40 Pts', '20 Reb', '20 Ast', 'Techs', 'HOB', 'Ast/TO', 'Stl/TO',
       'FT/FGA', 'W', 'Ls', 'Win %', 'OWS', 'DWS', 'WS', 'GP.3', 'TS%', 'eFG%',
       'ORB%', 'DRB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%',
       'Total S %', 'PPR', 'PPS', 'ORtg', 'DRtg', 'PER',
       'Wingspan', 'Pos', 'Hght\n(inches)', 'Wght',
       'RSCI Rank', 'RealGM Link', 'NCAA Seasons\n(D-I)', 'Hgt', 'Birthday (date format)']
col_touse = ['NBA_Experience', 'active','OWS', 'Ast/TO', 'TOV%', 'TS%', 'STL%', 'TRB%'] 

cols_for_sim = [ 'FGA/3A','TS%', 'AST%', 'ORB%', 'DRB%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'Hght\n(inches)', 'Wingspan', 'Wght']


ss = stats[col_touse]
stats_for_surv = stats_for_summary[col_touse]
stats_for_tsne = stats_for_surv.drop(['NBA_Experience', 'active'], axis=1)


stats_for_similar = stats1.copy()
stats_for_similar= stats_for_similar[~stats_for_similar.index.duplicated(keep='first')]
stats_for_similar['FGA/3A'] = stats_for_similar['FGA']/stats_for_similar['3PA']

stats_for_similar = stats_for_similar[cols_for_sim]

stats_for_similar = stats_for_similar.select_dtypes(['number'])


#fig = make_tsne(s, 'Stephen Curry')
#fig1 = make_tsne(s, 'Kevin Durant')


cph = CoxPHFitter()
cph.fit(stats_for_surv, 'NBA_Experience', event_col='active') #fit model once at the begining

tsne = t_sne.TSNE(n_components = 2, learning_rate = 750) #fit tsne at begining
X_std = StandardScaler().fit_transform(stats_for_tsne)
fit  = tsne.fit_transform(X_std)



kn = NearestNeighbors(n_neighbors=5)
drop = ['FG_pg', '2P_pg', '3P_pg', 'FT_pg']
stand = StandardScaler()
scaled =stand.fit_transform(stats_for_surv)
kn.fit(scaled)



stats_for_similar = stats_for_surv.copy()
stats_for_similar.index = stats_for_similar.index.str.replace('.', '')
stats_for_similar.index = stats_for_similar.index.str.replace("'", '')









app = dash.Dash(__name__, assets_external_path='https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css')
server = app.server
CORS(server)

app.scripts.config.serve_locally = True
app.css.config.serve_locally = False
player = 'Stephen Curry'
year = '2009'
yearOptions = get_years(stats)
playerOptions = get_players(stats, int(year))

tsnefig = make_tsne(stats_for_tsne, player)
summ = get_player_sum(stats_for_summary, player)
html_summary = create_summary(player)
html_similar = create_similar(player)
survfig = get_surv_curv(stats_for_surv, player)
similar = get_similar_players(stats_for_similar, player)
advancedOptions = [dict(label='Percentile', value = 'Percentile'),dict(label='Raw', value='Raw')]
advancedOptionsSim = [dict(label='Per Game', value = 'per_game'),dict(label='Advanced', value='Advanced'), dict(label='Totals', value='totals')]

colOptions = get_cols(stats)
colValues =col_touse
colValues.pop(0)
colValues.pop(0)
dotfig = nba_dot_plot(stats, player, col=colValues)



table = create_nba_stats_table(similar, nba_stats, 'per_game')






app.layout= html.Div([html.Div(
        [       
                html.Div(#title
                        [
                        html.H1('NBA Survival Analysis',
                        style={'text-align':'center', 'margin-top':'0px', 'margin-bottom':'0px'})],
                                            ),
                                
                html.Div(#drop downs
                        [
                                html.Div(
                                        [
                                            html.H3('Draft Class',
                                        style={'text-align':'center', 'margin-top':'0px', 'margin-bottom':'0px'}),
                                       
                                        dcc.Dropdown(id='year dropdown',
                                             options = yearOptions,
                                             value=year,
                                             ),                
                                                        
                                           ], style={'width':'49%','display':'inline-block'}),             
                                html.Div(
                                        [ 
                                            html.H3('Player Name',
                                        style={'text-align':'center', 'margin-top':'0px', 'margin-bottom':'0px'}),
                                                    
                                dcc.Dropdown(id='player dropdown',
                                             options = playerOptions,
                                             value=player,
                                             searchable = True,
                                             
                                             ),
                                        
                                ], style={'width':'49%', 'display':'inline-block'} )], ),
                html.Div(
                        [
                                
                        html.Div(#player summary
                                [
                                        html.Div([
                                           html_summary,
                                           
                                                ], id='player summary')],
                                style={'width':'19%', 'display':'inline-block', 'box-shadow':'1.5px 1.5px 5px 1px rgba(0,0,0,.2)'
                                       , 'background-color':'#B35E3B', 'margin-top':'8px', 'margin-right':'10px', 'vertical-align':'top',
                                       'height':'450px', 'border-radius':'10px'} ),
                        html.Div(
                                [   dcc.Graph(figure=survfig,
                                              id='survcurv',
                                      config={'displayModeBar': False}  )    
                                        ], style={'width':'80%', 'box-shadow':'1.5px 1.5px 5px 1px rgba(0,0,0,.2)','display':'inline-block', 'margin-top':'8px', 'margin-bottom':'10px', 'border-radius':'10px'} , )]
                                ),                 
                
                html.Div(
                        [   
    
                            html.Div(
                                    [
                                        dcc.RadioItems(id='data type',
                                         options = advancedOptions,
                                         value = 'Percentile',
                                        ),
                                                       
                                        dcc.Dropdown(id='col selector',
                                             options = colOptions,
                                             value=colValues,
                                             multi=True,
                                             
                                             ),   
                                            ], style={'width':'19%', 'display':'inline-block',
                                                    'vertical-align':'top', 'margin-top':'3%','margin-right':'10px'}),

                            
        
                            dcc.Graph(figure=dotfig,
                                      id='dotplot',
                                      style={'width':'80%', 'display':'inline-block', 'margin-bottom':'10px', 'box-shadow':'1.5px 1.5px 5px 1px rgba(0,0,0,.2)'},
                                      config={'displayModeBar': False}),    
                                ], style={'display':'inlinle-block'}),
                
                
                
                html.Div(
                        [
                            html.Div(
                                    [
                                html.Div(
                                        [html_similar], id='similar', style={'display':'inline-block', 'background-color':'#B35E3B', 'width':'19%', 'vertical-align':'top', 'margin-right':'10px' ,'margin-top':'75px', 'box-shadow':'1.5px 1.5px 5px 1px rgba(0,0,0,.2)',  'border-radius':'10px'}),
                                dcc.Graph(figure=tsnefig, 
                                      id='tsneplot', style={'display':'inline-block', 'width':'80%', 'box-shadow':'1.5px 1.5px 5px 1px rgba(0,0,0,.2)'}),
                                          
                                
                               
                                            ])
                            
                        ]
                        ),
                html.Div(
                        [html.H3('Similar Players NBA Career Averages', style={'text-align':'center'}),
                                dcc.RadioItems(id='sim data type',
                                     options = advancedOptionsSim,
                                     value = 'per_game',
                                        )]),
                html.Div(
                        [ 
                                table
                                ], id='similar_table')
                            
                
                                    
        ],  style = {'max-width':1250, 'margin':'auto'})],  style={ 'background-color':'#253046', 'color':'#F8F3F1'}  )
    
@app.callback(
        dash.dependencies.Output('player dropdown', 'options'),
        [dash.dependencies.Input('year dropdown', 'value')])
def update_players(year):
    players = get_players(stats, year)
    return players
@app.callback(
        dash.dependencies.Output('player dropdown', 'value'),
        [dash.dependencies.Input('year dropdown', 'value')])
def update_player_menu(year):
    players = get_players(stats, year)
    return players[0]







@app.callback( ### how to get everything to update properly when I update year?
        dash.dependencies.Output('player summary', 'children'),
        [dash.dependencies.Input('player dropdown', 'value')])
def update_summary(player):
    new = create_summary(player)
    return new

@app.callback(
        dash.dependencies.Output('survcurv', 'figure'),
        [dash.dependencies.Input('player dropdown', 'value')])
def get_surv_curv(player):
    X = stats_for_surv.loc[[player]].drop(['NBA_Experience', 'active'], axis = 1)
    league_surv = cph.baseline_survival_
    player_surv = cph.predict_survival_function(X)
    x = stats_for_surv.drop(['NBA_Experience', 'active'], axis = 1)
    predictions  = cph.predict_expectation(x)
    percentiles = predictions.rank(pct=True)
    player_pct = percentiles.loc[player]
    string='Career Length Prediction Percentile: ' +str(round(player_pct.values[0], 2))
    
    trace1 = go.Scatter(
        name = 'League Average',
        x=league_surv.index,
        y=league_surv['baseline survival'].values,
        marker = {'color':"#253046"}
            )
    trace2 = go.Scatter(
        name = player,
        x=player_surv.index,
        y=player_surv[player].values
            )

    data = [trace1, trace2]
    layout = go.Layout({
          "xaxis": {"title": "Years in the NBA", }, 
          "yaxis": {"title": "Probability of remaining in the NBA"},
           'margin': {'t':50, 'r':30},
          'annotations':[{'x':13, 'y':0.78, 'text':string, 'showarrow':False, 'font':{'size':14}}],
          'legend':{'x':.8, 'y':1, 'traceorder':'normal'},
           'paper_bgcolor':'#F3F2F4',
          'plot_bgcolor':'#F3F2F4'
           
             })
    
    fig = go.Figure(data=data, layout=layout)

    return fig

@app.callback(
        dash.dependencies.Output('tsneplot', 'figure'),
        [dash.dependencies.Input('player dropdown', 'value')])
def update_tsne(player):
    trace1 = go.Scatter(

        x=fit[:,0],
        y=fit[:,1],
        showlegend=False,
        text = stats_for_tsne.index.get_level_values(0).values,
        hoverinfo = 'text',
        mode='markers',
        marker=dict(
            size=12,
            color='#253046'
            )

        )
    )
    num = stats_for_tsne.index.get_loc(player)
    trace2 = go.Scatter(
        name = player,
        x=[fit[num, 0]],
        y=[fit[num, 1]],
        text = player,
        hoverinfo = 'text',
        mode='markers',
        marker=dict(
            size=14,
                color='#B35E3B',
               
            )

        )
    )
            

    data1 = [trace1, trace2]
    layout = go.Layout({'hovermode':'closest', 
                        'margin':{'t':0, 'r':0, 'l':0, 'b':0},
                       'legend':{'x':0, 'y':1},
                       'paper_bgcolor':'#F8F3F1',
                       'plot_bgcolor':'#F8F3F1'})
    fig = go.Figure(data = data1, layout=layout)
    return fig
    
@app.callback(
        dash.dependencies.Output('dotplot', 'figure'),
        [dash.dependencies.Input('player dropdown', 'value'),
         dash.dependencies.Input('data type', 'value'),
         dash.dependencies.Input('col selector', 'value')])
def update_dotplot(player, data_type, col):
    cols = col
    if data_type == 'Percentile':
        s = stats[cols].rank(pct=True).loc[player]
        s=round(s,2)
        s = s*100
    else:
        s = stats[cols].loc[player]
    
    if len(s.index) == 1:
        data1=[]
        trace2 = go.Bar(y= s.iloc[0], 
              x= cols, 
              marker= {"color": '#bdd7e7'}, 
              name = stats['Year at School'].iloc[0])
        data1 = [trace2]
    if len(s.index) == 2: 
        trace1 = {"y": s.iloc[0], 
              "x": cols, 
              "marker": {"color": "#bdd7e7"}, 
              
              "name":stats['Year at School'].loc[player].iloc[0], 
              "type": "bar", 
                 }
        trace2 = {"y": s.iloc[1], 
              "x": cols, 
              "marker": {"color": "#6baed6"}, 
               
              "name":stats['Year at School'].loc[player].iloc[1], 
              "type": "bar", 
                 }
        data1 = [trace1, trace2]
    if len(s.index) == 3: 
        trace1 = {"y": s.iloc[0], 
              "x": cols, 
              "marker": {"color": "#bdd7e7"}, 
               
              "name":stats['Year at School'].loc[player].iloc[0], 
              "type": "bar", 
                 }
        trace2 = {"y": s.iloc[1], 
              "x": cols, 
              "marker": {"color": "#6baed6"}, 
               
              "name":stats['Year at School'].loc[player].iloc[1], 
              "type": "bar", 
                 }
        trace3 = {"y": s.iloc[2], 
              "x": cols, 
              "marker": {"color": "#3182bd"}, 
               
              "name":stats['Year at School'].loc[player].iloc[2], 
              "type": "bar", 
              "hovertext":stats[cols].loc[player].values
                 }
        data1=[trace1, trace2, trace3]
    if len(s.index) == 4:
        trace1 = {"y": s.iloc[0], 
          "x": cols, 
          "marker":{"color": "#bdd7e7"}, 
           
          "name":stats['Year at School'].loc[player].iloc[0], 
          "type": "bar", 
             }
        trace2 = {"y": s.iloc[1], 
          "x": cols, 
          "marker": {"color": "#6baed6"}, 
           
          "name":stats['Year at School'].loc[player].iloc[1], 
          "type": "bar", 
             }
        trace3 = {"y": s.iloc[2], 
          "x": cols, 
          "marker": {"color": "#3182bd"}, 
           
          "name":stats['Year at School'].loc[player].iloc[2], 
          "type": "bar", 
             }
        trace4 = {"y": s.iloc[3], 
          "x": cols, 
          "marker": {"color": "#08519c"}, 
           
          "name":stats['Year at School'].loc[player].iloc[3], 
          "type": "bar", 
             }
        data1 = [trace1, trace2, trace3, trace4]
    if data_type == 'Percentile':
        
        layout = { "xaxis": {"title":"College Statsitic"},
              "yaxis": {"title": "Percentile"},
              'margin': {'t':25, 'r':0},
              'hovermode':'closest',
              'barmode':'group',
                  'paper_bgcolor':'#F8F3F1',
              'plot_bgcolor':'#F8F3F1'
                 }
    else:
        layout = { "xaxis": {"title":"College Statsitic"},
              "yaxis": {"title": "Count per Game"},
              'margin': {'t':25, 'r':0},
              'hovermode':'closest',
              'barmode':'group',
                  'paper_bgcolor':'#F8F3F1',
              'plot_bgcolor':'#F8F3F1'
                 }

    fig = go.Figure(data=data1, layout=layout )
    return fig

    
@app.callback(
    dash.dependencies.Output('similar', 'children'),
    [dash.dependencies.Input('player dropdown', 'value')])
def update_similar(player): 
    sim = create_similar(player)
    return sim
    
@app.callback(
        dash.dependencies.Output('similar_table', 'children'),
        [dash.dependencies.Input('player dropdown', 'value'),
         dash.dependencies.Input('sim data type', 'value')])
def update_sim_table(player,dtype):
    sim = get_similar_players(stats_for_surv, player)
    table = create_nba_stats_table(sim, nba_stats,dtype)
    return table




    
    
    
    
    
    
    
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)


















