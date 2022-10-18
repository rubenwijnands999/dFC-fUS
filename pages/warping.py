# -*- coding: utf-8 -*-
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
from data_import import Data
import plotly.graph_objects as go
from GUI_tools import *
import matplotlib
from app import app
import dash_bootstrap_components as dbc

num_regions = 4

warp_obj = GUI_Warp("Dummy")

# # load data
# TE = 1  # Default
#
# paths = util.create_paths_dict()
# MiceDict = util.read_csv_file(paths['csv_path'])
# ID = MiceDict['recID'][TE]
# sectionID = MiceDict['sectionID'][TE]
#
# cmap = matplotlib.cm.nipy_spectral
# color_list=[]
# for i in range(cmap.N):
#     rgba = cmap(i)
#     # rgb2hex accepts rgb or rgba
#     color_list.append(matplotlib.colors.rgb2hex(rgba))
#
# # IMPORT WARP
# norm_img = import_warp(ID,num_regions)
#
# ROI_list,flood_fill_color_table = util.create_region_color_list()

layout = html.Div([
    dbc.Container([
        dbc.Row(
            dbc.Col(html.H1("ROI selection", className="text-center"), className="mb-3 mt-3")
        ),
        dbc.Row(
            dbc.Col(html.Div(id='text_message_warp', className="text-center"), className="mb-1 mt-1")
        ),
        dbc.Row(
            html.Div([
                dcc.RadioItems(
                    id='ROI',
                    value="Anterior cingulate",
                    options=[
                        {'label': i, "value": j} for i,j in zip(warp_obj.ROI_list,range(len(warp_obj.ROI_list)))
                    ],
                    labelStyle={'display': 'block'},
                ),
                html.Button('Dilation', id='button_dilation'),
                html.Button('Erosion', id='button_erosion'),
                html.Button('Update colors', id='button_update'),
                html.Button('Save', id='button_save'),
                html.Div(id='text_message')
            ]),
        ),
        dbc.Row(
            html.Div(
                dcc.Graph(
                    id = 'roi_fig',
                    figure=px.imshow(warp_obj.norm_img,width=1000,height=1000,color_continuous_scale=warp_obj.color_list),
                    style={'margin': '0 auto', 'display': 'block', 'height': '1000px', 'width': '1000px'},
                ),
            ),
        )
    ])
])


@app.callback(Output('text_message', 'children'),
              [Input('button_save', 'n_clicks')])
def save_img(clicks):
    if clicks is not None:
        save_ROI_image(warp_obj.norm_img,warp_obj.ID,warp_obj.num_regions)
        return "Saved succesfully"

@app.callback(
    Output('roi_fig', 'figure'),
    [Input('roi_fig', 'clickData'),
    Input('button_dilation', 'n_clicks'),
    Input('button_erosion', 'n_clicks'),
    Input('button_update', 'n_clicks'),
    Input('text_message_warp', 'children'),
    State('ROI', 'value')])
def change_warp(xy,dil_but,ero_but,update,txt,color_value):
    ctx = dash.callback_context
    if ctx.triggered:
        button = ctx.triggered[0]['prop_id']

        if 'button_dilation' in button:
            return px.imshow(dilation(warp_obj.norm_img, warp_obj.flood_fill_color_table[color_value]),color_continuous_scale=warp_obj.color_list)
        elif 'button_erosion' in button:
            return px.imshow(erosion(warp_obj.norm_img, warp_obj.flood_fill_color_table[color_value]),color_continuous_scale=warp_obj.color_list)
        elif 'button_update' in button:
            return px.imshow(update_colors(warp_obj.norm_img),color_continuous_scale=warp_obj.color_list)
        elif xy is not None:
            floodfill(warp_obj.norm_img,xy,warp_obj.flood_fill_color_table[color_value])
            return px.imshow(warp_obj.norm_img, color_continuous_scale=warp_obj.color_list)
        else:
            return px.imshow(warp_obj.norm_img, width=1000, height=1000, color_continuous_scale=warp_obj.color_list)
    else:
        return px.imshow(warp_obj.norm_img,width=1000,height=1000,color_continuous_scale=warp_obj.color_list)

@app.callback(Output('text_message_warp', 'children'),
              [Input('mouse_select', 'value')])
def choose_mouse(value):
    print("Triggered")
    if value is not None:
        try:
            warp_obj.__init__(int(value))
            text = "Data set " + create_option_list()[int(value)]['label']
            return text
        except:
            pass
    else:
        return "Select a mouse"

@app.callback(Output('ROI', 'options'),
              [Input('text_message_warp', 'children')])
def update_options(text_input):
    return [{'label': i, "value": j} for i,j in zip(warp_obj.ROI_list,range(len(warp_obj.ROI_list)))]