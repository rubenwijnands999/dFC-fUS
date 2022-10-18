from dash import dcc
from dash import html
import dash
from dash.dependencies import Input, Output, State
import plotly.express as px
from app import app
from GUI_tools import *
import dash_bootstrap_components as dbc

# Dummy
GUI_obj = GUI_Data("Dummy")

layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("fUS data video", className="text-center"), className="mb-3 mt-3")
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='text_message2', className="text-center"), className="mb-1 mt-1")
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                    id = 'heatmap',
                    style={'margin': '0 auto', 'display': 'block', 'height': '500px', 'width': '500px'},
                    )
                )
            ]),
            dbc.Row([
                html.Div(
                    children=[
                        dbc.Button('Play', id='play', n_clicks=0,style={'border-radius':'5px','background-color':'black','color':'white','padding':'5px 10px','margin-right':'5px'}),
                        dbc.Button('Pause', id='pause', n_clicks=0,style={'border-radius':'5px','background-color':'black','color':'white','padding':'5px 10px','margin-right':'5px'}),
                        dbc.Button('Block', id='block', n_clicks=0,style={'border-radius':'5px','background-color':'black','color':'white','padding':'5px 10px','margin-right':'5px'}),
                    ],
                    style={'width':'20%','float':'left'}
                ),
                html.Div(
                    children=[
                        dcc.Interval(id='auto-stepper',interval=GUI_obj.refresh_time,n_intervals=0,disabled=True),
                        dcc.Slider(
                            id='frame_slider',
                            min=0,
                            max=GUI_obj.Nf-1,
                            value=0,
                            step=1,
                            marks=None,
                            tooltip={"placement":"top","always_visible":True}
                            ),
                    ],
                    style={'width': '60%', 'float': 'left'}
                )
            ])
        ])
])

# update the heatmap based on the value of the slider
@app.callback(Output("heatmap", "figure"),
    [Input("frame_slider", "value"),Input('text_message2', 'children')])
def update_figure(frame_index,mouse):
    if frame_index is not None:
        fig = px.imshow(np.log10(GUI_obj.GUI_data[frame_index][:][:]),
                        zmin=GUI_obj.c_min, zmax=GUI_obj.c_max,
                        width=500, height=500,color_continuous_scale='hot')
        return fig
    else:
        default_fig = px.imshow(np.log10(GUI_obj.GUI_data[0][:][:]).tolist(),
                        zmin=GUI_obj.c_min, zmax=GUI_obj.c_max,
                        width=500,height=500,color_continuous_scale='hot')

        return default_fig


@app.callback(Output("frame_slider", "value"),
        [Input("play", "n_clicks"),
        Input("pause", "n_clicks"),
        Input("block", "n_clicks"),
        Input("auto-stepper", "n_intervals"),
         Input('mouse_select', 'value')])
def update_slider(play_clicks, pause_clicks, block_clicks, nintervals,mouse):
    ctx = dash.callback_context
    if ctx.triggered:
        button = ctx.triggered[0]['prop_id']
        if 'block' in button:
            return 0 #start value
        elif 'pause' in button:
            return (nintervals+1)%GUI_obj.Nf
        elif 'play' in button:
            return (nintervals+1)%GUI_obj.Nf
        elif 'mouse_select' in button:
            return 0
        elif nintervals>0:
            return (nintervals+1)%GUI_obj.Nf

#Starts/pauses the 'interval' component, which starts/pauses
#the animation.
@app.callback(
    Output("auto-stepper","disabled"),
    [Input("play", "n_clicks"),
    Input("pause", "n_clicks"),
    Input("block", "n_clicks")],
    [State("auto-stepper","disabled")],
)
def pause_play(play_btn, pause_btn, block_btn, disabled):
    # Check which button was pressed
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    else:
        button = ctx.triggered[0]['prop_id']
        if 'play' in button:
            return False
        elif 'pause' in button:
            return True
        elif 'block' in button:
            return True
        else:
            return True

@app.callback(Output('text_message2', 'children'),
              Output("frame_slider", "max"),
              Output('auto-stepper','interval'),
              [Input('mouse_select', 'value')])
def choose_mouse(value):
    print("Triggered")
    if value is not None:
        try:
            GUI_obj.__init__(int(value))
            text = "Data set " + create_option_list()[int(value)]['label']
            return text,GUI_obj.Nf,GUI_obj.refresh_time
        except:
            pass
    else:
        return "Select a mouse",100,0.1