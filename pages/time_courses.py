from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from app import app
from GUI_tools import *
import dash_bootstrap_components as dbc

# load data
GUI_obj = GUI_Data("Dummy",time_course_data=True)

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("fUS time courses", className="text-center"), className="mb-3 mt-3")
        ]),
        dbc.Row([
            dbc.Col(html.Div(id='text_message_time', className="text-center"), className="mb-1 mt-1")
        ]),
        dbc.Row(
            dcc.Graph(
                id = 'meanfig',
                figure=px.imshow(GUI_obj.mean_log_fig,zmin=np.min(GUI_obj.mean_log_fig), zmax=np.max(GUI_obj.mean_log_fig),
                     width=500,height=500,color_continuous_scale='hot'),
                style={'margin': '0 auto', 'display': 'block', 'height': '500px', 'width': '500px'},
            )
        ),
        dbc.Row(
            dcc.Graph(id='time_course')
        )
    ])
])
@app.callback(
    Output('time_course', 'figure'),
    [Input('meanfig', 'clickData')])
def display_time_course(xy):
    if xy is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=GUI_obj.x_list, y=GUI_obj.GUI_data[xy['points'][0]['y'],xy['points'][0]['x'],:].tolist(), name="Raw"))
        fig.update_layout(title="Time course of pixel " + "(" + str(xy['points'][0]['x']) + "," + str(xy['points'][0]['y']) + ")",xaxis_title='Frame index',yaxis_title='Value')
        return fig
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=GUI_obj.x_list, y=GUI_obj.GUI_data[0,0,:].tolist(), name="Raw"))
        fig.update_layout(title="Time course of pixel (0,0)",xaxis_title='Frame index',yaxis_title='Value')
        return fig

@app.callback(Output('text_message_time', 'children'),
              [Input('mouse_select', 'value')])
def choose_mouse(value):
    print("Triggered")
    if value is not None:
        try:
            GUI_obj.__init__(int(value),time_course_data=True)
            text = "Data set " + create_option_list()[int(value)]['label']
            return text
        except:
            pass
    else:
        return "Select a mouse"

@app.callback(Output('meanfig', 'figure'),
              [Input('text_message_time', 'children')])
def update_mean_fig(text_input):
    return px.imshow(GUI_obj.mean_log_fig,zmin=np.min(GUI_obj.mean_log_fig), zmax=np.max(GUI_obj.mean_log_fig),
                     width=500,height=500,color_continuous_scale='hot')
