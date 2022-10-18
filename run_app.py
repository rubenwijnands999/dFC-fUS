from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app
# import all pages in the app
from pages import home,video,time_courses,warping
from GUI_tools import *
from data_import import Data

# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("fUS video", href="/fUS_video"),
        dbc.DropdownMenuItem("Time course inspector", href="/timecourses"),
    ],
    nav = True,
    in_navbar = True,
    label = "Data",
)


navbar = dbc.Navbar(
    dbc.Container([
        dbc.Col(
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/favicon.ico", height="30px")),
                        dbc.Col(dbc.NavbarBrand("fUS GUI", className="ml-auto")),
                    ]
                ),href="/home",style={'margin': 'left'}
            ),width="auto"
        ),
        dbc.Col([
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            )
        ], width="auto"),
        dbc.Col(
            dbc.Nav(dbc.NavItem(dbc.NavLink("Warping", href="/warping")),navbar=True)
            ,width="auto"
        ),
        dbc.Col([
            dbc.Select(
                id="mouse_select",
                options=create_option_list(),
            )
        ], width="auto"
        ),
    ]),
    color="dark",
    dark=True,
    className="mb-4",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/fUS_video':
        return video.layout
    elif pathname == '/timecourses':
        return time_courses.layout
    elif pathname == '/warping':
        return warping.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)