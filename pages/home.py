from dash import html
import dash_bootstrap_components as dbc
import dash

# external_stylesheets = [dbc.themes.LUX]
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the fUS GUI", className="text-center")
                    , className="mb-5 mt-5")
        ]),

        dbc.Row([
            dbc.Col(html.H5(children='Currently, you can inspect the data as a video or select specific pixels and'
                                     ' inspect their time courses. Also, you can select the regions within a brain warp.'
                            , className="text-center")
                    , className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='Read the paper',
                                               className="text-center"),
                                       dbc.Button("Link",
                                                  href="https://google.nl",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],class_name="text-center",
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='Access the code',
                                               className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://github.com/rubenwijnands999/dFC-fUS",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],class_name="text-center",
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='Read the MSc thesis',
                                               className="text-center"),
                                       dbc.Button("Repository",
                                                  href="https://repository.tudelft.nl/islandora/object/uuid%3Ae4692392-9010-4875-8392-6801513277c5",
                                                  color="primary",
                                                  className="mt-3"),

                                       ],class_name="text-center",
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4")
        ], className="mb-5"),

    ])

])

# if __name__ == '__main__':
#     app.run_server(debug=True)