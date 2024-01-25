import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Define the app layout
app.layout = dbc.Container(fluid=True, children=[
    html.H1('Interactive Bayesian Network', className='text-center my-4'),
    dbc.Row([
        dbc.Col([
            html.Label('Probability that the transaction amount is greater than $1000:'),
            dcc.Slider(id='prob-A', min=0, max=1, step=0.01, value=0.5, marks={i/10: str(i/10) for i in range(0, 11)}),
        ], width=4),
        dbc.Col([
            html.Label('Probability that the transaction is being performed on a risky country/location:'),
            dcc.Slider(id='prob-B', min=0, max=1, step=0.01, value=0.5, marks={i/10: str(i/10) for i in range(0, 11)}),
        ], width=4),
        dbc.Col([
            html.Label('Probability that the transaction is being performed at midnight - unusual time:'),
            dcc.Slider(id='prob-C', min=0, max=1, step=0.01, value=0.5, marks={i/10: str(i/10) for i in range(0, 11)}),
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select nodes to connect:'),
            dcc.Dropdown(
                id='node-connections',
                options=[
                    {'label': 'A -> B', 'value': 'AB'},
                    {'label': 'A -> C', 'value': 'AC'},
                    {'label': 'B -> C', 'value': 'BC'}
                ],
                multi=True,
                value=[]
            )
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='bayesian-network')
        ], width=12, style={'margin-bottom': '0px'}),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='joint-probability', className='text-center my-4')
        ], width=12, style={'margin-bottom': '0px'}),
    ]),
], style={'maxWidth': '960px'})

# Define the callback for updating the network graph and joint probability
@app.callback(
    [Output('bayesian-network', 'figure'),
     Output('joint-probability', 'children')],
    [Input('prob-A', 'value'),
     Input('prob-B', 'value'),
     Input('prob-C', 'value'),
     Input('node-connections', 'value')]
)
def update_graph(prob_A, prob_B, prob_C, connections):
# Node coordinates
    node_coords = {'A': (1,1), 'B': (2, 2), 'C': (3, 1)}
    
    # Initialize list for edge traces
    edge_traces = []
    
    # Create annotations list for the arrows
    annotations = []
    
    # Define the arrowhead size
    arrow_size = 1

    # Create edges based on selected connections and annotations for arrows
    for connection in connections:
        if connection == 'AB':
            edge_traces.append(('Node A', 'Node B'))
        elif connection == 'AC':
            edge_traces.append(('Node A', 'Node C'))
        elif connection == 'BC':
            edge_traces.append(('Node B', 'Node C'))

    # Create traces for edges and arrow annotations
    for edge in edge_traces:
        x0, y0 = node_coords[edge[0][-1]]
        x1, y1 = node_coords[edge[1][-1]]
        
        # Add line for the edge
        annotations.append(
            dict(
                x=x1, y=y1,  # arrowhead position
                ax=x0, ay=y0,  # tail position
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,  # style of the arrowhead
                arrowsize=arrow_size,
                arrowwidth=2
            )
        )

    # Create node trace
    base_size=40
    node_trace = go.Scatter(
        x=[node_coords['A'][0], node_coords['B'][0], node_coords['C'][0]],
        y=[node_coords['A'][1], node_coords['B'][1], node_coords['C'][1]],
        text=[f'P(Amount > $1000) = {prob_A}', 
              f'P(Location = Risky) = {prob_B}', 
              f'P(Time = Midnight) = {prob_C}'],
        mode='markers+text',
        textposition='top center',
        marker=dict(size=[base_size+prob_A*50, base_size+prob_B*50, base_size+prob_C*50], color='lightblue'),
        hoverinfo='text',
        textfont=dict(
            size=14,  # Ajusta el tamaño del texto según tus preferencias
            color='black',  # Color del texto
        )
    )
    
    # Calculate the joint probability
    joint_prob = prob_A * prob_B * prob_C

    # Define layout
    layout = go.Layout(
        title = {
        'text': f"Probability of Fraud = {joint_prob:.2f}",
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Anchor the title to the center
        'font': {
            'size': 30,  # Adjust font size as needed
            'color': 'black',  # Adjust font color as needed
         },
        },
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, 
                   showticklabels=False,
                   range = [0.45,3.5]),
        yaxis=dict(showgrid=False, zeroline=False, 
                   showticklabels=False, range=[0.5, 2.7]),
        annotations=annotations  # Add the annotations to the layout,
    )

    # Define figure
    figure = go.Figure(data=[node_trace], layout=layout)

    # Calculate the joint probability
    joint_prob = prob_A * prob_B * prob_C
    joint_prob_text = f'Joint Probability P(A) * P(B) * P(C) = {joint_prob:.2f}'
    
    return figure, "Disclaimer: This is a toy example, and does not represent how SciFraud works. This page was generated by MRM Analytics (www.mrmanalytics.com)"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)