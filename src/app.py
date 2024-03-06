import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from dash import dash_table
import dash_bootstrap_components as dbc
import pandas as pd

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Estilo para las celdas de las tablas
style_cell={
    'textAlign': 'left',
    'border': 'thin lightgrey solid'  # Añadir bordes a las celdas
}
style_header={
    'backgroundColor': 'lightgrey',
    'fontWeight': 'bold',
    'border': 'thin lightgrey solid',
    'color': 'black'
}
style_data={  # Estilos para las celdas que no están en modo de edición
    'backgroundColor': 'white',
    'color': 'black'
}

# Estilo para la tabla completa
style_table = {
    'overflowX': 'auto',  # Permite el desplazamiento horizontal si la tabla es muy ancha
    'border': 'thin lightgrey solid' # Borde alrededor de la tabla
}

# Datos para la tabla
data_table = pd.DataFrame({
    "Amount": [">$1000", ">$1000", ">$1000", ">$1000", "<$1000", "<$1000", "<$1000", "<$1000"],
    "Location": ["Usual", "Usual", "Unusual", "Unusual", "Usual", "Usual", "Unusual", "Unusual"],
    "Time": ["Day", "Night", "Day", "Night", "Day", "Night", "Day", "Night"],
    #"Count": [70,50,30,10,80,60,40,20]
    "Count": [300,70,7,3,350,150,40,5]
})

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Define the app layout
app.layout = dbc.Container(fluid=True, children=[
    html.H1('Interactive Bayesian Network', className='text-center my-4'),

    # Observations Table
    html.P("Imagine you're a detective analyzing transactions to find patterns that might suggest fraudulent activity. The table below is like a collection of clues, each row representing a unique combination of factors observed during different financial transactions in a dataset designed for training investigators like you.",
           style={'textAlign': 'justify'}),
    dbc.Row(
            dbc.Col(
                dash_table.DataTable(
                    id='observation-table',
                    columns=[{"name": i, "id": i} for i in data_table.columns],
                    data=data_table.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',  # Espaciado dentro de las celdas
                        'border': 'thin lightgrey solid'  # Añadir bordes a las celdas
                    },
                    style_header={
                        'backgroundColor': 'lightgrey',
                        'fontWeight': 'bold',
                        'border': 'thin lightgrey solid',
                        'color': 'black'
                    },
                    style_data={  # Estilos para las celdas que no están en modo de edición
                        'backgroundColor': 'white',
                        'color': 'black'
                    },
                ),
                width={'size': 6, 'offset': 3},  # Ajusta el tamaño y el offset para centrar
                style={'textAlign': 'center'}
            ),
            justify="left",  # Centra la fila
        ),
    html.Div(style={'marginTop': 10}),
    html.P("Let's dissect the first row as an example. It tells us that on 300 occasions, transactions exceeded $1000, took place in what we consider a 'usual' location, and happened during daylight hours. Each row is a piece of the puzzle, helping us to understand how common or rare certain combinations of 'Amount', 'Location', and 'Time' are.",
           style={'textAlign': 'justify'}),

    html.Div(style={'marginTop': 20}),

    # Bayesian Network Graph
    html.H3('Bayesian Network: A graphical representation'),
    html.P([
        "The visual below is your ",
        html.Strong('Bayesian Network'),
        ", a map that connects the dots between different aspects of our investigation. Each circle, or 'node', represents one of the factors: 'Amount', 'Location', and 'Time'."
    ],
    style={'textAlign': 'justify'}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='bayesian-network')
        ], width=12, style={'margin-bottom': '5px'}),
    ]),

    # Node Connection Selector
    html.P(["You are not just observing the map; you are shaping it. By ",
            html.Strong("connecting the nodes"),
            ", you hypothesize which factors might influence others."],
            style={'textAlign': 'justify'}),
    dcc.Dropdown(
        id='node-connections',
        options=[
            {'label': 'Amount -> Location', 'value': 'AB'},
            {'label': 'Amount -> Time', 'value': 'AC'},
            {'label': 'Location -> Time', 'value': 'BC'},
            {'label': 'Location -> Amount', 'value': 'BA'},
            {'label': 'Time -> Amount', 'value': 'CA'},
            {'label': 'Time -> Location', 'value': 'CB'}
        ],
        multi=True,
        value=[]
    ),
], style={'maxWidth': '960px'})

app.layout.children.extend([
    html.Div(style={'marginTop': 20}),
    html.H3('Conditional Probability Tables'),
    html.P(["As you delve deeper into the world of transaction analysis, the tables you see before you are your dynamic tools for understanding complex relationships in the data. The top row displays ",
            html.Strong("Conditional Probability Tables (CPTs)"),
            ", each tailored based on the connections you hypothesize between 'Amount', 'Location', and 'Time'."],
            style={'textAlign': 'justify'}),
    dbc.Row(id='conditional-probability-tables', children=[
        dbc.Col(id='conditional-table-amount', width=4),
        dbc.Col(id='conditional-table-location', width=4),
        dbc.Col(id='conditional-table-time', width=4),
    ]),
    html.Div(style={'marginTop': 10}),
    html.P("Consider these CPTs as adjustable lenses of a microscope. As you change the connections in your Bayesian Network, the CPTs recalibrate, offering new probabilities that reflect these hypotheses. A connection between 'Amount' and 'Location' might suggest that the frequency of transactions in different locations varies with the transaction amount. On the other hand, linking 'Time' to 'Location' might indicate that the usualness of a location depends on the time of day.",
           style={'textAlign': 'justify'}),
    html.Div(style={'marginTop': 20}),
    html.H3('Joint Probability Distribution'),
    html.P("The real magic happens in the 'Joint Probability Distribution' table below. Here, we combine the probabilities from our CPTs, turning individual insights into a comprehensive prediction model. This table will tell us the combined probability of all factors at play, considering the current structure of our Bayesian Network. It’s the culmination of our investigative story, where individual clues merge into a coherent narrative of transaction behaviors.", style={'textAlign': 'justify'}),
    dbc.Row(
        dbc.Col(id='joint-prob-table', width=7),
        style={'textAlign': 'center'},
        justify="center"
    ),
    html.Div(style={'marginTop': 10}),
    html.P("Picture this: a high-value transaction over $1000 occurs in an unusual location late at night. It's an anomaly that stands out against the backdrop of normal activity. With your Bayesian Network, you check the probabilities and find such a combination to be rare, a stark contrast to the usual pattern. This discrepancy suggests a potential fraud risk, prompting further investigation. Your network doesn't convict, but it does shine a light on where to dig deeper.", style={'textAlign': 'justify'}),
    html.Div(style={'marginTop': 20}),
    dcc.Store(id='store-joint-probability', data=None),  # Store component to hold the joint probability table
    html.H3('Calculate the probability of fraud'),
    html.Div(style={'marginTop': 10}),
    html.P("As you, the detective, select different transaction characteristics, remember: the rarer they are, the more suspicious they become. The probability of fraud is calculated as one minus the joint probability of the observed combination—essentially, how much the actual event stands out from the expected pattern", style={'textAlign': 'justify'}),
    html.P("Adjust 'Amount', 'Location', and 'Time' to see how the fraud probability changes. The tool you're using is like a magnifying glass, bringing into focus the oddities within the data, guiding your investigation towards the anomalies that could signal a scam.", style={'textAlign': 'justify'}),
    html.Div(style={'marginTop': 10}),
    dbc.Row([
    dbc.Col(
        dcc.Dropdown(
            id='selector-amount',
            options=[
                {'label': 'Less than $1000', 'value': '<$1000'},
                {'label': 'More than $1000', 'value': '>$1000'}
            ],
            value='<$1000',  # Valor predeterminado
            clearable=False,  # Evita que el usuario deje el selector vacío
            style={'width': '100%'}
        ),
        width=4  # Tamaño de la columna (4 de 12)
    ),
    dbc.Col(
        dcc.Dropdown(
            id='selector-location',
            options=[
                {'label': 'Usual', 'value': 'Usual'},
                {'label': 'Unusual', 'value': 'Unusual'}
            ],
            value='Usual',  # Valor predeterminado
            clearable=False,
            style={'width': '100%'}
        ),
        width=4  # Tamaño de la columna
    ),
    dbc.Col(
        dcc.Dropdown(
            id='selector-time',
            options=[
                {'label': 'Day', 'value': 'Day'},
                {'label': 'Night', 'value': 'Night'}
            ],
            value='Day',  # Valor predeterminado
            clearable=False,
            style={'width': '100%'}
        ),
        width=4  # Tamaño de la columna
    )
], className='mb-3'), 

html.Div(id='fraud-probability-result', style={'textAlign': 'center'},
         ),
html.P("*Red (above 95%), Orange (between 80% and 95%), Yellow (between 60% and 79.9%), and Green (below 60%).", style={'textAlign': 'center', "fontSize":12}),
html.Div(style={'marginTop': 50}),
html.P(["This widget was created by ",html.A("MRM Analytics®",
        href="https://www.mrmanalytics.com/", target="_blank")], 
        style={'textAlign': 'center'}),
])

# Define the callback for updating the network graph and joint probability
@app.callback(
    [
        Output('bayesian-network', 'figure'),
        Output('conditional-table-amount', 'children'),
        Output('conditional-table-location', 'children'),
        Output('conditional-table-time', 'children'),
        Output('joint-prob-table', 'children'),
        Output('store-joint-probability', 'data')
    ],
    [Input('node-connections', 'value')]
)
def update_graph(connections):
    prob_A, prob_B, prob_C = 1.0,1.0,1.0
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
        elif connection == 'BA':
            edge_traces.append(('Node B', 'Node A'))
        elif connection == 'CA':
            edge_traces.append(('Node C', 'Node A'))
        elif connection == 'CB':
            edge_traces.append(('Node C', 'Node B'))

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
        text=[f'Amount', 
              f'Location', 
              f'Time'],
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
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
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
    #joint_prob = prob_A * prob_B * prob_C
    #joint_prob_text = f'Joint Probability P(A) * P(B) * P(C) = {joint_prob:.2f}'
    
        # Initialize the conditional probability tables as empty DataFrames
    df_amount = pd.DataFrame()
    df_location = pd.DataFrame()
    df_time = pd.DataFrame()

    # Check the selected connections and calculate the corresponding crosstabs
    # This is a placeholder logic, you need to adapt it to your actual use case
    
    # Get graph edges
    maping = {"A":"Amount", "B":"Location", "C":"Time"}
    edges = []
    if connections != []:
        for ele in connections:
            edges.append((maping[ele[0]], maping[ele[1]]))

    
    # Get conditionals by node
    conditionals = {"Amount":[], "Location":[], "Time":[]}
    for node in conditionals:
        for edge in edges:
            if edge[1] == node:
                conditionals[node].append(edge[0])

    # Build CPTs
    if conditionals["Amount"] != []:
        df_amount = pd.crosstab(index=data_table['Amount'], 
                                columns=[data_table[key] 
                                         for key in conditionals["Amount"]], 
                                values=data_table['Count'], 
                                aggfunc='sum', normalize='index')
        df_amount = df_amount.T.reset_index()
        df_amount
    else:
        df_amount = data_table.groupby("Amount")["Count"].sum()
        df_amount = (df_amount / df_amount.sum())
        df_amount = df_amount.reset_index()
        df_amount = df_amount.rename(columns={"Count":"Prob"})
    df_amount=df_amount.round(2)
    
    if conditionals["Location"] != []:
        df_location = pd.crosstab(index=data_table['Location'], 
                                columns=[data_table[key] 
                                for key in conditionals["Location"]], 
                                values=data_table['Count'], 
                                aggfunc='sum', normalize='index')
        df_location = df_location.T.reset_index()
    else:
        df_location = data_table.groupby("Location")["Count"].sum()
        df_location = (df_location / df_location.sum())
        df_location = df_location.reset_index()
        df_location = df_location.rename(columns={"Count":"Prob"})
    df_location=df_location.round(2)

    if conditionals["Time"] != []:
        df_time = pd.crosstab(index=data_table['Time'], 
                                columns=[data_table[key] 
                                for key in conditionals["Time"]], 
                                values=data_table['Count'], 
                                aggfunc='sum', normalize='index')
        df_time=df_time.T.reset_index()
    else:
        df_time = data_table.groupby("Time")["Count"].sum()
        df_time = (df_time / df_time.sum())
        df_time = df_time.reset_index()
        df_time = df_time.rename(columns={"Count":"Prob"})
    df_time=df_time.round(2)

    joint_table = calculate_joint_probability(df_amount, df_location, df_time, conditionals)

    # Create DataTable components for the conditional probability tables
    table_amount = dash_table.DataTable(
        data=df_amount.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_amount.columns],
        style_cell=style_cell,
        style_header=style_header,
        style_table=style_table
    )

    table_location = dash_table.DataTable(
        data=df_location.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_location.columns],
        style_cell=style_cell,
        style_header=style_header,
        style_table=style_table,
    )

    table_time = dash_table.DataTable(
        data=df_time.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_time.columns],
        style_cell=style_cell,
        style_header=style_header,
        style_table=style_table,
    )

    table_joint = dash_table.DataTable(
        data=joint_table.to_dict('records'),
        columns=[{"name": i, "id": i} for i in joint_table.columns],
        style_cell={
            'textAlign': 'left',
            'padding': '10px',  # Espaciado dentro de las celdas
            'border': 'thin lightgrey solid'  # Añadir bordes a las celdas
        },
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold',
            'border': 'thin lightgrey solid',
            'color': 'black'
        },
        style_data={  # Estilos para las celdas que no están en modo de edición
            'backgroundColor': 'white',
            'color': 'black'
        },
    )

    joint_probability_data = joint_table.to_dict('records')
    
    return figure, table_amount, table_location, table_time, table_joint, joint_probability_data

import copy
import numpy as np

def calculate_joint_probability(df_amount, df_location, df_time, network_structure):
    # Calcular la probabilidad conjunta
    df_joint = copy.deepcopy(data_table)
    df_joint = df_joint.rename(columns={"Count":"Prob"})
    df_joint["Prob"] = 0.0

    for _,row in df_joint.iterrows():
        prob = 1
        for col in ["Amount", "Location", "Time"]:
            parents = network_structure.get(col)
            if col == "Amount":
                df = copy.deepcopy(df_amount).reset_index()
            elif col == "Location":
                df = copy.deepcopy(df_location).reset_index()
            elif col == "Time":
                df = copy.deepcopy(df_time).reset_index()

            if parents != []:
                val = None
                for parent in parents:
                    if val is None:
                         val = df[parent] == row[parent]
                    else:
                        val = val & (df[parent] == row[parent])
                
                    prob *= df.loc[val, row[col]].values[0]
            else:
                prob *= df.loc[df[col] == row[col], "Prob"].values[0]

        df_joint.loc[(df_joint["Amount"] == row["Amount"]) &
                    (df_joint["Location"] == row["Location"])&
                    (df_joint["Time"] == row["Time"]), "Prob"] = prob
        df_joint["Prob"] /= df_joint["Prob"].sum()
    
    return df_joint.round(3)

@app.callback(
    Output('fraud-probability-result', 'children'),
    [Input('store-joint-probability', 'data'),
     Input('selector-amount', 'value'),
     Input('selector-location', 'value'),
     Input('selector-time', 'value')]
)
def update_fraud_probability(joint_table, selected_amount, selected_location, selected_time):
    # Asumiendo que 'df_joint_probability' es tu DataFrame con la Joint Probability Distribution
    # Busca la probabilidad correspondiente
    tab = pd.DataFrame(joint_table)

    prob = tab.loc[
        (tab['Amount'] == selected_amount) &
        (tab['Location'] == selected_location) &
        (tab['Time'] == selected_time),
        'Prob'
    ].values[0]  # Obtiene el primer valor de la serie resultante

    lower = 0.05#tab["Prob"].quantile(0.2)
    mid = 0.2#tab["Prob"].quantile(0.5)
    upper = 0.4#tab["Prob"].quantile(0.8)

    if prob <= lower:
        color = "red"
    elif (prob > lower) & (prob <= mid):
        color = "orange"
    elif (prob > mid) & (prob <= upper):
        color = "#DAA520"
    else:
        color = "green"

    # Devuelve un mensaje con la probabilidad
    list_ = [
        "Probability that this is not a normal transaction: ",
         html.Strong(f"{1-prob:.2%}", style={'textAlign': 'justify', 'fontSize': '24px', 'color':color})
    ]

    if color == "red":
        list_.append(html.Strong(" (This is definitely fraud!)", 
                style={'textAlign': 'justify', 
                       'fontSize': '18px', 'color':color})
        )
    elif color == "orange":
        list_.append(html.Strong(" (Hmm, this doesn't add up...)", 
                style={'textAlign': 'justify', 
                       'fontSize': '18px', 'color':color})
        )
    elif color == "#DAA520":
        list_.append(html.Strong(" (Something seems a little bit odd)", 
                style={'textAlign': 'justify', 
                       'fontSize': '18px', 'color':color})
        )
    elif color == "green":
        list_.append(html.Strong(" (Great! This is definitely legitimate)", 
                style={'textAlign': 'justify', 
                       'fontSize': '16px', 'color':color})
        )

    return [html.P(list_)]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)