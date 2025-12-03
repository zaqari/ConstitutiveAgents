import plotly.graph_objects as go
import numpy as np
import plotly.express as px

def plot(data, variable_name: str):
    fig = go.Figure()

    colors = px.colors.qualitative.Alphabet
    color = np.random.choice(colors, size=(1,))[0]

    fig.add_trace(
        go.Scatter(
            x = list(range(len(data))),
            y = data,
            mode='lines',
            showlegend=True,
            legendgroup=variable_name,
            name=variable_name,
            line=dict(color=color)
        )
    )

    return fig

def add_to_plot(data, variable_name: str, fig):
    colors = px.colors.qualitative.Alphabet
    color = np.random.choice(colors, size=(1,))[0]

    fig.add_trace(
        go.Scatter(
            x=list(range(len(data))),
            y=data,
            mode='lines',
            showlegend=True,
            legendgroup=variable_name,
            name=variable_name,
            line=dict(color=color)
        )
    )

    return fig