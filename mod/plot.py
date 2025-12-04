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

def multi_agent_update_plot_from_df(df, agent_col: str='agent', time_col: str='interaction', dif_col: str='Δ P(w|m)', color_palette=px.colors.qualitative.Alphabet):
    agent_names = df[agent_col].unique()

    colors = color_palette
    if len(agent_names) < len(colors):
        colors = np.random.choice(colors, size=(len(agent_names),), replace=False)
    else:
        colors = np.random.choice(colors, size=(len(agent_names),))

    fig = go.Figure()

    for i,agent in enumerate(agent_names):
        sub_df = df.loc[df[agent_col].isin([agent])]
        fig.add_trace(
            go.Scatter(
                x=sub_df[time_col].values,
                y=sub_df[dif_col].values,
                mode='lines',
                showlegend=True,
                legendgroup='agent_'+str(agent),
                name='agent_'+str(agent),
                line=dict(color=colors[i])
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

def heatmap(data, x_labels, y_labels, colorscale: str='pubugn'):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            hoverongaps=False,
            colorscale=colorscale
        )
    )
    return fig