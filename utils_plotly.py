import plotly.express as px


def plot_frequency(data: list, title: str = None):
    return px.line(y=data, range_y=[0, 1], title=title)
