import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

class Heatmap(object):

    def __init__(self, xs, ys, sim):
        self.xs = [str(x) for x in xs]
        self.ys = ys
        # assert sim is a list of list
        self.sim = sim

    def plot(self, title, fn):
        import plotly.plotly as py
        import plotly.graph_objs as go
        data = [
            go.Heatmap(
                z=self.sim,
                x=self.ys,
                y=self.xs,
                colorscale='Jet',
            )
        ]
        layout = go.Layout(
            title=title,
            xaxis = dict(ticks='' ),
            yaxis = dict(ticks='' ),
            autosize=False,
            width=500,
            height=500,
            margin=go.Margin(
                l=100,
                r=100,
                b=100,
                t=100,
                pad=4
            ),
        )
        fig = go.Figure(data=data, layout=layout)
        py.image.save_as(fig, filename=fn)
