import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Layout
from plotly.offline import iplot
from pprint import pprint


def flatten_list(_2d_list):
    """

    :param _2d_list: list
        any two-dimensional list
    :return:
        list: flattened version of the input 2D list
    """
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def find_income_label(income, income_labels, start_num):
    """

    :param income: float
        the income to find the label for
    :param income_labels: list
        contains all possible income labels
    :param start_num: int
        index associated with the beginning of the income labels in the larger list of labels
    :return:
        int, the index of the appropriate income label for the input income
    """
    for label in income_labels:
        label_list = label.split(' - ')
        if 'above' in label_list:
            return income_labels.index(label) + start_num
        if float(label_list[0]) <= round(income, 2) <= float(label_list[1]):
            return income_labels.index(label) + start_num


def make_income_sources(df, income_labels, start_num):
    """

    :param df: dataframe
        contains relevant data
    :param income_labels: list
        contains income specific labels
    :param start_num: int
        index associated with the beginning of the income labels in the larger list of labels
    :return:
        sources, targets, size, all lists formatted for sankey
    """
    neighs = df['neighborhood']
    df['total population'] = df['White Alone'] + df['Black/African-American'] + df['Hispanic'] + \
                             df['Asian alone'] + df['Other Races']
    pops = df['total population'].tolist()
    sources = []
    targets = []
    size = []
    for i in range(len(neighs)):
        income = float(df['Per Capita Income'].tolist()[i])
        label = find_income_label(income, income_labels, start_num)
        sources.append(label)
        targets.append(i + 5)
        size.append(pops[i])
    return sources, targets, size


def make_income_labels(df, num_buckets):
    """

    :param df: dataframe
        contains the relevant data
    :param num_buckets: int
            the number of income buckets to make
    :return:
        labels: the list of income bucket labels
    """
    labels = []
    incomes = df['Per Capita Income'].tolist()
    incomes.sort()
    indices = [round(len(incomes) / num_buckets) * (i + 1) for i in range(num_buckets)]
    for idx in indices:
        if indices.index(idx) > 0:
            bottom = str(round(float(top) + 1, 2))
        else:
            bottom = '0'
        if indices.index(idx) == len(indices) - 1:
            top = 'above'
        else:
            top = str(round(incomes[idx], 2))
        label = bottom + ' - ' + top
        labels.append(label)
    return labels


def neigh_filter(neigh_to_keep, link, labels):
    """

    :param neigh_to_keep: string
        name of the neighborhood to filter for
    :param link: dictionary
        contains unfiltered sankey data
    :param labels: list
        contains sankey node labels
    :return:
        link: filtered sankey data
    """
    new_src = []
    new_trg = []
    new_sz = []
    for i in range(len(link['source'])):
        # if neither the source nor the target is the desired neighborhood, skip to next iteration
        if labels[link['source'][i]] != neigh_to_keep:
            if labels[link['target'][i]] != neigh_to_keep:
                continue
        new_src.append(link['source'][i])
        new_trg.append(link['target'][i])
        new_sz.append(link['value'][i])
    link = {'source': new_src, 'target': new_trg, 'value': new_sz}

    return link


def make_sankey(link, labels, src, targ, **kwargs):
    """

    :param link: dictionary
        contains data formatted for sankey
    :param labels: list
        contains labels for sankey nodes
    :param src: list
        contains indices for node sources
    :param targ: list
        contains indices for node targets
    :param kwargs: keyword arguments
    :return:
        sankey figure
        layers
    """
    # sets parameters for sankey diagram and creates diagram
    pad = kwargs.get('pad', 50)
    thickness = kwargs.get('thickness', 30)
    line_color = kwargs.get('line_color', 'black')

    node = {'label': labels, 'pad': pad, 'thickness': thickness, 'line_color': line_color}

    sk = go.Sankey(link=link, node=node)

    # makes grid not visible
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(sk, layout=layout)

    # labels layers
    layers = flatten_list(src + [elem for elem in targ if elem not in src])
    for x_coor, layer_name in enumerate(layers):
        fig.add_annotation(
            x=x_coor,
            y=1.075,
            xref="x",
            yref="paper",
            text=layer_name.title(),
            showarrow=False,
            font=dict(
                family="Tahoma",
                size=16,
                color="black"
            ),
            align="left",
        )

    # removes x and y axes
    fig.update_xaxes(showgrid=False, color='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=False, color='rgba(0,0,0,0)')

    return fig, layers


def sankey_wrapper(link, labels, src, targ, **kwargs):
    """

    :param link: dictionary
        contains data formatted for sankey
    :param labels: list
        contains labels for sankey nodes
    :param src: list
        contains indices for node sources
    :param targ: list
        contains indices for node targets
    :param kwargs: keyword arguments
    :return: sankey figure
        only the figure itself, not the figure and the layers
    """
    return make_sankey(link, labels, src, targ, **kwargs)[0]


def build_link_labels(df, races, num_buckets):
    """

    :param df: dataframe
        contains relevant data
    :param races: list
        contains race data to add to the labels
    :param num_buckets: int
        the number of income buckets to generate
    :return:
        link: dictionary
            contains data formatted for sankey
        labels: list
            contains node labels for sankey
    """
    sources = []
    targets = []
    size = []
    income_labels = make_income_labels(df, num_buckets)
    labels = races + list(df['neighborhood']) + income_labels

    # populate sources, targets, and sizes for sankey
    for i in range(len(races)):
        for j in range(len(races), len(races) + len(df['neighborhood'])):
            sources.append(j)
            targets.append(i)
            size.append(df[races[i]][j - len(races)])

    src, trg, sz = make_income_sources(df, income_labels, j + 1)
    sources += src
    targets += trg
    size += sz

    # multiply all sizes by 500 for better overall format (increases node size)
    size = [500 * elem for elem in size]

    # format link
    link = {'source': sources, 'target': targets, 'value': size}
    return link, labels


def main():
    # reads data into dataframe and prepares for visualization
    demographics = pd.read_excel('2015-2019_neighborhood_tables_2021.12.21.xlsm', sheet_name=11)
    race = pd.read_excel('2015-2019_neighborhood_tables_2021.12.21.xlsm', sheet_name=2)
    overall = race.merge(demographics, on='neighborhood')
    overall = overall.dropna()
    overall['white percent'] = overall['%']
    overall['black percent'] = overall['%.1']
    overall['hispanic percent'] = overall['%.2']
    overall['asian percent'] = overall['%.3']
    overall['other percent'] = overall['%.4']
    overall = overall[['White Alone', 'Black/African-American', 'Hispanic', 'Asian alone', 'Other Races',
                       'Per Capita Income', 'neighborhood']]

    races = ['White Alone', 'Black/African-American', 'Hispanic', 'Asian alone', 'Other Races']
    neighs = overall['neighborhood']

    link, labels = build_link_labels(overall, races, 3)

    # call to make sankey and show visualization
    sankey_fig, layers = make_sankey(link, labels, ['Per Capita Income (USD)', 'Neighborhood'],
                                     ['Neighborhood', 'Demographic'])
    fig = go.Figure(sankey_fig)

    # get updated sankeys for filtering and add dropdown menu with neighborhood buttons
    neigh_update_data = [neigh_filter(neighs[i], link, labels) for i in range(len(neighs))]
    neigh_update_menu = [{'method': 'animate', 'label': neighs[i], 'args': [
        sankey_wrapper(neigh_update_data[i], labels,
                       ['Per Capita Income (USD)', 'Neighborhood'],
                       ['Neighborhood', 'Demographic'])]}
                         for i in range(len(neighs))]
    neigh_update_menu.insert(0, {'method': 'animate', 'label': 'all', 'args': [fig]})
    updatemenus = [{'buttons': neigh_update_menu}]
    fig.update_layout(updatemenus=updatemenus)

    # add labels for sankey layers
    for x_coor, layer_name in enumerate(layers):
        fig.add_annotation(
            x=x_coor,
            y=1.075,
            xref="x",
            yref="paper",
            text=layer_name.title(),
            showarrow=False,
            font=dict(
                family="Tahoma",
                size=16,
                color="black"
            ),
            align="left",
        )
    # add label for dropdown filter
    fig.add_annotation(dict(text="Neighborhood", x=-.2, xref="paper", y=1.06, yref="paper",
                            align="left", showarrow=False))

    # write to html file for website
    fig.write_html("sankey.html")
    fig.show()


if __name__ == "__main__":
    main()
