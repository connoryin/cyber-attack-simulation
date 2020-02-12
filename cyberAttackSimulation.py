from graph_tool.all import *
from numpy.random import *
import numpy as np
from random import sample, gauss
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
import sys, os, os.path
import time, math
import statsmodels.api as sm
from arch import arch_model

seed()

from gi.repository import Gtk, Gdk, GdkPixbuf, GObject

plt.switch_backend('cairo')

# parameters:
time_to_stop = 200
graph_to_show = 'random'
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'Orange', 'Tomato', 'Navy', 'Plum', 'Purple']
simulation = 0


def create_qq_plot():
    f = Figure()
    a = f.add_subplot(221)
    b = f.add_subplot(222)
    c = f.add_subplot(223)
    d = f.add_subplot(224)
    f.subplots_adjust(hspace=0.4)
    f.suptitle('QQ plots of PDF & log-return vs. normal distribution', fontweight='bold')
    canvas = FigureCanvas(f)
    return a, b, c, d, canvas

def create_plot(xlabel, ylabel, title, data):
    f = Figure()
    a = f.add_subplot(211)
    num = 0
    for d in data[0::2]:
        a.plot(d, colors[num])
        num += 1
        if num >= len(colors):
            num = 0
    a.set_xlabel(xlabel, labelpad=0, fontdict={'fontweight': 'bold'})
    a.set_ylabel(ylabel, labelpad=0, fontdict={'fontweight': 'bold'})
    # a.legend()
    b = f.add_subplot(212)
    num = 0
    for d in data[1::2]:
        b.plot(d, colors[num])
        num += 1
        if num >= len(colors):
            num = 0
    b.set_xlabel(xlabel, labelpad=0, fontdict={'fontweight': 'bold'})
    b.set_ylabel(ylabel, labelpad=0, fontdict={'fontweight': 'bold'})
    if (title == 'Prediction'): f.subplots_adjust(hspace=0.6)
    canvas = FigureCanvas(f)
    return a, b, canvas


def create_graph(type, size=100,
                 p_for_random_graph=0.1,
                 mean_for_homogeneous_graph=6,
                 standard_deviation_for_homogeneous_graph=0.25):
    g = Graph(directed=False)
    if type == 'price':
        g = price_network(size)
        g = GraphView(g, directed=False)
    elif type == 'ring':
        g = circular_graph(size)
    elif type == 'star':
        node = g.add_vertex()
        leaves = g.add_vertex(size - 1)
        for l in leaves:
            g.add_edge(l, node)
    elif type == 'cluster':
        running_size = size
        rnd = random_integers(1, running_size)
        g = complete_graph(rnd)
        v = g.vertex(randint(0, g.num_vertices()))
        preV = g.vertex(randint(0, g.num_vertices()))
        running_size -= rnd
        while (running_size > 0):
            rnd = random_integers(1, running_size)
            g.add_vertex(rnd)
            for i in range(g.num_vertices() - rnd, g.num_vertices()):
                for j in range(i + 1, g.num_vertices()):
                    g.add_edge(i, j)
            running_size -= rnd
            curV = g.vertex(randint(g.num_vertices() - rnd, g.num_vertices()))
            g.add_edge(preV, curV)
            preV = g.vertex(randint(g.num_vertices() - rnd, g.num_vertices()))
        g.add_edge(preV, v)
    elif type == 'random':
        g.add_vertex(size)
        for i, j in [(i, j) for i in range(0, size) for j in range(i + 1, size)]:
            if random() < p_for_random_graph:
                g.add_edge(g.vertex(i), g.vertex(j))
    elif type == 'homogeneous':
        g = random_graph(size,
                         lambda: math.ceil(gauss(mean_for_homogeneous_graph, standard_deviation_for_homogeneous_graph)),
                         directed=False)
    return g


def update_data(figure, xlabel, ylabel, title, data):
    figure.clear()
    num = 0
    for d in data:
        if title == 'Prediction':
            figure.hist(d, cumulative=True, color=colors[num], histtype='bar', rwidth=0.1)

        else:
            figure.plot(d, colors[num])
        num += 1
        if num >= len(colors):
            num = 0
    figure.set_xlabel(xlabel, labelpad=0, fontdict={'fontweight': 'bold'})
    figure.set_ylabel(ylabel, labelpad=0, fontdict={'fontweight': 'bold'})
    figure.set_title(title, {'fontweight': 'bold'})
    figure.grid()


r = 0.1  # I->S probability
s = 0.1  # R->S probability
beta = 0.05
incubation = 0.5  # E->I probability
number_of_infected_at_beginning = 5

graph_list = []
g = create_graph(graph_to_show)
graph_list.append(g)
# graph_list.append(mstg)
# graph_list.append(create_graph('star'))
graph_list.append(g.copy())

layout_list = []
for graph in graph_list:
    layout_list.append(sfdp_layout(graph))

simulation_list = graph_list.copy()
mst = min_spanning_tree(g)
mstg = GraphView(g, efilt=mst, directed=False)
mstg = Graph(mstg, prune=True)
simulation_list.append(mstg)
simulation_list.append(mstg.copy())

cover, c = max_cardinality_matching(g)
coverg = GraphView(g, efilt=cover, directed=False)
coverg = Graph(coverg, prune=True)
simulation_list.append(coverg)
simulation_list.append(coverg.copy())


def find_rep(a, list):
    if list[a] == a:
        return a
    else:
        list[a] = find_rep(list[a], list)
        return list[a]


connected_coverg = coverg.copy()
for v in connected_coverg.vertices():
    flag = False
    for n in v.all_neighbors():
        for nn in n.all_neighbors():
            if not connected_coverg.vertex_index[nn]:
                flag = True
    if not flag:
        connected_coverg.add_edge(connected_coverg.vertex(0), v)
simulation_list.append(connected_coverg)
simulation_list.append(connected_coverg.copy())

cutg = g.copy()
cut = g.new_vertex_property('bool')
cut_array = cut.a
for i in range(len(cut_array)):
    cut_array[i] = randint(0, 2)
for e in g.edges():
    if cut[e.source()] != cut[e.target()]:
        if len(list(cutg.vertex(g.vertex_index[e.source()]).all_neighbors())) != 1 and len(
                list(cutg.vertex(g.vertex_index[e.target()]).all_neighbors())) != 1:
            cutg.remove_edge(e)
simulation_list.append(cutg)
simulation_list.append(cutg.copy())

model_list = ['SIS', 'SIRS', 'SIS', 'SIRS', 'SIS', 'SIRS', 'SIS', 'SIRS', 'SIS', 'SIRS']
label_list = ['SIS', 'SIRS', 'MST', 'Edge Cover', 'Connected Edge Cover', 'Exact Cover', 'MST', 'Edge Cover',
              'Connected Edge Cover', 'Exact Cover']

S = [0, 1, 0, 1]  # Green color
I = [1, 0, 0, 1]  # Red color
R = [0, 1, 1, 1]  # Blue color
E = [1, 0, 1, 1]  # Purple color

state_list = []
for g in simulation_list:
    state_list.append(g.new_vertex_property("vector<double>"))
for i, g in enumerate(simulation_list):
    for v in g.vertices():
        state_list[i][v] = S
    vt = list(g.vertices())
    sp = sample(vt, number_of_infected_at_beginning)
    for s in sp:
        state_list[i][s] = I

frequency_list = []
for g in simulation_list:
    frequency_list.append([number_of_infected_at_beginning / g.num_vertices()])
distribution_list = []
for g in simulation_list:
    distribution_list.append([0] * (g.num_vertices() + 1))
num_infected_list = [number_of_infected_at_beginning] * len(simulation_list)
newly_infected_list = []
for g in simulation_list:
    newly_infected_list.append(g.new_vertex_property("bool"))
edge_state_list = []
for g in graph_list:
    eprop = g.new_edge_property('vector<double>')
    for e in g.edges():
        eprop[e] = [0.8, 0.8, 0.8, 1]
    edge_state_list.append(eprop)
log_list = [[] for i in range(len(simulation_list))]
error_list = [[] for i in range(len(simulation_list))]

time = 1

# If True, the frames will be dumped to disk as images.
offscreen = sys.argv[1] == "offscreen" if len(sys.argv) > 1 else False
max_count = 20
if offscreen and not os.path.exists("../frames"):
    os.mkdir("../frames")


class SimulationWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title='Cyber Attack Simulation')
        self.big_box = Gtk.Box()
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.box_2 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.legend_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.graph_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(self.big_box)
        self.graphs = []

        for i, g in enumerate(graph_list):
            label = Gtk.Label(
                '<span size="xx-large" weight="bold">' + model_list[i] + '</span>')
            label.set_use_markup(True)
            self.graph_box.pack_start(label, True, True, 0)
            graph_draw(g, pos=layout_list[i],
                       vprops={'fill_color': state_list[i], 'halo': newly_infected_list[i], 'halo_color': [1, 0, 1, 1]},
                       eprops={'color': edge_state_list[i]}, output=str(i) + '.png',
                       output_size=(400, 400))
            img = Gtk.Image()
            img.set_from_file(str(i) + '.png')
            self.graphs.append(img)
            self.graph_box.pack_start(self.graphs[i], True, True, 0)

        img = Gtk.Image()
        img.set_from_file('legend.png')
        self.graph_box.pack_start(img, True, True, 0)

        self.big_box.pack_start(self.graph_box, False, False, 0)

        self.a11, self.a12, self.canvas1 = create_plot('', '', '',
                                                       frequency_list)
        self.box.pack_start(self.canvas1, True, True, 0)

        self.a21, self.a22, self.canvas2 = create_plot('Number of infected nodes', 'Frequency',
                                                       'Frequency Density (PDF)',
                                                       [[x / time for x in distribution_list[i]] for i in
                                                        range(len(distribution_list))])
        self.box.pack_start(self.canvas2, True, True, 0)

        self.a31, self.a32, self.canvas3 = create_plot('', '', '', [])
        self.box.pack_start(self.canvas3, True, True, 0)

        self.a41, self.a42, self.canvas4 = create_plot('', '', 'Prediction', log_list)
        self.box_2.pack_start(self.canvas4, True, True, 0)

        self.a51, self.a52, self.canvas5 = create_plot('', '', 'Prediction', [])
        self.box_2.pack_start(self.canvas5, True, True, 0)

        self.a61, self.a62, self.a63, self.a64, self.canvas6 = create_qq_plot()
        self.box_2.pack_start(self.canvas6, True, True, 0)

        self.big_box.pack_start(self.box, True, True, 0)
        self.big_box.pack_start(self.box_2, True, True, 0)

        self.set_default_size(1920, 1080)


# This creates a GTK+ window with the initial graph layout
if not offscreen:
    win = SimulationWindow()
else:
    pass


# count = 0
# win = Gtk.OffscreenWindow()
# win.set_default_size(1920, 1080)
# win.graph = GraphWidget(g, pos, edge_color=[0.6, 0.6, 0.6, 1],
#                         vertex_fill_color=state,
#                         vertex_halo=newly_infected,
#                         vertex_halo_color=[0.8, 0, 0, 0.6])
# win.add(win.graph)
#
# f1 = Figure(figsize=(5, 4))
# a1 = f1.add_subplot(111)
# a1.plot(frequency, '-b')
# a1.set_xlabel('Time', labelpad=0)
# a1.set_ylabel('Proportion', labelpad=0)
# a1.set_title('Proportion of infected nodes vs. time')
# canvas1 = FigureCanvas(f1)
#
# f2 = Figure(figsize=(5, 4))
# a2 = f2.add_subplot(111)
# p2, = a2.plot(distribution, '-b')
# a2.set_xlabel('Number of infected nodes', labelpad=0)
# a2.set_ylabel('Frequency', labelpad=0)
# a2.set_title('Density of number of infected nodes')
# canvas2 = FigureCanvas(f2)


# This function will be called repeatedly by the GTK+ main loop, and we use it
# to update the state according to the SIRS dynamics.
def update_state():
    for n in newly_infected_list:
        n.a = False
    global time
    global simulation

    # visit the nodes in random order
    for i, g in enumerate(simulation_list):
        if i < len(graph_list):
            for e in g.edges():
                edge_state_list[i][e] = [0.8, 0.8, 0.8, 1]
        newState = state_list[i].copy()
        for v in g.vertices():
            if state_list[i][v] == I:
                if random() < r:
                    if model_list[i] == 'SIS':
                        newState[v] = S
                    else:
                        newState[v] = R
                    num_infected_list[i] -= 1
            elif state_list[i][v] == E:
                if random() < incubation:
                    newState[v] = I
            elif state_list[i][v] == S:
                ns = list(v.all_neighbors())
                p = 0
                for neighbor in ns:
                    if state_list[i][neighbor] == I:
                        p += 1
                p *= beta
                if random() < p:
                    if model_list[i] == 'SEIR':
                        newState[v] = E
                    else:
                        newState[v] = I
                    newly_infected_list[i][v] = True
                    num_infected_list[i] += 1
                    if i < len(graph_list):
                        for e in v.all_edges():
                            edge_state_list[i][e] = [1, 165 / 255, 0, 1]
            elif state_list[i][v] == R:
                if model_list[i] == 'SIRS':
                    if random() < s:
                        newState[v] = S
        state_list[i].swap(newState)
        frequency_list[i].append(num_infected_list[i] / g.num_vertices())
        log_list[i].append(math.log1p(frequency_list[i][-1]) - math.log1p(frequency_list[i][-2]))
        distribution_list[i][num_infected_list[i]] += 1
    time += 1
    # Filter out the recovered vertices
    # g.set_vertex_filter(removed, inverted=True)

    # The following will force the re-drawing of the graph, and issue a
    # re-drawing of the GTK window.
    if offscreen:
        pass
    # win.graph.regenerate_surface()
    # win.graph.queue_draw()
    # a1.plot(frequency, colors[simulation % colors.__len__()])
    # p2.set_ydata([x / time for x in distribution])
    # canvas1.draw()
    # canvas2.draw()
    else:
        for i, graph in enumerate(win.graphs):
            graph_draw(graph_list[i], pos=layout_list[i],
                       vprops={'fill_color': state_list[i], 'halo': newly_infected_list[i], 'halo_color': [1, 0, 1, 1]},
                       eprops={'color': edge_state_list[i]}, output=str(i) + '.png',
                       output_size=(400, 400))
            graph.set_from_file(str(i) + '.png')
        update_data(win.a11, '', 'Percentage(SIS)', 'Percentage of Infected Nodes (Time Series)',
                    frequency_list[0::2])
        update_data(win.a12, 'Time', 'Percentage(SIRS)', '',
                    frequency_list[1::2])
        update_data(win.a21, '', 'Frequency(SIS)',
                    'Frequency Density (PDF) ',
                    [[x / time for x in distribution_list[i]] for i in range(len(distribution_list)) if i % 2 == 0])
        update_data(win.a22, 'Percentage of infected nodes', 'Frequency(SIRS)',
                    '',
                    [[x / time for x in distribution_list[i]] for i in
                     range(len(distribution_list)) if i % 2 != 0])
        for i in range(len(frequency_list)):
            if i % 2 == 0:
                win.a21.axvline((100 * np.array(frequency_list[i])).mean(), color=colors[int(i / 2)])
                win.a21.axvline(np.median(100 * np.array(frequency_list[i])), color=colors[int(i / 2)])
            else:
                win.a22.axvline((100 * np.array(frequency_list[i])).mean(), color=colors[int((i - 1) / 2)])
                win.a22.axvline(np.median(100 * np.array(frequency_list[i])), color=colors[int((i - 1) / 2)])
        update_data(win.a41, 'time', 'Log-Return', 'Log-Return & Prediction of SIS', log_list[0::2])
        update_data(win.a51, 'time', 'Log-Return', 'Log-Return & Prediction of SIRS', log_list[1::2])

        if time > 2:
            win.a31.clear()
            for i, log in enumerate(log_list[0::2]):
                sm.graphics.tsa.plot_acf((np.array(log) ** 2), win.a31, c=colors[i], markersize=4)
            win.a31.set_title('ACF of log-return squared', {'fontweight': 'bold'})
            win.a31.set_ylabel('(SIS)', {'fontweight': 'bold'})
            win.a31.grid()

            win.a32.clear()
            for i, log in enumerate(log_list[1::2]):
                sm.graphics.tsa.plot_acf((np.array(log) ** 2), win.a32, c=colors[i], markersize=4)
            win.a32.set_title('')
            win.a32.set_ylabel('(SIRS)', {'fontweight': 'bold'})
            win.a32.grid()
            win.a61.clear()
            for i,d in enumerate(distribution_list[0::2]):
                sm.qqplot(np.array(d), line='45', ax=win.a61, c=colors[i], markersize=4)
            win.a61.set_title('PDF (SIS)', {'fontweight': 'bold'})
            win.a61.set_xlabel('')
            win.a61.set_ylabel('')
            win.a61.grid()

            win.a62.clear()
            for i, d in enumerate(distribution_list[1::2]):
                sm.qqplot(np.array(d), line='45', ax=win.a62, c=colors[i], markersize=4)
            win.a62.set_title('PDF (SIRS)', {'fontweight': 'bold'})
            win.a62.set_xlabel('')
            win.a62.set_ylabel('')
            win.a62.grid()

            win.a63.clear()
            for i,l in enumerate(log_list[0::2]):
                sm.qqplot(np.array(l), line='45', ax=win.a63, c=colors[i], markersize=4)
            win.a63.set_title('log-return (SIS)', {'fontweight': 'bold'})
            win.a63.set_ylabel('')
            win.a63.set_xlabel('')
            win.a63.grid()

            win.a64.clear()
            for i, l in enumerate(log_list[1::2]):
                sm.qqplot(np.array(l), line='45', ax=win.a64, c=colors[i], markersize=4)
            win.a64.set_title('log-return (SIRS)', {'fontweight': 'bold'})
            win.a64.set_ylabel('')
            win.a64.set_xlabel('')
            win.a64.grid()

            for i, log in enumerate(log_list):
                if log[-1]:
                    print('################'+str(time)+'######################')
                    prediction = arch_model(100*(np.array(log[:-1]))).fit().forecast(horizon=5).mean.at[time-3,'h.1']
                    error_list[i].append(abs((prediction/100 - log[-1]) / log[-1]))

            update_data(win.a42, 'error', 'frequency', 'Prediction', error_list[0::2])
            update_data(win.a52, 'error', 'frequency', 'Prediction', error_list[1::2])

        win.canvas1.draw()
        win.canvas2.draw()
        win.canvas3.draw()
        win.canvas4.draw()
        win.canvas5.draw()
        win.canvas6.draw()

    # if doing an offscreen animation, dump frame to disk
    if offscreen:
        global count
        pixbuf = win.get_pixbuf()
        pixbuf.savev(r'./frames/%06dgraph.png' % count, 'png', [], [])
        # canvas1.print_png(r'./frames/%06dplot1.png' % count)
        # canvas2.print_png(r'./frames/%06dplot2.png' % count)
        if count > max_count:
            sys.exit(0)
        count += 1

    # We need to return True so that the main loop will call this function more
    # than once.

    # if time == time_to_stop:
    # if not simulation:
    #     # m, n = max_cardinality_matching(g)
    #     # vs.a = False
    #     # for e in g.edges():
    #     #     if m[e]:
    #     #         vs[e.source()] = vs[e.target()] = True
    #     m = min_spanning_tree(g)
    # # newm = m.copy()
    # # shuffle(elist)
    # # g.set_edge_filter(m)
    # # comp, hist = label_components(g)
    # # rep = comp.a
    # # for e in elist:
    # #     if (find_rep(rep, g.vertex_index[e.source()]) != find_rep(rep, g.vertex_index[e.target()])):
    # #         newm[e] = True
    # #         rep[g.vertex_index[e.target()]] = g.vertex_index[e.source()]
    # # g.set_edge_filter(newm)
    # g.set_edge_filter(m)
    #
    # for v in g.vertices():
    #     state[v] = S
    # vt = list(g.vertices())
    # sp = sample(vt, number_of_infected_at_beginning)
    # for p in sp:
    #     state[p] = I
    # error.clear()
    # distribution = [0] * (size + 1)
    # num_infected = number_of_infected_at_beginning
    # frequency = [num_infected / size]
    # simulation += 1
    # time = 0
    if time == time_to_stop:
        return False
    return True


# Bind the function above as an 'idle' callback.
cid = GObject.idle_add(update_state)

# We will give the user the ability to stop the program by closing the window.
win.connect("delete_event", Gtk.main_quit)

# Actually show the window, and start the main loop.
win.show_all()
Gtk.main()
