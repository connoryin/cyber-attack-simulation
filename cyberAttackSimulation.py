from graph_tool.all import *
from numpy.random import *
import numpy as np
from random import sample, gauss
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
import sys, os, os.path
import time, math

seed()

from gi.repository import Gtk, Gdk, GdkPixbuf, GObject

# parameters:
time_to_stop = 100
colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']
simulation = 0


def create_plot(xlabel, ylabel, title, data):
    f = Figure(figsize=(5, 4))
    a = f.add_subplot(111)
    num = 0
    for d, l in data:
        a.plot(d, colors[num], label=l)
        num += 1
        if num >= len(colors):
            num = 0
    a.set_xlabel(xlabel, labelpad=0)
    a.set_ylabel(ylabel, labelpad=0)
    a.set_title(title)
    a.legend()
    canvas = FigureCanvas(f)
    return a, canvas


def create_graph(type, size=100,
                 p_for_random_graph=0.1,
                 mean_for_homogeneous_graph=6,
                 standard_deviation_for_homogeneous_graph=0.25):
    g = Graph(directed=False)
    if type == 'price':
        g = price_network(size)
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
    for d, l in data:
        figure.plot(d, colors[num], label=l)
        num += 1
        if num >= len(colors):
            num = 0
    figure.set_xlabel(xlabel, labelpad=0)
    figure.set_ylabel(ylabel, labelpad=0)
    figure.set_title(title)
    figure.legend()

r = 0.1  # I->S probability
s = 0.1  # R->S probability
beta = 0.05
incubation = 0.5  # E->I probability
number_of_infected_at_beginning = 5

graph_list = []
g = create_graph('random')
graph_list.append(g)
mst = min_spanning_tree(g)
mstg = GraphView(g, efilt=mst, directed=False)
mstg = Graph(mstg, prune=True)
graph_list.append(mstg)
graph_list.append(create_graph('star'))
graph_list.append(create_graph('cluster'))


model_list = ['SIS', 'SIS', 'SIS', 'SIS']
label_list = ['Random', 'MST', 'Star', 'Cluster']

S = [0, 1, 0, 1]  # Green color
I = [1, 0, 0, 1]  # Red color
R = [0, 1, 1, 1]  # Blue color
E = [1, 0, 1, 1]  # Purple color

state_list = []
for g in graph_list:
    state_list.append(g.new_vertex_property("vector<double>"))
for i, g in enumerate(graph_list):
    for v in g.vertices():
        state_list[i][v] = S
    vt = list(g.vertices())
    sp = sample(vt, number_of_infected_at_beginning)
    for s in sp:
        state_list[i][s] = I

frequency_list = []
for g in graph_list:
    frequency_list.append([number_of_infected_at_beginning / g.num_vertices()])
distribution_list = []
for g in graph_list:
    distribution_list.append([0] * (g.num_vertices() + 1))
num_infected_list = [number_of_infected_at_beginning] * len(graph_list)
newly_infected_list = []
for g in graph_list:
    newly_infected_list.append(g.new_vertex_property("bool"))

time = 1

# If True, the frames will be dumped to disk as images.
offscreen = sys.argv[1] == "offscreen" if len(sys.argv) > 1 else False
max_count = 20
if offscreen and not os.path.exists("../frames"):
    os.mkdir("../frames")


class SimulationWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title='Cyber Attack Simulation')
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.box = Gtk.Box()
        self.box2 = Gtk.Box()
        self.add(self.vbox)
        self.graphs = []
        for i, g in enumerate(graph_list):
            self.graphs.append(GraphWidget(g, sfdp_layout(g), edge_color=[0.6, 0.6, 0.6, 1],
                                     vertex_fill_color=state_list[i],
                                     vertex_halo=newly_infected_list[i],
                                     vertex_halo_color=[0.8, 0, 0, 0.6]))
            f = Gtk.Frame()
            f.set_label(label_list[i])
            f.add(self.graphs[i])
            f.set_label_align(.5, .5)
            f.set_shadow_type(0)
            self.box2.pack_start(f, True, True, 0)

        # self.f3 = Figure(figsize=(5, 4))
        # self.a3 = self.f3.add_subplot(111)
        # self.a3.plot(error, '-b')
        # self.canvas3 = FigureCanvas(self.f3)

        self.vbox.pack_start(self.box2, True, True, 0)

        self.a1, self.canvas1 = create_plot('Time', 'Proportion', 'Proportion of infected nodes vs. time',
                                            zip(frequency_list, label_list))
        self.box.pack_start(self.canvas1, True, True, 0)

        self.a2, self.canvas2 = create_plot('Number of infected nodes', 'Frequency',
                                            'Density of number of infected nodes',
                                            zip([[x / time for x in distribution_list[i]] for i in
                                                 range(len(distribution_list))], label_list))
        self.box.pack_start(self.canvas2, True, True, 0)

        self.vbox.pack_start(self.box, True, True, 0)

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
    for i, g in enumerate(graph_list):
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
            elif state_list[i][v] == R:
                if model_list[i] == 'SIRS':
                    if random() < s:
                        newState[v] = S

        state_list[i].swap(newState)
        frequency_list[i].append(num_infected_list[i] / g.num_vertices())
        distribution_list[i][num_infected_list[i]] += 1
        # error.append(math.log1p(frequency[-1]) - math.log1p(frequency[-2]))
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
        for graph in win.graphs:
            graph.regenerate_surface()
            graph.queue_draw()
        update_data(win.a1, 'Time', 'Proportion', 'Proportion of infected nodes vs. time', zip(frequency_list, label_list))
        update_data(win.a2, 'Number of infected nodes', 'Frequency',
                    'Density of number of infected nodes',
                    zip([[x / time for x in distribution_list[i]] for i in range(len(distribution_list))], label_list))
        # win.a3.plot(error, colors[simulation % colors.__len__()])
        win.canvas1.draw()
        win.canvas2.draw()
        # win.canvas3.draw()

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

    return True


# Bind the function above as an 'idle' callback.
cid = GObject.idle_add(update_state)

# We will give the user the ability to stop the program by closing the window.
win.connect("delete_event", Gtk.main_quit)

# Actually show the window, and start the main loop.
win.show_all()
Gtk.main()
