# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torch import nn, optim
import numpy as np
import time
import tkinter
from PIL import Image, ImageTk


class ForestFire(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self):
        super(ForestFire, self).__init__()

        self.neighborhood = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        #weights = torch.tensor([[1., 1., 1.],
        #                        [1., 0., 1.],
        #                        [1., 1., 1.]])
        weights = torch.tensor([[0., 1., 0.],
                                [1., 0., 1.],
                                [0., 1., 0.]])
        weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        with torch.no_grad():
            self.neighborhood.weight = nn.Parameter(weights)

        # self.p_fire = p_fire
        # self.g_tree = g_tree
        # self.random_fire = nn.Dropout(p=p_fire)  # dropout layers get deactivated in eval mode
        # self.grow_tree = nn.Dropout(p=g_tree)

    def forward(self, tree, fire, g_tree, p_fire):
        fire_old = fire
        fire_neighbor = self.neighborhood(fire)  # calculate neighborhood to fire
        fire_neighbor = torch.clamp(fire_neighbor, -0.01, 1.01)  # normalize neighborhood
        fire_new = torch.mul(fire_neighbor, tree)  # tree next to fire catch fire themselves
        fire_new = torch.round(fire_new)

        tree_new = torch.subtract(tree, fire_new)  # remove burning trees
        # tree_rand_fire = self.random_fire(tree_new)  # same as below but with dropout layer
        tree_rand_fire = torch.nn.functional.dropout(tree_new,
                                                     p=p_fire)  # randomly remove empty fields, with p=grow_tree
        tree_rand_fire = torch.mul(tree_rand_fire, (1 - p_fire))  # inverse dropout normalization
        rand_fire = torch.subtract(tree_new, tree_rand_fire)  # removed trees are on fire
        fire_new = torch.add(fire_new, rand_fire)  # fire propagation and random fires combined
        tree_new = torch.subtract(tree_new, rand_fire)  # remove burning trees

        empty = torch.add(tree, fire)
        empty = torch.subtract(empty, 1)  # all -1 are empty fields
        # empty_rand_grow = self.grow_tree(empty) # same as below but with dropout layer
        empty_rand_grow = torch.nn.functional.dropout(empty, p=g_tree)  # randomly remove empty fields, with p=grow_tree
        empty_rand_grow = torch.mul(empty_rand_grow, (1 - g_tree))  # inverse dropout normalization
        rand_grow = torch.subtract(empty_rand_grow, empty)  # note: subtraction of minus values results in positive
        tree_new = torch.add(tree_new, rand_grow)  # add randomly grown trees

        tree_new = torch.round(tree_new)  # correct floating point errors that come with
        fire_new = torch.round(fire_new)  # torch functions
        return tree_new, fire_new


def burn_x_and_y(event):
    global burnx, burny
    burnx, burny = event.x, event.y


def plant_x_and_y(event):
    global plantx, planty
    plantx, planty = event.x, event.y


def makeform(root, fields):
    entries = {}
    for field in fields:
        row = tkinter.Frame(root)
        lab = tkinter.Label(row, width=22, text=field + ": ", anchor='w')
        ent = tkinter.Entry(row)
        ent.insert(0, "0")
        row.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        lab.pack(side=tkinter.LEFT)
        ent.pack(side=tkinter.RIGHT, expand=tkinter.YES, fill=tkinter.X)
        entries[field] = ent
    return entries


def set_param(entries):
    global p_fire, g_tree
    p_fire = float(entries['p_fire'].get())
    g_tree = float(entries['g_tree'].get())


def on_closing():
    global run_program
    print("Program closed")
    run_program = False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # init global variables for communication between windows and main program
    global run_program
    run_program = True
    global burnx, burny, plantx, planty
    burnx, burny, plantx, planty = -1, -1, -1, -1
    global p_fire, g_tree
    g_tree, p_fire = 0, 0

    # choose CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"  # force CPU
    if device == "cpu":
        torch.set_num_threads(16)  # parallelization on CPU
    print(device)

    # setup of game field
    num_rows = 1024

    tree_array = []
    tree_data = np.ones((num_rows, num_rows))
    tree_array.append(np.asarray(tree_data))
    tree_array = np.asarray(tree_array)
    tree = torch.from_numpy(tree_array)
    tree = tree.unsqueeze(0).float().to(device)

    fire_array = []
    fire_data = np.zeros((num_rows, num_rows))
    fire_array.append(np.asarray(fire_data))
    fire_array = np.asarray(fire_array)
    fire = torch.from_numpy(fire_array)
    fire = fire.unsqueeze(0).float().to(device)

    b = np.zeros((num_rows, num_rows))

    # setup pseudo NN
    ForestFire = ForestFire().to(device)
    ForestFire.eval()

    # image window
    app = tkinter.Tk()
    app.geometry(str(num_rows) + "x" + str(num_rows+98))  # hack to fit window size
    canvas = tkinter.Canvas(app, bg='black')
    canvas.pack(anchor='nw', fill='both', expand=1)
    canvas.bind("<Button-1>", burn_x_and_y)
    app.protocol("WM_DELETE_WINDOW", on_closing)

    # control panel, attached to window
    fields = ('p_fire', 'g_tree')
    ents = makeform(app, fields)
    app.bind('<Return>', (lambda event, e=ents: tkinter.fetch(e)))
    b1 = tkinter.Button(app, text='Set Parameter',
                        command=(lambda e=ents: set_param(e)))
    b1.pack(side=tkinter.LEFT, padx=5, pady=5)
    b2 = tkinter.Button(app, text='Quit', command=on_closing)
    b2.pack(side=tkinter.LEFT, padx=5, pady=5)

    app.update_idletasks()
    app.update()

    last_time = time.time()
    imax = 10
    i = imax - 2
    while run_program:

        # retrieve image and plot to window
        g = tree.detach().cpu().numpy()[0] * 150  # better than 255
        r = fire.detach().cpu().numpy()[0] * 255
        rgb = np.dstack((r[0], g[0], b))
        rgb = rgb.astype(np.uint8)
        # images.append(rgb)
        image = Image.fromarray(rgb)
        image = image.resize((num_rows, num_rows), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image=image, anchor='nw')
        app.update_idletasks()
        app.update()

        # query mouse input
        # x and y are flipped between array and window
        if burnx > 0:
            #print(str(burnx) + " , "  + str(burny))
            tree[0, 0, burny, burnx] = 0
            fire[0, 0, burny, burnx] = 1
            burnx, burny = -1, -1

        i += 1
        if i == imax:
            print("%s fps ---" % (imax / (time.time() - last_time)))
            last_time = time.time()
            i = 0

        # calulate new game field
        with torch.no_grad():  # otherwise the calculated gradient exceeds GPU memory
            tree, fire = ForestFire(tree=tree, fire=fire, g_tree=g_tree, p_fire=p_fire)

    # pure performance run without display / other overhead
    #last_time = time.time()
    #imax = 1000  # with 1000 -> 320 fps
    #for i in range(imax):
    #    with torch.no_grad():  # otherwise the calculated gradient exceeds GPU memory
    #        tree, fire = ForestFire(tree=tree, fire=fire, g_tree=0.01, p_fire=0.001)
    #print("%s fps ---" % (imax/(time.time() - last_time)))