
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm,colors
from sklearn import  manifold
from sklearn.decomposition import PCA
import umap

#Load the MNIST dataset
def load_idx(filename):
     with open(filename,"rb") as f:
        zero, datatype, dims = struct.unpack(">HBB", f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

#3D plot
def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()

#2D plot
def plot_2d(points, points_color, title,cmap):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=24)
    add_2d_scatter(ax, points, points_color,cmap)
    norm = colors.Normalize(min(points_color), max(points_color))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    plt.show()

#Create a 2D scatter plot
def add_2d_scatter(ax, points, points_color,cmap):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=20, cmap=cmap)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

#Run & plot U Map
def u_map(X,y,title='U-Map',cmap='Paired',n_neighbors=15,min_dist=0.1,random_state=0):
    fit = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,metric='euclidean',random_state=random_state)
    S_umap = fit.fit_transform(X)
    plot_2d(S_umap, y, title,cmap)

#Run & plot tSNE
def tsne(X,y,title='U-Map',cmap='Paired',perplexity=40,max_iter=300,random_state=0):
    t_sne = manifold.TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        max_iter=max_iter,
        random_state=random_state,
    )

    S_tsne = t_sne.fit_transform(X)
    plot_2d(S_tsne, y, title,cmap)

#Run & plot PCA
def pca(X,y,title='PCA',cmap='Paired'):
    pca_model = PCA(n_components=2)
    pca_model.fit(X)
    S_pca = pca_model.transform(X)
    plot_2d(S_pca, y, title,cmap)