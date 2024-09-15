import matplotlib.pyplot as plt
import seaborn as sns

def draw_hist(data, bins=10, title='', xlabel='', ylabel='', color='blue', alpha=0.75, save_path=None):
    """
    Draw a histogram.
    
    Parameters:
    -----------
    data: list, array-like
        The data to be plotted.
    bins: int, optional (default=10)
        The number of bins.
    title: str, optional (default='')
        The title of the plot.
    xlabel: str, optional (default='')
        The label of the x-axis.
    ylabel: str, optional (default='')
        The label of the y-axis.
    color: str, optional (default='blue')
        The color of the bars.
    alpha: float, optional (default=0.75)
        The transparency of the bars.
    save_path: str, optional (default=None)
        The path to save the plot.
    """
    sns.histplot(data, bins=bins, color=color, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
