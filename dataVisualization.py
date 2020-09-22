import matplotlib.pyplot as plt


def createHistogram(arousal, valence, directory, name):

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(arousal, bins=20)
    axs[1].hist(valence, bins=20)

    plt.savefig(directory + "/_"+str(name)+"_dataDistribution_.png")

    plt.clf()