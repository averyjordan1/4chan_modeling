import matplotlib.pyplot as plt
from chan_modeling.archiving import get_single_vector


def plot_word(word, plt, cosine_score, y, count):
    # draw a point on the line

    px = (10 * (cosine_score - .5) * 2)
    plt.plot(px, y, 'ro', ms=15, mfc='r')

    align = 'right'
    # add an arrow
    annotation_y = 2
    # adjust up or down
    if count % 2 == 0:
        annotation_y = -2

    plt.annotate("{} @ {}".format(word, round(cosine_score, 2)), (px, y), xytext=(px - .1, y + annotation_y),
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 horizontalalignment='right', rotation=45)


def plot_similarity_vector(vector, main_word):
    # set up the figure
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)

    # draw lines
    xmin = 0
    xmax = 10
    y = 5
    height = 1

    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2., y + height / 2.)
    plt.vlines(xmax, y - height / 2., y + height / 2.)

    for count, component in enumerate(vector):
        plot_word(component[0], plt, component[1], y, count)

    # add numbers
    plt.text(xmin - 0.01, y, '.5', horizontalalignment='right')
    plt.text(xmax + 0.01, y, '1', horizontalalignment='left')

    plt.axis('off')
    plt.show()

def plot_vector(word, model):
    vec = get_single_vector(word, model)
    plot_similarity_vector(vec, word)