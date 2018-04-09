import matplotlib.animation
import matplotlib.pyplot as plt
import numpy


def plot_log(data, x, y, ax, **kwargs):
    x_ = []
    y_ = []
    for i in data:
        if x in i.keys() and y in i.keys():
            x_.append(i[x])
            y_.append(i[y])
    return ax.plot(x_, y_, **kwargs)


class Plotter(object):
    def __init__(self, out='tmp.pdf', loc=111):
        self._loc = loc
        self._out = out
        self._fig = None
        self._ax = None

    def __enter__(self):
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(self._loc)
        return self._fig, self._ax

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self._loc = None
            self._out = None
            self._fig = None
            self._ax = None
            return False
        self._fig.savefig(self._out)
        plt.close(self._fig)
        return True


def hist(array, fig, ax, **kwargs):
    return _HistFunction()(array, fig, ax, **kwargs)


def matshow(array, fig, ax, **kwargs):
    return _MatshowFunction()(array, fig, ax, **kwargs)


def hist_animation(arrays, fig, out='hist.gif', **kwargs):
    return animation(_HistFunction(), arrays, fig, out, **kwargs)


def matshow_animation(arrays, fig, out='matshow.gif', **kwargs):
    return animation(_MatshowFunction(), arrays, fig, out, **kwargs)


def animation(func, arrays, fig, out, **kwargs):
    artists = list()
    ax = fig.add_subplot(1, 1, 1)
    func.initialize(fig)
    for array in arrays:
        artists.append(func(array, fig, ax, **kwargs))
    func.finalize(fig)
    ret = matplotlib.animation.ArtistAnimation(fig, artists, interval=200)
    ret.save(out, writer='imagemagick')
    return ret


class _Function(object):
    def initialize(self, fig):
        pass

    def __call__(self, array, fig, ax, **kwargs):
        raise NotImplementedError

    def finalize(self, fig):
        pass


class _HistFunction(_Function):
    def __call__(self, array, fig, ax, **kwargs):
        hist, bin_edges = numpy.histogram(array.flatten(), bins=128)
        x = numpy.array([bin_edges[:-1], bin_edges[1:]]).T.flatten()
        y = numpy.array([hist, hist]).T.flatten()
        return ax.plot(x, y, **kwargs)


class _MatshowFunction(_Function):
    def __call__(self, array, fig, ax, **kwargs):
        if len(array.shape) == 1:
            array_ = [array, array]
        elif len(array.shape) == 2:
            array_ = array
        elif len(array.shape) == 4:
            out_channles, in_channles, kh, kw = array.shape
            h = kh * in_channles
            w = kw * out_channles
            array_ = numpy.zeros((h, w))
            for i in range(in_channles):
                for j in range(out_channles):
                    array_[i*kh:(i+1)*kh, j*kw:(j+1)*kw] = array[j, i, :, :]
        else:
            raise NotImplementedError('Array shape {} not supported.'
                                      .format(array.shape))
        cax = ax.matshow(array_, cmap='bwr', vmin=-0.2, vmax=0.2, **kwargs)
        ax.get_xaxis().set_tick_params(
            which='both', bottom=False, top=False, left=False, right=False)
        ax.get_yaxis().set_tick_params(
            which='both', bottom=False, top=False, left=False, right=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return [cax]
