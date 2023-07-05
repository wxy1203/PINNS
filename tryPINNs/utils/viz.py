from matplotlib import pyplot as plt
import numpy as np
import os


def viz(u, pred, domain, path=None):
    # plot

    plt.figure(figsize=(14, 7))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(2, 2, 1)
    plt.imshow(u, interpolation='nearest', cmap='seismic',
               extent=list(domain[1]) + list(domain[0]), origin='lower', aspect='auto')
    plt.colorbar()

    plt.xlabel('$y$')
    plt.ylabel('$x$')
    plt.title('$Exact: u$', fontsize=10)

    plt.subplot(2, 2, 3)
    plt.imshow(pred, interpolation='nearest', cmap='seismic',
               extent=list(domain[1]) + list(domain[0]), origin='lower', aspect='auto')
    plt.colorbar()

    xmin, xmax = domain[0]
    line = np.linspace(xmin, xmax, 2)[:, None]
    ymin, ymax = domain[1]
    plt.plot((0.25 * (ymax - ymin) + ymin) * np.ones((2, 1)), line, 'w-', linewidth=1)
    plt.plot((0.5 * (ymax - ymin) + ymin) * np.ones((2, 1)), line, 'w-', linewidth=1)
    plt.plot((0.75 * (ymax - ymin) + ymin) * np.ones((2, 1)), line, 'w-', linewidth=1)
    plt.xlabel('$y$')
    plt.ylabel('$x$')
    plt.title('$Prediction: u$', fontsize=10)

    delta = 1.0 / u.shape[0]
    x = np.array([(i + 0.5) * delta for i in range(u.shape[0])]) * (xmax - xmin) + xmin
    pos = [3, 4, 7, 8]

    for i in range(4):
        plt.subplot(2, 4, pos[i])
        plt.plot(x, u[:, int(i / 4 * u.shape[1])], 'b-', linewidth=2, label='Exact')
        plt.plot(x, pred[:, int(i / 4 * u.shape[1])], 'r--', linewidth=2, label='Prediction')
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        plt.title('$y = {}$'.format(round(i / 4 * (ymax - ymin) + ymin, 2)), fontsize=10)
        plt.xlim([xmin, xmax])
        plt.ylim([u.min(), u.max()])
        plt.legend(loc='upper right')

    plt.tight_layout()
    if path is not None:
        plt.savefig(os.path.join(path, 'viz.pdf'))
    plt.show()


def error_plot_pinn(u, pred, domain, path):
    plt.figure()
    pinn_error = abs(u-pred)
    plt.imshow(pinn_error, interpolation='nearest', cmap='seismic',
               extent=list(domain[1]) + list(domain[0]), origin='lower', aspect='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Error')
    plt.colorbar()
    plt.savefig(os.path.join(path, 'error.pdf'))
