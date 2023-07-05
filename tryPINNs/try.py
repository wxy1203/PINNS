import numpy as np
import torch
import argparse
from tqdm import tqdm

from utils.data import load_data
from utils.others import set_seed
from utils.logger import Logger
from utils.viz import viz, error_plot_pinn
from model.model import FcNet


class Burgers(object):
    def __init__(self, path='burgers.mat'):
        self.data, self.domain, self.sol = load_data(path)
        self.input_dim = len(self.domain)
        self.params = {'nu': 0.01 / np.pi}

    def initial_term(self, x):
        """
        Calculate the labels for the initial term
        :param x: input of shape (n_i, input_dim)
        :return: labels of shape (n_i, 1)

        TODO: you can modify everything in this code block
        HINT: the initial condition is u(x, t=0) = -sin(pi * x)
        """
        # begin your code
        return torch.zeros((x.shape[0], 1))
        # end your code

    def boundary_term(self, x):
        """
        Calculate the labels for the boundary term
        :param x: input of shape (n_b, input_dim)
        :return: labels of shape (n_b, 1)
        """
        return torch.zeros((x.shape[0], 1))

    def force_term(self, x):
        """
        Calculate the labels for force term
        :param x: input of shape (n_f, input_dim)
        :return: labels of shape (n_f, 1)
        """
        return torch.zeros((x.shape[0], 1))

    def f(self, x, model):
        """
        Define the computational graph of residual term given a model
        :param x: input of shape (n_f, input_dim)
        :param model: differentiable model takes x as input and will output a tensor of shape (n_f, 1) as the prediction
        of the solution u(x).
        :return: the prediction of the residual term

        TODO: you can modify everything in this code block
        HINT: You can use torch.autograd.grad
        HINT: Burgers Eq: u_t + u * u_x = nu * u_xx
        """
        nu = self.params['nu']
        # begin your code
        return torch.zeros((x.shape[0], 1))
        # end your code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--hidden_size', type=int, default=20, help='size of hidden layers')
    parser.add_argument('--depth', type=int, default=8, help='depth of the network')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # PINN params
    parser.add_argument('--n_i', type=int, default=100, help='number of datapoints for initial condition')
    parser.add_argument('--lamb_i', type=float, default=1., help='weight of loss_i')
    parser.add_argument('--n_b', type=int, default=100, help='number of datapoints for boundary condition')
    parser.add_argument('--lamb_b', type=float, default=1., help='weight of loss_b')
    parser.add_argument('--n_f', type=int, default=10000, help='number of datapoints for force (residual) condition')
    parser.add_argument('--lamb_f', type=float, default=1., help='weight of loss_f')
    parser.add_argument('--n_t', type=int, default=1000, help='number of datapoints for test')
    # Optimization params - Adam
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--steps', type=int, default=20000, help='number of adam steps')
    parser.add_argument('--steps_per_eval', type=int, default=1000, help='number of adam steps per evaluation')
    # Others
    parser.add_argument('--device', type=str, default="cpu", help='device, cpu/cuda/cuda:{id}')
    parser.add_argument('--root_dir', type=str, default="logs", help='root dir for logger')

    args = parser.parse_args()
    exp_name = 'burgers'
    exp_name += '_seed={}'.format(args.seed)
    logger = Logger(exp_name, root_dir=args.root_dir, with_timestamp=True)
    logger.add_params(vars(args))
    set_seed(args.seed)
    device = torch.device(args.device)

    # PDE
    pde = Burgers()

    # model
    model = FcNet(db=args.hidden_size, depth=args.depth, activation='tanh', dx=len(pde.domain), dy=1)
    model.to(device)

    # data
    x_i = np.random.random([args.n_i, 1]) * 2 - 1.
    t_i = np.zeros_like(x_i)
    input_i = torch.from_numpy(np.concatenate([x_i, t_i], axis=-1)).float().to(device)
    u_i = pde.initial_term(input_i).float().to(device)

    x_b = (np.random.random((args.n_b, 1)) > 0.5) * 2 - 1.
    t_b = np.random.random([args.n_b, 1])
    input_b = torch.from_numpy(np.concatenate([x_b, t_b], axis=-1)).float().to(device)
    u_b = pde.boundary_term(input_b).float().to(device)

    x_f = np.random.random([args.n_f, 1]) * 2 - 1.
    t_f = np.random.random([args.n_f, 1])
    input_f = torch.from_numpy(np.concatenate([x_f, t_f], axis=-1)).float().to(device)
    f = pde.force_term(input_f).float().to(device)

    input_t = torch.from_numpy(pde.data[:, :-1]).float().to(device)
    u_t = torch.from_numpy(pde.data[:, -1:]).float().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training
    for adam_step in tqdm(range(1, args.steps + 1)):
        with logger.measure_time('adam'):
            model.train()
            optimizer.zero_grad()

            # initial term
            # TODO: calculate the initial term loss_i
            # begin your code
            loss_i = torch.zeros(1)
            # end your code

            # boundary term
            logger.add_metric('loss_i', loss_i.item())
            loss_b = torch.mean((model(input_b) - u_b) ** 2)
            logger.add_metric('loss_b', loss_b.item())

            # residual term
            # TODO: calculate the residual term loss_f
            # begin your code
            loss_f = torch.zeros(1)
            # end your code
            logger.add_metric('loss_f', loss_f.item())

            loss = args.lamb_i * loss_i + args.lamb_b * loss_b + args.lamb_f * loss_f
            logger.add_metric('loss', loss.item())

            loss.backward()
            optimizer.step()
        if adam_step % args.steps_per_eval == 0:
            model.eval()
            pred = model(input_t)
            rel_l2_error = torch.norm(model(input_t) - u_t, p=2) / torch.norm(u_t, p=2)
            logger.add_metric('rel_l2_error', rel_l2_error.item())
            logger.commit(epoch=adam_step // args.steps_per_eval, step=adam_step)

    x, u = pde.data[:, :-1], pde.sol
    x = torch.from_numpy(x).float().to(device)
    pred = model(x).detach().cpu().numpy().reshape(u.shape)
    viz(u=u.T, pred=pred.T, domain=pde.domain, path=logger.logdir)
    error_plot_pinn(u=u.T, pred=pred.T, domain=pde.domain, path=logger.logdir)
