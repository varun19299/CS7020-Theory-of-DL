import streamlit as st

import jax.numpy as np
from jax import random
from jax.experimental import optimizers
from jax.api import jit, grad, vmap

import functools
from functools import partial
import neural_tangents as nt
from neural_tangents import stax

from scipy import signal
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

key = random.PRNGKey(10)

# Titles
st.title("Neural Tanget Kernels and NN Gausian Processes Explored")

"""
**Author: Varun Sundar, @varun19299.**  
**Date: 24th November, 2019.**
"""


"""
In this post, we shall explore two recent advances
in the theoretical understanding of infinitely wide neural networks (NNs):
(a) the Neural Tanget Kernel (NTK),
b) Neural Networks as a Gaussian Process (NNGP).

My two-cents while running this notebook: *Start with smaller sizes, steps, ensembles: 
unless you have really good hardware, try not to max out these sliders! 
This work was done as a part of the capstone project for
 CS7020, Advances in the theory of Deep Learning.*
"""


sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})


def format_plot(x=None, y=None):
    # plt.grid(False)
    ax = plt.gca()
    if x is not None:
        plt.xlabel(x, fontsize=20)
    if y is not None:
        plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1],
    )
    plt.tight_layout()


legend = functools.partial(plt.legend, fontsize=10)


def plot_fn(train, test, *fs, xlims=(-np.pi, np.pi), ylims=(-1.5, 1.5)):
    train_xs, train_ys = train

    plt.plot(train_xs, train_ys, "ro", markersize=10, label="train")

    if test != None:
        test_xs, test_ys = test
        plt.plot(test_xs, test_ys, "k--", linewidth=3, label="$f(x)$")

        for f in fs:
            plt.plot(test_xs, f(test_xs), "-", linewidth=3)

    plt.xlim(xlims)
    plt.ylim(ylims)

    format_plot("$x$", "$f$")


def loss_fn(predict_fn, ys, t):
    mean, var = predict_fn(t)
    mean = np.reshape(mean, (-1,))
    var = np.diag(var)
    ys = np.reshape(ys, (-1,))

    mean_predictions = 0.5 * np.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2)

    return mean_predictions


st.header("Objective")

"""
Recent work has shown:

(a) An infinitely wide deep network when randomly initialised is a Gaussian Process
, referred to as **NNGP**. This allows us to use classical tools 
from Gaussian Process and Bayesian Inference literature.

(b) An infinitely wide network evolves in a linearised fashion. 
In fact, its weights barely move at all, and it can be described by a kernel function,
referred to as the **Neural Tanget Kernel (NTK)**.

Note the difference between the two; the first is purely Bayesian, while the second is
a kernel machine. We do have some interesting connections between kernel machines and Gaussian Processes.
We shall delve into that in a bit.

Unfortunately (_or unsurprisingly?_), the kernels for the formulations **do not** match.
"""

st.header("Toy Dataset: Synthetic Functions")

st.markdown(r"We shall consider a toy dataset comprised of:")

st.latex(r"y = f(x) + \epsilon")

st.markdown("Where $\epsilon \sim \mathcal{N}(0,\sigma)$.")

st.subheader("Choose Function, Range and Periodicity")

st.sidebar.markdown("## Toy Dataset")

train_points = st.sidebar.slider("Train points", 0, 20, 5)
test_points = st.sidebar.slider("Test points", 0, 100, 50)
noise_scale = st.sidebar.slider("Noise scale", 0.0, 1.0, 1e-1, step=1e-1)

st.sidebar.markdown("## Function Properties")

target_fn_option = st.sidebar.selectbox(
    "Function", ("sine", "cosine", "triangle", "square")
)
range_values = st.sidebar.slider("Range", -2 * np.pi, 2 * np.pi, (-np.pi, np.pi))

period = st.sidebar.slider("Period", 0.0, 3.0, 1.0)


# st.write(
#     f"Function chosen: {target_fn_option}, range: {range_values}, period: {period}."
# )


target_fn_dict = {
    "sine": np.sin,
    "cosine": np.cos,
    "triangle": partial(signal.sawtooth, width=0.5),
    "square": signal.square,
}
target_fn = target_fn_dict[target_fn_option]

key, x_key, y_key = random.split(key, 3)
train_xs = random.uniform(
    x_key, (train_points, 1), minval=range_values[0], maxval=range_values[1]
)
train_ys = target_fn(train_xs) + noise_scale * random.normal(y_key, (train_points, 1))
train = (train_xs, train_ys)

test_xs = np.linspace(range_values[0], range_values[1], test_points)
test_xs = np.reshape(test_xs, (test_points, 1))

test_ys = target_fn(test_xs / period)
test = (test_xs, test_ys)

ylims = (np.min(train_ys) - 1, np.max(train_ys) + 1)
plot_fn(train, test, xlims=range_values, ylims=ylims)
legend(loc="upper left")
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

st.header("Define your (Finite) Network")

# st.sidebar.markdown("## Network")
n_hidden = st.slider("Hidden width", 16, 2048, 512, step=16)
depth = st.slider("Network depth (excludes input and output)", 1, 10, 2, step=1)
sigma_w = st.slider("Sigma w", 0.1, 3.0, 1.5, step=0.1)
sigma_b = st.slider("Sigma b", 0.01, 0.1, 0.05, step=0.01)

activation_fn = st.selectbox("Activation Function", ("Erf", "ReLU", "None"))

activation_fn_dict = {"Erf": stax.Erf(), "ReLU": stax.Relu(), "None": None}
activation_fn = activation_fn_dict[activation_fn]

sequence = (
    (stax.Dense(n_hidden, W_std=sigma_w, b_std=sigma_b), activation_fn)
    if activation_fn
    else (stax.Dense(n_hidden, W_std=sigma_w, b_std=sigma_b),)
)
init_fn, apply_fn, kernel_fn = stax.serial(
    *(sequence * depth), stax.Dense(1, W_std=sigma_w, b_std=sigma_b)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnums=(2,))

st.markdown(
    """
We define our network using the **Neural Tanget Stax** module.
It allows us to define the architecture, initialisation and 
returns the network function plus (infinite width) kernel function.

To make this clear, `stax.serial` returns `init_fn`, `apply_fn` and `kernel_fn`. 
We can initialise the function by `init_fn` and use `apply_fn` similar to Pytorch's forward. 
`kernel_fn` corresponds to the infinite width kernel.
"""
)

prior_draws = []
for _ in range(10):
    key, net_key = random.split(key)
    _, params = init_fn(net_key, (-1, 1))
    prior_draws += [apply_fn(params, test_xs)]

plot_fn(train, test, xlims=range_values, ylims=ylims)
for p in prior_draws:
    plt.plot(test_xs, p, linewidth=3, color=[1, 0.65, 0.65])

plt.legend(["train", "$f(x)$", "random draw"], loc="upper left", fontsize=10)
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

"""
Here, we drew multiple functions from the specified (finite) network. 
Let's compare this to the infinite case by utilising the variance specified by the NNGP kernel.
"""

kernel = kernel_fn(test_xs, test_xs, "nngp")
std_dev = np.sqrt(np.diag(kernel))
plot_fn(train, test, xlims=range_values, ylims=ylims)
plt.fill_between(np.reshape(test_xs, (-1,)), 2 * std_dev, -2 * std_dev, alpha=0.4)
for p in prior_draws:
    plt.plot(test_xs, p, linewidth=3, color=[1, 0.65, 0.65])
finalize_plot((0.85, 0.6))
plt.legend(["train", "$f(x)$", "random draw"], loc="upper left", fontsize=10)
st.pyplot()
plt.close()

st.header("Bayesian Inference")

st.markdown(
    """
Let us first use the NNGP kernel to perform Bayesian Inference.
Note that this corresponds to no training (or training with just the last layer).

We shall make use of the function `neural_tangents.predict.gp_inference` 
which performs this Bayesian inference exactly.
"""
)

diag_reg = st.slider("Noise level to model", 0.0, 1e-3, 1e-4, step=1e-4, format="%.4f")

nngp_mean, nngp_covariance = nt.predict.gp_inference(
    kernel_fn,
    train_xs,
    train_ys,
    test_xs,
    diag_reg=diag_reg,
    get="nngp",
    compute_cov=True,
)

nngp_mean = np.reshape(nngp_mean, (-1,))
nngp_std = np.sqrt(np.diag(nngp_covariance))

plot_fn(train, test, xlims=range_values, ylims=ylims)
plt.plot(test_xs, nngp_mean, "r-", linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    nngp_mean - 2 * nngp_std,
    nngp_mean + 2 * nngp_std,
    color="red",
    alpha=0.2,
)
plt.xlim(range_values)
plt.ylim(ylims)
legend(["Train", "f(x)", "NNGP Bayesian Inference"], loc="upper left")
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

"""
Notice that some of the train points **do not** lie on the function.
This is where the noise level for inference plays a role,
it lets us decide how much to "believe" the train points.
"""

st.header("NTK Inference: Infinite time inference")

"""
Let's now use the NTK instead. 
This corresponds to training an infinite width network with 
the architecture specified to convergence ("infinite training").
"""

ntk_mean, ntk_covariance = nt.predict.gp_inference(
    kernel_fn, train_xs, train_ys, test_xs, diag_reg=1e-4, get="ntk", compute_cov=True
)

ntk_mean = np.reshape(ntk_mean, (-1,))
ntk_std = np.sqrt(np.diag(ntk_covariance))

plot_fn(train, test, xlims=range_values, ylims=ylims)

plt.plot(test_xs, nngp_mean, "r-", linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    nngp_mean - 2 * nngp_std,
    nngp_mean + 2 * nngp_std,
    color="red",
    alpha=0.2,
)
plt.plot(test_xs, ntk_mean, "b-", linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    ntk_mean - 2 * ntk_std,
    ntk_mean + 2 * ntk_std,
    color="blue",
    alpha=0.2,
)
plt.xlim(range_values)
plt.ylim(ylims)
legend(
    ["Train", "f(x)", "NNGP Bayesian Inference", "NTK Bayesian Inference"],
    loc="upper left",
)
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

"""
We notice that the results are quite similar, but **not equal**. This is because the NNGP kernel does not coincide with the NTK.
"""


"""
In this section, we exploited the following close relation between a kernel regression machine
 and the Gaussian Process defined by **its kernel**:

* The function obtained by the Kernel Ridge Regression (KRR) is equal to the MAP of the Kernel Gaussian Process inference.
* Notice that MAP is a frequentist concept, so we are equating deterministic functions to deterministic functions and all is good.
* This function corressponds to a fully trained KRR.

"""

st.subheader("Here's a short proof of the first point")

st.markdown(
    r"""GP regression gives you a posterior with mean function:
$$
m(x) = k_{xX}(k_{XX} + \sigma^2 I_n)^{-1}Y
$$ 

for the zero prior mean case, like we have here. Here, $x$ is a test point, and we have $n$ train points in $X$. $k_{XX}$ is the kernel evaluated on train points. 
$Y$ is a vector of stacked train targets, ie..., $Y = [y_1,y_2,...,y_n]^T$.

Now, KRR gives us: 
"""
)

st.markdown(
    r"""$$
\hat{f} = \text{argmin}_{f \in \mathcal{H_k}} \frac{1}{n} \sum_{i=1}^n (f(x_i) - y_i)^2 + \lambda||f||^2_{\mathcal{H}_k}
$$

where, $[\alpha_1,...,\alpha_n]^T := (k_{XX} + \sigma^2I_n)^{-1}Y \in \mathbb{R}^n$. 
$\mathcal{H}_k$ is the **Reproducing Kernel Hilbert Space** (RKHS) defined by the kernel $k(.,.)$.
"""
)

"""
This gives (by plugging in definitions of a kernel function): 
"""

st.markdown(
    r"""$$
\hat{f}(x) = k_{xX}(k_{XX} + \sigma^2I_n)^{-1}Y = \sum_{i=1}^n \alpha_ik(x,x_i), \quad x \in X
$$"""
)

"""
**For the astute reader:** we utilised definitions of RKHS, which guarantees
 that each function may be "reproduced" over its inner product, and hence represented as a sum of weighted cannonical feature maps ($k(.,x_i)$).
 The rest follows from simple calculus.
 
"""


""" In practice, we cannot use this trick at will, since running a KRR evaluation is
$O(n^2)$ while using GP Bayesian Inference is atleast $O(n^3)$ thanks to the matrix inversion
involved."""

st.header("NTK Inference: Finite time inference")


"""
In this section, we use the NTK to perform ridge regression. 
We will use this to predict the mean of the train and test losses over the course of training. 
To compute the mean MSE loss, we need to access the mean and variance of our networks predictions as a function of time. 
To do this, we use the function:

```predict_fn = nt.predict.gradient_descent_mse_gp(kernel_fn, train_xs, train_ys, xs, 'ntk')```

This is provided by the Neural Tangents library and returns a `predict_fn(t)` 
that computes the mean and variance of function values evaluated on $x$'s at a time $t$. 
"""

train_predict_fn = nt.predict.gradient_descent_mse_gp(
    kernel_fn, train_xs, train_ys, train_xs, "ntk", 1e-4, compute_cov=True
)
test_predict_fn = nt.predict.gradient_descent_mse_gp(
    kernel_fn, train_xs, train_ys, test_xs, "ntk", 1e-4, compute_cov=True
)

train_loss_fn = functools.partial(loss_fn, train_predict_fn, train_ys)
test_loss_fn = functools.partial(loss_fn, test_predict_fn, test_ys)


training_steps = st.slider("Training Steps", 5, 10000, 100, step=100)

ts = np.arange(0, training_steps)
ntk_train_loss_mean = vmap(train_loss_fn)(ts)
ntk_test_loss_mean = vmap(test_loss_fn)(ts)


plt.plot(ts, ntk_train_loss_mean, linewidth=3)
plt.plot(ts, ntk_test_loss_mean, linewidth=3)
plt.xlim((0, training_steps))
format_plot("Step", "Loss")
legend(["Train", "Test"])
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

"""
Notice that it more or less converges after 200 steps. 
For completeness, here's a log-log plot of the same.
"""

plt.loglog(ts, ntk_train_loss_mean, linewidth=3)
plt.loglog(ts, ntk_test_loss_mean, linewidth=3)
format_plot("Step", "Loss")
legend(["Train", "Test"])
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

st.header("Comparison against finite SGD-NNs")

"""
Finally, let's bring back the practioner loved Neural Networks for a comparison. 
"""

learning_rate = st.slider("Learning rate", 1e-4, 1.0, 0.1, step=1e-4, format="%.4f")

opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
opt_update = jit(opt_update)
loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

train_losses = []
test_losses = []

opt_state = opt_init(params)

for i in range(training_steps):
    opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

    train_losses += [loss(get_params(opt_state), *train)]
    test_losses += [loss(get_params(opt_state), *test)]

# NTK loss
plt.loglog(ts, ntk_train_loss_mean, linewidth=3)
plt.loglog(ts, ntk_test_loss_mean, linewidth=3)
# SGD NN loss
plt.loglog(ts, train_losses, "k-", linewidth=2)
plt.loglog(ts, test_losses, "k-", linewidth=2)
format_plot("Step", "Loss")
legend(["NTK Train", "NTK Test", "SGD-NN Train", "SGD-NN Test"])
finalize_plot((0.85, 0.6))
plt.title("Loss curve comparisons")
st.pyplot()
plt.close()

plot_fn(train, test, xlims=range_values, ylims=ylims)
# NNGP
plt.plot(test_xs, nngp_mean, "r-", linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    nngp_mean - 2 * nngp_std,
    nngp_mean + 2 * nngp_std,
    color="red",
    alpha=0.2,
)
# NTK
plt.plot(test_xs, ntk_mean, "b-", linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    ntk_mean - 2 * ntk_std,
    ntk_mean + 2 * ntk_std,
    color="blue",
    alpha=0.2,
)
# SGD NN draw
plt.plot(test_xs, apply_fn(get_params(opt_state), test_xs), "k-", linewidth=2)
legend(["Train", "f(x)", "NNGP", "NTK", "SGD-NN"], loc="upper left")
plt.title("Function draw comparisons")
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

st.subheader("Comparison against a finite ensemble of SGD-NNs")

"""
The previous plot is quite compelling, it shows that the function obtained from SGD-NN agrees well with NTK inference. 
*If it doesn't you should probably try a higher number of training steps*. 
Let's go ahead and get a variance comparison by training an ensemble of finite SGD-NNs, and plot the variance in their predictions. 

We leave out **NNGP** from this comparison for clarity.
"""


def train_network(key):
    train_losses = []
    test_losses = []

    _, params = init_fn(key, (-1, 1))
    opt_state = opt_init(params)

    for i in range(training_steps):
        train_losses += [np.reshape(loss(get_params(opt_state), *train), (1,))]
        test_losses += [np.reshape(loss(get_params(opt_state), *test), (1,))]
        opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

    train_losses = np.concatenate(train_losses)
    test_losses = np.concatenate(test_losses)
    return get_params(opt_state), train_losses, test_losses


ensemble_size = st.slider("Ensemble size", 5, 500, 5, step=10)
ensemble_key = random.split(key, ensemble_size)
params, train_loss, test_loss = vmap(train_network)(ensemble_key)

# SGD NN
mean_train_loss = np.mean(train_loss, axis=0)
mean_test_loss = np.mean(test_loss, axis=0)
# NTK

plt.plot(ts, ntk_train_loss_mean, linewidth=3)
plt.plot(ts, ntk_test_loss_mean, linewidth=3)
plt.plot(ts, mean_train_loss, "k-", linewidth=2)
plt.plot(ts, mean_test_loss, "k-", linewidth=2)
plt.xlim([10 ** 0, training_steps])
plt.title("Loss Curve Comparisons")
format_plot("Step", "Loss")
legend(["NTK Train", "NTK Test", "SDG-NN Ensemble Train", "SDG-NN Ensemble Test"])
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

plt.loglog(ts, ntk_train_loss_mean, linewidth=3)
plt.loglog(ts, ntk_test_loss_mean, linewidth=3)
plt.loglog(ts, mean_train_loss, "k-", linewidth=2)
plt.loglog(ts, mean_test_loss, "k-", linewidth=2)
plt.xlim([10 ** 0, training_steps])
plt.xscale("log")
plt.yscale("log")
format_plot("Step", "Loss")
plt.title("Loss Curve Comparisons (log-log)")
legend(["NTK Train", "NTK Test", "SDG-NN Ensemble Train", "SDG-NN Ensemble Test"])
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

plot_fn(train, test, xlims=range_values, ylims=ylims)
# NTK
plt.plot(test_xs, ntk_mean, "b-", linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    ntk_mean - 2 * ntk_std,
    ntk_mean + 2 * ntk_std,
    color="blue",
    alpha=0.2,
)

ensemble_fx = vmap(apply_fn, (0, None))(params, test_xs)
mean_fx = np.reshape(np.mean(ensemble_fx, axis=0), (-1,))
std_fx = np.reshape(np.std(ensemble_fx, axis=0), (-1,))
# SGD NN Ensemble
plt.plot(test_xs, mean_fx - 2 * std_fx, "k--", label="_nolegend_")
plt.plot(test_xs, mean_fx + 2 * std_fx, "k--", label="_nolegend_")
plt.plot(test_xs, mean_fx, linewidth=2, color="black")
plt.title("Function Draw Comparison")
legend(["Train", "f(x)", "NTK", "SGD-NN Ensemble"], loc="upper left")
plt.xlim(range_values)
plt.ylim(ylims)
format_plot("$x$", "$f$")
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

"""
We see pretty nice agreement between exact inference of the infinite-width networks and the result of training an ensemble! 
Note that we do see some deviations in the training loss at the end of training. 
**This is ameliorated by using a wider network.**
"""

st.header("Extending this to Residual Architectures")

"""
Thanks to the expressivity of the *neural tangents* library, 
we can repeat the previous plots (NNGP, NTK and finite ensemble) for the case of residual architectures.
"""

n_hidden = st.slider("Hidden width for Residual Case", 16, 2048, 64, step=16)
depth = st.slider("Network depth (excludes input and output) for each Residual Block", 1, 10, 1, step=1)
sigma_w = st.slider("Sigma w for Residual Case ", 0.1, 3.0, 1.5, step=0.1)
sigma_b = st.slider("Sigma b for Residual Case", 0.0, 0.1, 0.05, step=0.01)

activation_fn = st.selectbox("Activation Function for Residual Case", ("Erf", "ReLU", "None"))

activation_fn = activation_fn_dict[activation_fn]

sequence = (
    (activation_fn, stax.Dense(n_hidden, W_std=sigma_w, b_std=sigma_b))
    if activation_fn
    else (stax.Dense(n_hidden, W_std=sigma_w, b_std=sigma_b),)
)

ResBlock = stax.serial(
    stax.FanOut(2),
    stax.parallel(stax.serial(*(sequence * depth)), stax.Identity()),
    stax.FanInSum(),
)

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(n_hidden, W_std=sigma_w, b_std=sigma_b),
    ResBlock,
    ResBlock,
    activation_fn,
    stax.Dense(1, W_std=sigma_w, b_std=sigma_b),
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnums=(2,))

opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
opt_update = jit(opt_update)

key, = random.split(key, 1)
ensemble_key = random.split(key, ensemble_size)
params, train_loss, test_loss = vmap(train_network)(ensemble_key)

diag_reg = st.slider("Noise level to model for Residual Case", 0.0, 1e-3, 1e-4, step=1e-4, format="%.4f")

kernel = kernel_fn(test_xs, test_xs, "nngp")

nngp_mean, nngp_covariance = nt.predict.gp_inference(
    kernel_fn,
    train_xs,
    train_ys,
    test_xs,
    diag_reg=diag_reg,
    get="nngp",
    compute_cov=True,
)

nngp_mean = np.reshape(nngp_mean, (-1,))
nngp_std = np.sqrt(np.diag(nngp_covariance))

ntk_mean, ntk_var = nt.predict.gp_inference(
    kernel_fn, train_xs, train_ys, test_xs, diag_reg=1e-4, get="ntk", compute_cov=True
)
ntk_mean = np.reshape(ntk_mean, (-1,))
ntk_std = np.sqrt(np.diag(ntk_var))

train_predict_fn = nt.predict.gradient_descent_mse_gp(
    kernel_fn, train_xs, train_ys, train_xs, "ntk", 1e-4, compute_cov=True
)
test_predict_fn = nt.predict.gradient_descent_mse_gp(
    kernel_fn, train_xs, train_ys, test_xs, "ntk", 1e-4, compute_cov=True
)

train_loss_fn = functools.partial(loss_fn, train_predict_fn, train_ys)
test_loss_fn = functools.partial(loss_fn, test_predict_fn, test_ys)

ntk_train_loss_mean = vmap(train_loss_fn)(ts)
ntk_test_loss_mean = vmap(test_loss_fn)(ts)

# SGD NN
mean_train_loss = np.mean(train_loss, axis=0)
mean_test_loss = np.mean(test_loss, axis=0)
# NTK

plt.plot(ts, ntk_train_loss_mean, linewidth=3)
plt.plot(ts, ntk_test_loss_mean, linewidth=3)
plt.plot(ts, mean_train_loss, "k-", linewidth=2)
plt.plot(ts, mean_test_loss, "k-", linewidth=2)
plt.xlim([10 ** 0, training_steps])
plt.title("Loss Curve Comparisons")
format_plot("Step", "Loss")
legend(["NTK Train", "NTK Test", "SDG-NN Ensemble Train", "SDG-NN Ensemble Test"])
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

plt.loglog(ts, ntk_train_loss_mean, linewidth=3)
plt.loglog(ts, ntk_test_loss_mean, linewidth=3)
plt.loglog(ts, mean_train_loss, "k-", linewidth=2)
plt.loglog(ts, mean_test_loss, "k-", linewidth=2)
plt.xlim([10 ** 0, training_steps])
plt.xscale("log")
plt.yscale("log")
format_plot("Step", "Loss")
plt.title("Loss Curve Comparisons (log-log)")
legend(["NTK Train", "NTK Test", "SDG-NN Ensemble Train", "SDG-NN Ensemble Test"])
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

plot_fn(train, test, xlims=range_values, ylims=ylims)
# NNGP
plt.plot(test_xs, nngp_mean, "r-", linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    nngp_mean - 2 * nngp_std,
    nngp_mean + 2 * nngp_std,
    color="red",
    alpha=0.2,
)
# NTK
plt.plot(test_xs, ntk_mean, "b-", linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    ntk_mean - 2 * ntk_std,
    ntk_mean + 2 * ntk_std,
    color="blue",
    alpha=0.2,
)

ensemble_fx = vmap(apply_fn, (0, None))(params, test_xs)
mean_fx = np.reshape(np.mean(ensemble_fx, axis=0), (-1,))
std_fx = np.reshape(np.std(ensemble_fx, axis=0), (-1,))
# SGD NN Ensemble
plt.plot(test_xs, mean_fx - 2 * std_fx, "k--", label="_nolegend_")
plt.plot(test_xs, mean_fx + 2 * std_fx, "k--", label="_nolegend_")
plt.plot(test_xs, mean_fx, linewidth=2, color="black")
plt.title("Function Draw Comparison")
legend(
    ["Train", "f(x)", "Residual NNGP", "Residual NTK", "Residual SGD-NN Ensemble"],
    loc="upper left",
)
plt.xlim(range_values)
plt.ylim(ylims)
format_plot("$x$", "$f$")
finalize_plot((0.85, 0.6))
st.pyplot()
plt.close()

st.header("Acknowledgements")

"""
A bulk of the inspiration and code behind this post is attributed to the *Neural Tangents Cookbook* [1]. 
I've personally found the *Cookbook* to be quite informative, although its dense nature can make it hard for a first read. 
A lot of effort has been put into ensuring that the results of the NNGP-NTK idea can be toyed around first, 
with the hope that readers develop stronger intuition for the two concepts.

If you liked this post, do go ahead and take a look at the [neural tangets repository](https://github.com/google/neural-tangents) on Github. 
In case you would like to read (*the several*) papers behind this, I've cited all of them in the references below.

This was my first time using streamlit, and honestly, I've found it very . Kudos to the developers!

Finally, none of this would have been possible if not for the course *CS:7020, Fall 2019*. 
To this end, I would like to credit Professor [Harish Guruprasad](https://sites.google.com/site/harishguruprasad/)
 for providing this opportunity. 
I would also like to thank [Rajat VD](https://github.com/rajatvd) for insightful discussions on this topic.
"""

st.header("References")

"""
1. [Neural Tangents Cookbook](https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/neural_tangents_cookbook.ipynb#scrollTo=IvqFKuxY5pWx).
2. **NNGP Literature:**
    a. Lee and Bahri et al.,"",ICLR 2018.
    b. Mathews *et al.*, "", ICLR 2018.
3. **NTK Literature:**
    a. 
4. **GP-KRR Relation:**
    a. 
5. [Source code] for this post.
"""
