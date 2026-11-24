"""Train a small CNN on random data with the JAX version of Tensorion.

    pip install "tensorion[jax]"
    python examples/quickstart_jax.py
"""

import jax
import jax.numpy as jnp
import optax

import tensorion_jax

key = jax.random.PRNGKey(0)
k1, k2, k3, kx, ky = jax.random.split(key, 5)
params = {
    "conv1": {"kernel": 0.1 * jax.random.normal(k1, (32, 3, 3, 3)), "bias": jnp.zeros(32)},
    "conv2": {"kernel": 0.1 * jax.random.normal(k2, (32, 32, 3, 3)), "bias": jnp.zeros(32)},
    "head": {"kernel": 0.1 * jax.random.normal(k3, (10, 32)), "bias": jnp.zeros(10)},
}


def conv(x, layer):
    y = jax.lax.conv_general_dilated(x, layer["kernel"], (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return jax.nn.relu(y + layer["bias"][None, :, None, None])


def loss_fn(params, x, y):
    h = conv(conv(x, params["conv1"]), params["conv2"]).mean(axis=(2, 3))
    logits = h @ params["head"]["kernel"].T + params["head"]["bias"]
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


# Kernels and the head get the Tensorion update, biases get AdamW.
# conv2 picks its unfolding online, every other layer uses the offline rule.
optimizer = tensorion_jax.tensorion(1e-2, unfolding={"conv2/kernel": "online"})
state = optimizer.init(params)


@jax.jit
def step(params, state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, state = optimizer.update(grads, state, params)
    return optax.apply_updates(params, updates), state, loss


x = jax.random.normal(kx, (256, 3, 16, 16))
y = jax.random.randint(ky, (256,), 0, 10)
for i in range(50):
    params, state, loss = step(params, state, x, y)
    if i % 10 == 0 or i == 49:
        print(f"step {i:2d}  loss {float(loss):.3f}")
