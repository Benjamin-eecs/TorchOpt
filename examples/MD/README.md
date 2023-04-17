# Reinforcement learning with Model-Agnostic Meta-Learning (MAML)

Many physical simulations require solving optimization problems, such as energy minimization in molecular. In this experiment, we revisit an example from JAX-MD [76], the problem of finding energy minimizing configurations to a system of $k$ packed particles in a 2-dimensional box of size $\ell$
$$
x^{\star}(\theta)=\underset{x \in \mathbb{R}^{k \times 2}}{\operatorname{argmin}} f(x, \theta):=\sum_{i, j} U\left(x_{i, j}, \theta\right),
$$
where $x^{\star}(\theta) \in \mathbb{R}^{k \times 2}$ are the optimal coordinates of the $k$ particles, $U\left(x_{i, j}, \theta\right)$ is the pairwise potential energy function, with half the particles at diameter 1 and half at diameter $\theta=0.6$, which we optimize with a domain-specific optimizer [15]. Here we consider sensitivity of particle position with respect to diameter $\partial x^{\star}(\theta)$, rather than sensitivity of the total energy from the original experiment. Figure 6 shows results calculated via forward-mode implicit

## Usage

Specify the seed to train.

```bash
### Run MAML
python maml.py --seed 1
### Run torchrl MAML implementation
python maml_torchrl.py --seed 1
```

## Results

The training curve and testing curve between initial policy and adapted policy validate the effectiveness of algorithms.
