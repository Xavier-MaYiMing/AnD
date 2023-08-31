### AnD: A many-objective evolutionary algorithm with angle-based selection and shift-based density estimation

##### Reference: Liu Z Z, Wang Y, Huang P Q. AnD: A many-objective evolutionary algorithm with angle-based selection and shift-based density estimation[J]. Information Sciences, 2020, 509: 400-419.

##### AnD is a many-objective evolutionary algorithm (MaOEA).

| Variables | Meaning                                              |
| --------- | ---------------------------------------------------- |
| npop      | Population size                                      |
| iter      | Iteration number                                     |
| lb        | Lower bound                                          |
| ub        | Upper bound                                          |
| nobj      | The dimension of objective space (default = 3)       |
| pc        | Crossover probability (default = 1)                  |
| pm        | Mutation probability (default = 1)                   |
| eta_c     | Spread factor distribution index (default = 20)      |
| eta_m     | Perturbance factor distribution index (default = 20) |
| nvar      | The dimension of decision space                      |
| pop       | Population                                           |
| objs      | Objectives                                           |
| zmin      | Ideal point                                          |
| zmax      | Nadir point                                          |
| angle     | The angle between each pair of objectives            |
| remain    | The remaining individuals to the next generation     |
| pf        | Pareto front                                         |

#### Test problem: DTLZ1

$$
\begin{aligned}
	& k = nvar - nobj + 1, \text{ the last $k$ variables is represented as $x_M$} \\
	& g(x_M) = 100 \left[|x_M| + \sum_{x_i \in x_M}(x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5)) \right] \\
	& \min \\
	& f_1(x) = \frac{1}{2}x_1x_2 \cdots x_{M - 1}(1 + g(x_M)) \\
	& f_2(x) = \frac{1}{2}x_1x_2 \cdots (1 - x_{M - 1})(1 + g(x_M)) \\
	& \vdots \\
	& f_{M - 1}(x) = \frac{1}{2}x_1(1 - x_2)(1 + g(x_M)) \\
	& f_M(x) = \frac{1}{2}(1 - x_1)(1 + g(x_M)) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 400, np.array([0] * 7), np.array([1] * 7))
```

##### Output:

![Pareto front](/Users/xavier/Desktop/Xavier Ma/个人算法主页/AnD/Pareto front.png)



