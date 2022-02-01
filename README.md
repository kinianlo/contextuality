# Contextualiy
A library for handling sheaf-theoretic empirical models. The library currently support the following computation:
1. Contextual fraction
2. Signalling fraction 
3. CbD measure for _binary cyclic measurement scenarios_

## Install
1. Clone the repository:  
```
git clone https://github.com/kinianlo/contextuality.git
```

2. Change directory into `contextuality`:  
```
cd contextuality
```

3. Install the package using `pip`:  
```
pip install . --upgrade
```  
Note: The `--upgrade` flag here is to ensure that the newest version gets installed even if an older version is previously installed.


## Examples
```
from contextuality.model import pr_model
pr_box = pr_model()
print(pr)
cf = pr_box.contextual_fraction()
print(f"The contetual fraction of the PR box is {cf:.4f}."}
```

One can also construct an empirical model from a table of distributions.

```
from contextuality.model import Model, chsh_scenario
scneario = chsh_scenario()
table = [[4/8, 0/8, 0/8, 4/8],
         [3/8, 1/8, 1/8, 3/8],
         [3/8, 1/8, 1/8, 3/8],
         [1/8, 3/8, 3/8, 1/8]]
bell_model = Model(scneario, table)
cf = bell_model.contextual_fraction()
print(f"The contetual fraction of the Bell model is {cf:.4f}."}
```
