from simplex import Simplex
import numpy as np

objective = ('maximize', '2x_1 + 1x_2')
constraints = ['3x_1 + 1x_2  <= 30',
               '4x_1 + 3x_2 <= 16',
               '1x_1 + 2x_2 >= 4',
               '1x_1 + 3x_2 >= 5']
model = Simplex(num_vars=2, constraints=constraints, objective_function=objective)




# Checking specific lines of tex of tableaux
print(model.tableaux_list[0])

# Composing tex text ready to use for a specific subset of iterations
print(model.compose_tableaux())

reduced_cost = model.all_bases[0].reduced_cost
u = model.all_bases[0].u
c = model.c

# to move to simple.py
reduced_cost = np.array(reduced_cost).reshape(-1, 1)
u = np.array(u).reshape(-1, 1)

