from simplex import Simplex
import numpy as np

objective = ('maximize', '1x_1 + 2x_2 + 3x_3')
constraints = ['1x_1 + 1x_2 + 1x_3 <= 16',
               '3x_1 + 2x_2 + 2x_3 = 26',
               '1x_1 + 0x_2 + 1x_3 >= 10']
               # '1x_1 + 3x_2 = 5']
model = Simplex(num_vars=3, constraints=constraints, objective_function=objective)

model.solve_model()
print(model.compose_tableaux())
model.sense


print(model.var_names)
print(model.A)
print(model.b)
print(model.r_rows)
print(model.var_names)
print(model.tableaux_list)
print(model.coeff_matrix)
print(model.coeff_matrix[0])




model.solution

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



