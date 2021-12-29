from simplex import Simplex
objective = ('maximize', '1x_1 + 2x_2 + 3x_3')
constraints = ['1x_1 + 1x_2 + 1x_3 <= 16', '3x_1 + 2x_2 + 2x_3 = 26', '1x_1 + 1x_3 >= 10']
Lp_system = Simplex(num_vars=3, constraints=constraints, objective_function=objective)
print(Lp_system.tableaux_tex)
# print(Lp_system.var_names)
