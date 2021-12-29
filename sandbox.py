from simplex import Simplex
objective = ('maximize', '2x_1 + 1x_2')
constraints = ['3x_1 + 1x_2  <= 30',
               '4x_1 + 3x_2 <= 16',
               '1x_1 + 2x_2 >= 4',
               '1x_1 + 3x_2 >= 5']
Lp_system = Simplex(num_vars=2, constraints=constraints, objective_function=objective)
# print(Lp_system.tableaux_tex)
# print(Lp_system.var_names)
len(Lp_system.tableaux_list)

Lp_system.tableaux_list[len(Lp_system.tableaux_list)-1]
Lp_system.tableaux_list[0]

print(Lp_system.compose_tableaux(len(Lp_system.tableaux_list)-1))

print(Lp_system.compose_tableaux(3))