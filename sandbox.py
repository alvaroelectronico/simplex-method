from simplex import Simplex
objective = ('minimize', '4x_1 + 1x_2')
constraints = ['3x_1 + 1x_2 = 3', '4x_1 + 3x_2 >= 6', '1x_1 + 2x_2 <= 4']
Lp_system = Simplex(num_vars=2, constraints=constraints, objective_function=objective)

str = "\\begin{center}\n"
str += "\\begin{tabular}{c|"
str += "ccc"
str += "|}\n"
for i, row in enumerate(Lp_system.coeff_matrix):
    if i == 1:
        str += "\hline\n"
    str += "{}".format(row[len(row)-1])
    for i in range(1, len(row)-1):
        str += " & {}".format(row[i])
    str += "\\\\ \n"
str += "\hline\n"
str += "\end{tabular}\n\end{center}\n"
print(str)

