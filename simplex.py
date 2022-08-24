from fractions import Fraction
from warnings import warn
from collections import namedtuple
import numpy as np
from mpmath.functions.rszeta import coef

class Simplex(object):
    def __init__(self, num_vars, constraints, objective_function):
        """
        num_vars: Number of variables

        equations: A list of strings representing constraints
        each variable should be start with x followed by a underscore
        and a number
        eg of constraints
        ['1x_1 + 2x_2 >= 4', '2x_3 + 3x_1 <= 5', 'x_3 + 3x_2 = 6']
        Note that in x_num, num should not be more than num_vars.
        Also single spaces should be used in expressions.

        objective_function: should be a tuple with first element
        either 'min' or 'max', and second element be the equation
        eg 
        ('min', '2x_1 + 4x_3 + 5x_2')

        For solution finding algorithm uses two-phase simplex method
        """
        self.num_vars = num_vars
        self.constraints = constraints
        self.objective = objective_function[0]
        self.objective_function = objective_function[1]

        # Building all matrices and info for the simplex method.
        # coeff_matrix: list containing the full matrix:
        # * element 0: list with reduced costs and obj. function value
        # * elements 1 to no. constraints: list with B^(-1)A and B^(-1)
        # r_rows is a list containing the indices of coeff_matrix corresponding to rows with artif. vars.
        # num_s_vars (num_r_vars): no. of slack (artifitial) vars.
        # b: constraints RHS
        # A: A matrix (no. constraints x no. vars)

        self.coeff_matrix,  self.s_rows, self.r_rows, self.num_s_vars, self.num_r_vars, self.c1, \
        self.c2, self.A, self.b = self.construct_matrix_from_constraints()

        self.total_vars = self.num_vars + self.num_s_vars + self.num_r_vars

        # Getting the sense of the problem
        if 'min' in self.objective.lower():
            self.sense = 'min'
        else:
            self.sense = 'max'


        # Generating the .tex text with the model formulation
        self.model_tex = self.formulation_tex()

        # var_names ia a list containing the names
        # x_1...x_{num_vars}, h_1...h_{num_s_vars}, a_1...a_{num_r_vars}
        # to be displayed when tableaux are built
        self.var_names = self.name_variables()

        # tableaux_list is a list where each element contains the tex format for building the tableau
        self.tableaux_list = list()

        # all_solutions is a list with as many namedtuple solutions as iterations of the simplex method
        self.all_bases = list()

    def solve_model(self):

        self.phase1()
        # TO-DO: stop if problem is infeasible
        self.phase2()
        self.optimize_val = self.coeff_matrix[0][-1]

    def construct_matrix_from_constraints(self):

        num_s_vars = 0  # number of slack variables
        num_r_vars = 0  # number of artificial variables
        for expression in self.constraints:
            if '<=' in expression:
                num_s_vars += 1
            elif '>=' in expression:
                num_s_vars += 1
                num_r_vars += 1
            elif '=' in expression:
                num_r_vars += 1
        total_vars = self.num_vars + num_s_vars + num_r_vars

        coeff_matrix = [[Fraction("0/1") for i in range(total_vars+1)] for j in range(len(self.constraints)+1)]

        s_index = self.num_vars
        r_index = self.num_vars + num_s_vars
        r_rows = list()  # stores the non-zero index of art. variables
        s_rows = list()  # stores the non-zero index of slack. variables

        c1 = [Fraction("0/1") for i in range(total_vars)]

        for i in range(1, len(self.constraints)+1):
            constraint = self.constraints[i-1].split(' ')
            for j in range(len(constraint)):
                if '_' in constraint[j]:
                    coeff, index = constraint[j].split('_')
                    if constraint[j-1] == '-':
                        coeff_matrix[i][int(index) - 1] = Fraction(-coeff[:-1]).limit_denominator()
                    else:
                        coeff_matrix[i][int(index) - 1] = Fraction(coeff[:-1]).limit_denominator()

                elif constraint[j] == '<=':
                    coeff_matrix[i][s_index] = Fraction("1/1")  # add surplus variable
                    s_index += 1
                    s_rows.append(i)

                elif constraint[j] == '>=':
                    coeff_matrix[i][s_index] = Fraction("-1/1")  # slack variable
                    coeff_matrix[i][r_index] = Fraction("1/1")   # r variable
                    s_index += 1
                    r_index += 1
                    s_rows.append(i)
                    r_rows.append(i)
                    c1[s_index+1] = -1

                elif constraint[j] == '=':
                    coeff_matrix[i][r_index] = Fraction("1/1")  # r variable
                    r_index += 1
                    r_rows.append(i)

            coeff_matrix[i][-1] = Fraction(constraint[-1] + "/1")

        c2 = [Fraction("0/1") for i in range(total_vars)]
        objective_function_coeffs = self.objective_function.split()
        for i in range(len(objective_function_coeffs)):
            if '_' in objective_function_coeffs[i]:
                coeff, index = objective_function_coeffs[i].split('_')
                if objective_function_coeffs[i - 1] == '-':
                    c2[int(index) - 1] = Fraction("-" +coeff[:-1] + "/1")
                else:
                    c2[int(index) - 1] = Fraction(coeff[:-1] + "/1")

        c1 = np.array(c1).reshape(-1,1)
        c1 = array_to_fraction(c1)

        c2 = np.array(c2).reshape(-1, 1)
        c2 = array_to_fraction(c2)


        a = [[coeff_matrix[i][j] for j in range(len(coeff_matrix[0])-1)] for i in range(1, len(coeff_matrix))]
        a = np.array(a)
        a = array_to_fraction(a)

        b = [coeff_matrix[i][len(coeff_matrix[0])-1] for i in range(1, len(coeff_matrix))]
        b = np.array(b).reshape(-1,1)
        b = array_to_fraction(b)


        return coeff_matrix, s_rows, r_rows, num_s_vars, num_r_vars, c1, c2, a, b

    def phase1(self):
        # Objective function here is minimize r1+ r2 + r3 + ... + rn

        # r_index contains the position of the first artificial variable
        r_index = self.num_vars + self.num_s_vars

        # basic_vars initialized to a list with as many elements as constraints.
        # TO-DO. Revise tha fact that the list goes from 0 to no. constraints (it seems there is an extra element)
        self.basic_vars = [0 for i in range(len(self.coeff_matrix))]

        # Changing first row (index = 0) of coeff_matrix according to the phase 1
        for i in range(r_index, len(self.coeff_matrix[0])-1):
            self.coeff_matrix[0][i] = Fraction("-1/1")

        # Computing reduced costs for first base (where slack and art. variables are basic)
        # Basic variable indices corresponding to art. variables are stored in basic_vars
        for i in self.r_rows:
            self.coeff_matrix[0] = sum_rows(self.coeff_matrix[0], self.coeff_matrix[i])
            self.basic_vars[i] = r_index
            r_index += 1

        # Basic variable indices corresponding to slack. variables (+1 coeff.) are stored in basic_vars
        s_index = self.num_vars
        for i in range(1, len(self.basic_vars)):
            if self.basic_vars[i] == 0:
                self.basic_vars[i] = s_index
                s_index += 1

        # Run the simplex iterations
        # Selecting the input variable
        key_column = max_index(self.coeff_matrix[0][:-1])
        # There are candidate input variables if the corresponding reduced cost is > 0
        condition = self.coeff_matrix[0][key_column] > 0
        # The tableau and the current base is stored in tableaux_list and all_bases, respect.
        self.tableaux_list.append(self.tableau_tex_from_coeff_matrix())
        self.all_bases.append([int(i) for i in self.basic_vars[1:]])

        while condition is True:
            # The ouput variable is selected via find_key_row method
            key_row = self.find_key_row(key_column=key_column)
            # The output var. is replaced with the input var. in basic_vars
            self.basic_vars[key_row] = key_column
            # Getting the value of the pivot
            pivot = self.coeff_matrix[key_row][key_column]
            # Pivoting: dividing the pivot-row by the pivot.
            self.normalize_to_pivot(key_row, pivot)
            # Pivoting: making all values (but the pivot) of the pivot-column 0
            self.make_key_column_zero(key_column, key_row)
            # Selecting the input variable
            key_column = max_index(self.coeff_matrix[0][:-1])
            # There are candidate input variables if the corresponding reduced cost is > 0
            condition = self.coeff_matrix[0][key_column] > 0
            # The tableau and the current base is stored in tableaux_list and all_bases, respect.
            self.tableaux_list.append(self.tableau_tex_from_coeff_matrix())
            self.all_bases.append([int(i) for i in self.basic_vars[1:]])

        # If any art. var is basic the problem is infeseasible. No phase 2 required.
        # TO-DO. If all art. vars are 0, the problem is feaseible
        for i in self.basic_vars:
            if i > r_index:
                raise ValueError("Infeasible problem")

    def phase2(self):

        # Replacing element 0 in coeff_matrix (OF coeff. for phase 2).
        self.update_objective_function()

        # Getting reduced
        for row, column in enumerate(self.basic_vars[1:]):
            if self.coeff_matrix[0][column] != 0:
                self.coeff_matrix[0] = sum_rows(self.coeff_matrix[0],
                                                multiply_const_row(-self.coeff_matrix[0][column],
                                                                   self.coeff_matrix[row + 1]))

        # Selecting the input variable and checking if there is room for improvement
        # art. vars. are not eligible whenr chosing the input variable
        if self.sense == 'max':
            print("")
            print(self.coeff_matrix[0][:self.num_vars + self.num_s_vars])
            key_column = max_index(self.coeff_matrix[0][:self.num_vars + self.num_s_vars])
            print(key_column)
            print("")
            condition = self.coeff_matrix[0][key_column] > 0
        else:
            key_column = min_index(self.coeff_matrix[0][:self.num_vars + self.num_s_vars])
            condition = self.coeff_matrix[0][key_column] < 0

        self.tableaux_list.append(self.tableau_tex_from_coeff_matrix())
        self.all_bases.append([int(i) for i in self.basic_vars[1:]])

        while condition is True:

            key_row = self.find_key_row(key_column=key_column)
            self.basic_vars[key_row] = key_column
            pivot = self.coeff_matrix[key_row][key_column]
            self.normalize_to_pivot(key_row, pivot)
            self.make_key_column_zero(key_column, key_row)

            # r vars are not considered for chosing the pivot column
            if self.sense == 'max':
                key_column = max_index(self.coeff_matrix[0][:self.num_vars + self.num_s_vars])
                condition = self.coeff_matrix[0][key_column] > 0
            else:
                key_column = min_index(self.coeff_matrix[0][:self.num_vars + self.num_s_vars])
                condition = self.coeff_matrix[0][key_column] < 0
            self.tableaux_list.append(self.tableau_tex_from_coeff_matrix())
            self.all_bases.append([int(i) for i in self.basic_vars[1:]])

        solution = {}
        for i, var in enumerate(self.basic_vars[1:]):
            if var < self.num_vars:
                solution['x_' + str(var + 1)] = self.coeff_matrix[i + 1][-1]

        for i in range(0, self.num_vars):
            if i not in self.basic_vars[1:]:
                solution['x_' + str(i + 1)] = Fraction("0/1")

        self.check_alternate_solution()

        self.solution = solution

    def find_key_row(self, key_column):
        min_val = float("inf")
        min_i = 0
        for i in range(1, len(self.coeff_matrix)):
            if self.coeff_matrix[i][key_column] > 0:
                val = self.coeff_matrix[i][-1] / self.coeff_matrix[i][key_column]
                if val < min_val:
                    min_val = val
                    min_i = i
        if min_val == float("inf"):
            raise ValueError("Unbounded problem")
        if min_val == 0:
            warn("Degeneracy")
        return min_i

    def normalize_to_pivot(self, key_row, pivot):
        for i in range(len(self.coeff_matrix[0])):
            self.coeff_matrix[key_row][i] /= pivot

    def make_key_column_zero(self, key_column, key_row):
        num_columns = len(self.coeff_matrix[0])
        for i in range(len(self.coeff_matrix)):
            if i != key_row:
                factor = self.coeff_matrix[i][key_column]
                for j in range(num_columns):
                    self.coeff_matrix[i][j] -= self.coeff_matrix[key_row][j] * factor

    def delete_r_vars(self):
        for i in range(len(self.coeff_matrix)):
            non_r_length = self.num_vars + self.num_s_vars + 1
            length = len(self.coeff_matrix[i])
            while length != non_r_length:
                del self.coeff_matrix[i][non_r_length-1]
                length -= 1

    def update_objective_function(self):
        self.coeff_matrix[0] = [Fraction("0/1") for i in range(self.total_vars + 1)]
        objective_function_coeffs = self.objective_function.split()
        for i in range(len(objective_function_coeffs)):
            if '_' in objective_function_coeffs[i]:
                coeff, index = objective_function_coeffs[i].split('_')
                if objective_function_coeffs[i-1] == '-':
                    self.coeff_matrix[0][int(index)-1] = Fraction("-" + coeff[:-1] + "/1")
                else:
                    self.coeff_matrix[0][int(index)-1] = Fraction(coeff[:-1] + "/1")

    def check_alternate_solution(self):
        for i in range(len(self.coeff_matrix[0])):
            if self.coeff_matrix[0][i] and i not in self.basic_vars[1:]:
                warn("Alternate Solution exists")
                break

    def name_variables(self):
        var_names = list()
        for i in range(1, self.num_vars+1):
            var_names.append("x_{}".format(i))
        for i in range(0, self.num_s_vars):
            var_names.append("h_{}".format(self.s_rows[i]))
        for i in range(0, self.num_r_vars):
            var_names.append("a_{}".format(self.r_rows[i]))
        return var_names

    def tableau_tex_header(self):
        str = "\\begin{center}\n \\begin{tabular}{c|c|"
        str += "c"* (self.num_vars + self.num_s_vars + self.num_r_vars)
        str += "|}\n"
        str += " & $z$"
        for i in range(self.num_vars + self.num_s_vars + self.num_r_vars):
            str += " & ${}$".format(self.var_names[i])
        str += "\\\\ \n"
        return str

    def tableau_tex_wrap(self):
        return "\end{tabular}\n\end{center}\n"

    def tableau_tex_from_coeff_matrix(self):
        str = ""
        for i, row in enumerate(self.coeff_matrix):
            if i == 1:
                str += "\hline\n"
            if i != 0:
                str += "${}$".format(self.var_names[self.basic_vars[i]])
            if i == 0:
                str += " & {}".format(row[len(row) - 1])
            else:
                str += " & {}".format(row[len(row) - 1])
            for j in range(0, len(row) - 1):
                if i == 0:
                    str += " & {}".format(row[j])
                else:
                    str += " & {}".format(row[j])
            str += "\\\\ \n"
        str += "\hline\n"
        return str

    def formulation_tex(self):
        tex_str = "\\begin{equation}\n\\begin{split}\n"
        if 'min' in self.objective.lower():
            tex_str += "\mbox{min. } z = " + self.objective_function + "\\\\\n"
        else:
            tex_str += "\mbox{max. } z = " + self.objective_function + "\\\\\n"
        tex_str += "s.a.:\\\\\n"
        char_to_replace = {'<=': '\\leq',
                           '>=': '\\geq'}
        for expression in self.constraints:
            # Iterate over all key-value pairs in dictionary
            for key, value in char_to_replace.items():
                # Replace key character with value character in string
                expression = expression.replace(key, value)
            tex_str += expression + "\\\\\n"
        for i in range(1, self.num_vars+1):
            tex_str += "x_{}".format(i)
            if i!=self.num_vars:
                tex_str += ",\\,\\,"
        tex_str+="\geq 0\\\\\n"
        tex_str += "\\end{split}\n\\end{equation}"
        return tex_str

    def compose_tableaux(self, first_tableau=0, last_tableau=float('inf')):
        last_tableau = min(len(self.tableaux_list), last_tableau)
        first_tableau = max(0, min(first_tableau, last_tableau-1))
        str = self.tableau_tex_header()
        for i in range(first_tableau, last_tableau):
            str += self.tableaux_list[i]
        str += self.tableau_tex_wrap()
        return str

    def get_base(self):
        base = self.A[:, self.basic_vars[1:]]
        return base

def sum_rows(row1, row2):
    row_sum = [0 for i in range(len(row1))]
    for i in range(len(row1)):
        row_sum[i] = row1[i] + row2[i]
    return row_sum

def max_index(row):
    print("row in max_index function")
    print(row)
    max_i = 0
    for i in range(0, len(row)):
        print("{}, evaluated item".format(row[i]))
        if row[i] > row[max_i]:
            max_i = i
    return max_i

def multiply_const_row(const, row):
    mul_row = []
    for i in row:
        mul_row.append(const*i)
    return mul_row

def min_index(row):
    min_i = 0
    for i in range(0, len(row)):
        if row[min_i] > row[i]:
            min_i = i
    return min_i

def array_to_fraction(arr):
    to_fraction = lambda t: Fraction(t).limit_denominator()
    vfunc = np.vectorize(to_fraction)
    return vfunc(arr)



