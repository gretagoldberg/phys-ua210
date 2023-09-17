#question_4

import numpy as np
import cmath

def quadratic_formula(a, b, c):
    if a == 0:
        return "Your a value is zero. You cannot divide by zero"
    else:
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            x_positive = ((-1)*b+np.sqrt(discriminant))/(2*a)
            x_negative = ((-1)*b-np.sqrt(discriminant))/(2*a)
            solution1 = "the positive solution to the quadratic is " + str(x_positive)
            solution2 = "the negative solution to the quadratic is " + str(x_negative)
        else:
            x_positive = ((-1)*b+cmath.sqrt(discriminant))/(2*a)
            x_negative = ((-1)*b-cmath.sqrt(discriminant))/(2*a)
            solution1 = "the positive solution to the quadratic is " + str(x_positive)
            solution2 = "the negative solution to the quadratic is " + str(x_negative)
        return solution1, solution2

#first test solution
print(quadratic_formula(0.001,1000,0.001))

#quadratic formula in other form
def quadratic_formula_form2(a, b, c):
    if a == 0:
        return "Your a value is zero. You cannot divide by zero"
    else:
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            x_positive = 2*c/((-1)*b+np.sqrt(discriminant))
            x_negative = 2*c/((-1)*b-np.sqrt(discriminant))
            solution1 = "the positive solution to the quadratic is " + str(x_positive)
            solution2 = "the negative solution to the quadratic is " + str(x_negative)
        else:
            x_positive = 2*c/((-1)*b+cmath.sqrt(discriminant))
            x_negative = 2*c/((-1)*b-cmath.sqrt(discriminant))
            solution1 = "the positive solution to the quadratic is " + str(x_positive)
            solution2 = "the negative solution to the quadratic is " + str(x_negative)
        return solution1, solution2

#second test solution
print(quadratic_formula_form2(0.001,1000,0.001))


#these give different solutions because of the way floats are stored
#this causes precision issues

#rewritten selecting the positive solution from quadratic_forumula and negative solution from quadratic_formula_form2
def quadratic_formula_form3(a, b, c):
    if a == 0:
        return "Your a value is zero. You cannot divide by zero"
    else:
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            x_positive = ((-1)*b+np.sqrt(discriminant))/(2*a)
            x_negative = 2*c/((-1)*b-np.sqrt(discriminant))
            solution1 = "the positive solution to the quadratic is " + str(x_positive)
            solution2 = "the negative solution to the quadratic is " + str(x_negative)
        else:
            x_positive = ((-1)*b+cmath.sqrt(discriminant))/(2*a)
            x_negative = 2*c/((-1)*b-cmath.sqrt(discriminant))
            solution1 = "the positive solution to the quadratic is " + str(x_positive)
            solution2 = "the negative solution to the quadratic is " + str(x_negative)
        return solution1, solution2
    

#third test solution
print(quadratic_formula_form3(0.001,1000,0.001))