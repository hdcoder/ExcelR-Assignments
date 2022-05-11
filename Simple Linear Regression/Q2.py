
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def estimate_coef(x, y):

    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
 
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
 
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return (b_0, b_1)
 
def plot_regression_line(x, y, b):
    
    plt.scatter(x, y, color = "r",marker = "o", s = 30)
 
    y_pred = b[0] + b[1]*x
 
    plt.plot(x, y_pred, color = "b")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

Salary = pd.read_csv("Salary_Data.csv")

X = Salary["Years_Experience"]
Y = Salary["Salary"]

b = estimate_coef(X,Y)

print("Estimated coefficients: \nb_0 = {}  \nb_1 = {}".format(b[0], b[1]))
print("Y = {}X + {}".format(b[0],b[1]))

plot_regression_line(X,Y,b)