############################################
############################################
##
## The MyCode.py file is used to store code
## that is useful for future programming.
##
## John Akwei, ECMp ERMp Data Scientist
## ContextBase, contextbase.github.io
## johnakwei1@gmail.com
##
############################################
############################################

############################################
############################################
##
## 1) Terms and Conditions
##
############################################
############################################

# 1) The Vendor, (John Akwei, ECMp ERMp Data Scientist, ContextBase), has
# established a record of successful R programming projects via the Internet
# that have met, or exceeded, the expectations of the Clients. Verification of
# this is found at www.fiverr.com/johnakwei.

# 2) The Client, (or "Buyer"), agrees that lack of knowledge, (by the Client), of
# the R programming language is not grounds for cancellation, or non-payment, of
# the R language deliverable by the Vendor, (or "Seller").

# 3) The Client agrees that lack of knowledge of RStudio is not grounds for
# cancellation, or non-payment, of the R language deliverable by the Vendor.

# 4) The Vendor provides help with the R programming language, and with RStudio,
# upon delivery of the Client's ordered R language software.

# 5) Sofware deliverables, (by the Vendor), are assured to operate without errors,
# and proof of error-free operation of Vendor-provided software is also a free
# inclusion with delivery of the internet project.

# 6) The Client agrees that certain Data Science projects, (i.e. Natural Language
# Processing, Webscraping, Predictive Analytics), are continuously refinable and
# the capabilities of the delivered software is dependent on the time/budget
# scope of the project. Therefore, the Client agrees to reasonable software
# capabilities in keeping with the time/budget scope of the project.

# 7) The Client's agreement to the above Terms and Conditions is made by
# acceptance of the Custom Offer from the Vendor.

############################################
############################################
##
## 2) Session Information
##
############################################
############################################
import sys
print('Python %s on %s' % (sys.version, sys.platform))

import platform
my_system = platform.uname()

print(f"System: {my_system.system}")
print(f"Node Name: {my_system.node}")
print(f"Release: {my_system.release}")
print(f"Version: {my_system.version}")
print(f"Machine: {my_system.machine}")
print(f"Processor: {my_system.processor}")

############################################
############################################
##
## 3) Online Publishing
##
############################################
############################################

# ContextBase Logo
# <img src="ContextBase_Logo.jpg" alt="ContextBase Logo"  width="550" height="300">

# HTML code for markdown documents
# <br />
#
# <h1 align="center" style="color:blue;font-weight:bold;">Working Directory,
# and Required Packages</h1>

# Publish to Jupyter Notebooks

# Royalty-free images: pixabay, stocksnap, pexel and unsplash.
# http://stats.stackexchange.com/
# https://data.library.virginia.edu/diagnostic-plots/
# http://www.gardenersown.co.uk/education/lectures/r/correl.htm#correlation
# http://rpubs.com/etanabe/musical_taste_likelihood

############################################
############################################
##
## 4) Education
##
############################################
############################################

# Resources
# https://wiki.python.org/moin/SimplePrograms
# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

############################################
############################################
##
## 5) Set path to Project Directory
##
############################################
############################################

# Create a virtual environment
# Press Ctrl+Alt+S to open the project Settings/Preferences.

# In the Settings/Preferences dialogCtrl+Alt+S, select Project <project name> |
# Python Interpreter. Click the The Configure project interpreter icon and select Add.

# In the left-hand pane of the Add Python Interpreter dialog, select Virtualenv
# Environment. The following actions depend on whether the virtual environment existed before.

# If New environment is selected:
# Specify the location of the new virtual environment in the text field, or
# click Virtual environment location and find location in your file system. Note that the directory where the new virtual environment should be located, must be empty!

# Choose the base interpreter from the list, or click Choose the base interpreter and
# find a Python executable in the your file system.

# If PyCharm detects no Python on your machine, it provides two options:
# to download the latest Python versions from python.org or to specify a path to
# the Python executable (in case of non-standard installation).

# Select the Inherit global site-packages checkbox if you want to inherit your
# global site-packages directory. This checkbox corresponds to the --
# system-site-packages option of the virtualenv tool.

# Select the Make available to all projects checkbox, if needed.

# If Existing environment is selected:

# Expand the Interpreter list and select any of the existing interpreters.
# Alternatively, click Select an interpreter and specify a path to the Python
# executable in your file system, for example, C:\Python36\python.exe.

# Select the checkbox Make available to all projects, if needed.

# Click OK to complete the task.

############################################
############################################
##
## 6) Load Required Packages
##
############################################
############################################

# when you call the statement import numpy as np , you are shortening the phrase "numpy" to "np"
# to make your code easier to read. It also helps to avoid namespace issues.
# (tkinter and ttk are a good example of what can happen when you do have that issue.
import numpy as np

from time import localtime

############################################
############################################
##
## 7) Updating Python
##
############################################
############################################

# x.z (patch) Python version, just go to Python downloads page get the
# latest version and start the installation. Since you already have Python installed
# on your machine installer will prompt you for "Upgrade Now". Click on that button and
# it will replace the existing version with a new one.

############################################
############################################
##
## 8) Basic Python Scripting
##
############################################
############################################
# define variables
a: int = 10
b = 50

# first operation
c = a + b
print(a, "+", b, "=", c)

# second operation
d = a * b
print(a, "*", b, "=", d)

# Hello, World
a: str = 'World'
print('Hello,', a, '!')
print("Hello, World!")
print("Hello!")

hello: int = "0"
print(hello)

# Input, assignment
name = input('What is your name?\n')
print('Hi, %s.' % name)

# For loop, built-in enumerate function, new style formatting
friends = ['john', 'pat', 'gary', 'michael']
for i, name in enumerate(friends):
    print("iteration {iteration} is {name}".format(iteration=i, name=name))

# Fibonacci, tuple assignment
parents, babies = (1, 1)
while babies < 100:
    print('This generation has {0} babies'.format(babies))
    parents, babies = (babies, parents + babies)

############################################
############################################
##
## 9) Import data into Python
##
############################################
############################################
# Open .csv file
import csv

with open('BeerWineReviews.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)

# importing data into pandas
# https://www.datacamp.com/community/tutorials/pandas-read-csv?utm_source=adwords_ppc&utm_campaignid=1565261270&utm_adgroupid=67750485268&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332661264374&utm_targetid=aud-748597547652:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1014895&gclid=Cj0KCQiAtqL-BRC0ARIsAF4K3WEMwPpUzRW5HpTg4Gvhr6zE6l3I683DpKO6DYPMzHZ0mQCqoRGL8uEaAm1ZEALw_wcB
import pandas as pd
beerWine_df = pd.read_csv("BeerWineReviews.csv")

input_data = pandas.read_table("height.csv", header=0, sep=",", names=("weight", "height"))

############################################
############################################
##
## 10) Exploratory Data Analysis
##
############################################
############################################

# Basic numpy script - https://numpy.org/doc/stable/user/quickstart.html
import numpy as np

a = np.arange(15).reshape(3, 5)
a
a.shape
a.ndim
a.dtype.name
a.itemsize
a.size
type(a)

b = np.array([6, 7, 8])
b
type(b)

############################################
############################################
##
## 11) Tables
##
############################################
############################################

############################################
############################################
##
## 12) Functions
##
############################################
############################################

def greet(name):
    print('Hello', name)

greet('Jack')
greet('Jill')
greet('Bob')

# Import, regular expressions
import re

for test_string in ['555-1212', 'ILL-EGAL']:
    if re.match(r'^\d{3}-\d{4}$', test_string):
        print(test_string, 'is a valid US local phone number')
    else:
        print(test_string, 'rejected')

# Time, conditionals, from..import, for..else
from time import localtime

activities = {8: 'Sleeping',
              9: 'Commuting',
              17: 'Working',
              18: 'Commuting',
              20: 'Eating',
              22: 'Resting'}

time_now = localtime()
hour = time_now.tm_hour

for activity_time in sorted(activities.keys()):
    if hour < activity_time:
        print(activities[activity_time])
        break
else:
    print('Unknown, AFK or sleeping!')

# Triple-quoted strings, while loop
REFRAIN = '''
%d bottles of beer on the wall,
%d bottles of beer,
take one down, pass it around,
%d bottles of beer on the wall!
'''
bottles_of_beer = 9
while bottles_of_beer > 1:
    print(REFRAIN % (bottles_of_beer, bottles_of_beer,
                     bottles_of_beer - 1))
    bottles_of_beer -= 1

############################################
############################################
##
## 13) Classes
##
############################################
############################################

class BankAccount(object):
    def __init__(self, initial_balance=0):
        self.balance = initial_balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        self.balance -= amount

    def overdrawn(self):
        return self.balance < 0

my_account = BankAccount(15)
my_account.withdraw(50)
print(my_account.balance, my_account.overdrawn())

# Open .csv file
import csv

with open('BeerWineReviews.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)

# Interest Calculator
print('Interest Calculator:')

amount = float(input('Principal amount ?'))
roi = float(input('Rate of Interest ?'))
yrs = int(input('Duration (no. of years) ?'))

total = (amount * pow(1 + (roi / 100), yrs))
interest = total - amount
print('\nInterest = %0.2f' % interest)

# Dictionaries, generator expressions
prices = {'apple': 0.40, 'banana': 0.50}
my_purchase = {
    'apple': 1,
    'banana': 6}
grocery_bill = sum(prices[fruit] * my_purchase[fruit]
                   for fruit in my_purchase)
print('I owe the grocer $%.2f' % grocery_bill)

# Command line arguments, exception handling
# This program adds up integers that have been passed as arguments in the command line
import sys

try:
    total = sum(int(arg) for arg in sys.argv[1:])
    print('sum =', total)
except ValueError:
    print('Please supply integer arguments')

# Opening files
# indent your Python code to put into an email
import glob

# glob supports Unix style pathname extensions
python_files = glob.glob('*.py')
for file_name in sorted(python_files):
    print('    ------' + file_name)

    with open(file_name) as f:
        for line in f:
            print('    ' + line.rstrip())

    print()

# Basic numpy script - https://numpy.org/doc/stable/user/quickstart.html
import numpy as np

a = np.arange(15).reshape(3, 5)
a
a.shape
a.ndim
a.dtype.name
a.itemsize
a.size
type(a)

b = np.array([6, 7, 8])
b
type(b)

############################################
############################################
##
## 14) Array Creation
##
############################################
############################################
# import numpy as np

a = np.array([2, 3, 4])
a
a.dtype

b = np.array([1.2, 3.5, 5.1])
b.dtype

# Basic pandas script - https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
import pandas as pd

# Object Creation
# Creating a Series by passing a list of values, letting pandas create a default integer index:
s = pd.Series([1, 3, 5, np.nan, 6, 8])
s

# Creating a DataFrame by passing a NumPy array, with a datetime index and labeled columns:
dates = pd.date_range('20130101', periods=6)
dates

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df

# Creating a DataFrame by passing a dict of objects that can be converted to series-like.
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

# importing data into pandas
# https://www.datacamp.com/community/tutorials/pandas-read-csv?utm_source=adwords_ppc&utm_campaignid=1565261270&utm_adgroupid=67750485268&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332661264374&utm_targetid=aud-748597547652:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1014895&gclid=Cj0KCQiAtqL-BRC0ARIsAF4K3WEMwPpUzRW5HpTg4Gvhr6zE6l3I683DpKO6DYPMzHZ0mQCqoRGL8uEaAm1ZEALw_wcB
import pandas as pd
beerWine_df = pd.read_csv("BeerWineReviews.csv")

df2
df2.dtypes
df.head()
df.tail()
df.index
df.columns
df.to_numpy()

# For df2, the DataFrame with multiple dtypes, DataFrame.to_numpy() is relatively expensive.
df2.to_numpy()

# a quick statistic summary of your data:
df.describe()
df2.describe()

# Transposing your data:
df.T

# Sorting by an axis:
df.sort_index(axis=1, ascending=False)

# Sorting by values:
df.sort_values(by='B')

# Selecting a single column, which yields a Series, equivalent to df.A:
df['A']

# Selecting via bracket, which slices the rows.
df[0:3]
df['20130102':'20130104']

# Selection by Label. For getting a cross section using a label:
df.loc[dates[0]]

# Selecting on a multi-axis by label:
df.loc[:, ['A', 'B']]

############################################
############################################
##
## 15) Statistics
##
############################################
############################################

import numpy as np

a = np.array([10, 7, 14, 23, 15, 7, 32])
a
a.dtype

# Find the mean, median, mode, and range of the following dataset.

############################################
############################################
##
## 16) Linear Regression
##
############################################
############################################

from sklearn import linear_model
import pandas as pd
import pandas
import matplotlib.pyplot as plt

input_data = pandas.read_table("height.csv", header=0, sep=",", names=("weight", "height"))

plt.scatter(input_data["weight"], input_data["height"])
plt.show()

predictor = pd.DataFrame(input_data, columns=["weight"])
outcome = pd.DataFrame(input_data, columns=["height"])

lm = linear_model.LinearRegression()
lm_model = lm.fit(predictor, outcome)

predicted_heights = lm.predict(predictor)

r_squared = lm.score(predictor, outcome)

print(predicted_heights)
print("Predicted:")
print(predicted_heights[0:6])
print("Actual:")
print(outcome[0:6])
print(r_squared)

############################################
############################################
##
## 17) Logistic Regression
##
############################################
############################################



############################################
############################################
##
## 18) Polynomial Regression
##
############################################
############################################

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

print(r2_score(y, mymodel(x)))

speed = mymodel(17)
print(speed)

"""
# sklearn 's implementation
import pandas as pd

# Importing the dataset 
data = pd.read_csv('data.csv')
data

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Fitting Polynomial Regression to the dataset 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Visualising the Polynomial Regression results 
plt.scatter(X, y, color='blue')
plt.plot(X, lin2.predict(poly.fit_transform(X)), color='red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

"""
Created on Fri Dec 21 18:59:49 2018
@author: Nhan Tran
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
# Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

viz_linear()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

viz_polymonial()

# Additional feature
# Making the plot line (Blue one) more smooth
def viz_polymonial_smooth():
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape(len(X_grid),
                            1)  # Why do we need to reshape? (https://www.tutorialspoint.com/numpy/numpy_reshape.htm)
    # Visualizing the Polymonial Regression results
    plt.scatter(X, y, color='red')
    plt.plot(X_grid, pol_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

viz_polymonial_smooth()

# Predicting a new result with Linear Regression
lin_reg.predict([[5.5]])
# output should be 249500

# Predicting a new result with Polymonial Regression
pol_reg.predict(poly_reg.fit_transform([[5.5]]))
# output should be 132148.43750003

############################################
############################################
##
## 19) Naive Bayes
## https://www.facebook.com/1026788430809320/videos/832542214221735
##
############################################
############################################

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,[4]].values

# Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

############################################
############################################
##
## 20) Decision Tree Classification
## https://www.facebook.com/1026788430809320/videos/360230405081773
##
############################################
############################################
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,[4]].values

# Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Print accuracy of the predictions
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

############################################
############################################
##
## 21) Random Forest Classification
## https://www.facebook.com/1026788430809320/videos/907846826409491
##
############################################
############################################
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,[4]].values

# Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000,criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Print accuracy of the predictions
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

############################################
############################################
##
## 22) Support Vector Machines
## https://www.facebook.com/1026788430809320/videos/428328051573056
## https://colab.research.google.com/github/grochmal/daml/blob/master/nb/sl-complete-procedure.ipynb
##
############################################
############################################
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=50)
len(faces.data), faces.target_names

fig, axes = plt.subplots(1, 11, figsize=(16, 3))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
first_img = [np.argmax(faces.target == x) for x in list(range(len(faces.target_names)))]
for i, ax in enumerate(axes.flat):
    idx = first_img[i]
    ax.imshow(faces.data[idx].reshape(62, 47), cmap='gray')
    if i % 2:
        ax.set_title(faces.target_names[i], fontsize=10)
    else:
        ax.set_title(faces.target_names[i], fontsize=10, y=-0.2)
    ax.axis('off')
    
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(faces.data, faces.target, test_size=0.2)

from sklearn.model_selection import KFold, GridSearchCV
model = make_pipeline(PCA(n_components=128, svd_solver='randomized'), SVC())
param_grid = [
    {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100]},
    {'svc__kernel': ['rbf'], 'svc__C': [1, 10, 100], 'svc__gamma': [0.1, 1.0, 10.0]}
]
grid = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, shuffle=True))
grid.fit(xtrain, ytrain)
grid.best_score_, grid.best_estimator_

from sklearn.metrics import classification_report
y_hat = grid.best_estimator_.predict(xtest)
print(classification_report(ytest, y_hat, target_names=faces.target_names))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_hat)
fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(mat, cmap='summer')
ticks = np.arange(0,len(faces.target_names))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(faces.target_names, rotation=45, ha='right')
ax.set_yticklabels(faces.target_names, rotation=45, ha='right')
ax.set_ylabel('true label')
ax.set_xlabel('predicted label')
ax.xaxis.set_ticks_position('bottom')

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, mat[i, j], ha='center', va='center')
        
        fig, axes = plt.subplots(2, 6, figsize=(16, 5))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
names = faces.target_names
for i, ax in enumerate(axes.flat):
    ax.imshow(xtest[i].reshape(62, 47), cmap='gray')
    ax.set_title('true: %s\npredicted: %s' % (names[ytest[i]], names[y_hat[i]]), fontsize=10)
    ax.axis('off')

##################################################
##################################################
##
## 23) Feature Scaling, Feature Normalization
##
##################################################
##################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('wine_data.csv', header=None, usecols=[1,2])
df.columns=['Alcohol', 'Malic acid']
df.head()
df.describe()

plt.figure(figsize=(8,6))
plt.scatter(df['Alcohol'], df['Malic acid'],
            color='Maroon',label='input scale', alpha=0.5)
plt.title('Alcohol and Malic Acid content of the wine dataset')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.legend(loc='upper left')
plt.grid()
plt.show()

# StandardScaler
std_scale = preprocessing.StandardScaler().fit(df[['Alcohol', 'Malic acid']])
df_std = std_scale.transform(df[['Alcohol', 'Malic acid']])

print('Mean after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'
      .format(df_std[:,0].mean(), df_std[:,1].mean()))
print('\nStandard deviation after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'
      .format(df_std[:,0].std(), df_std[:,1].std()))

# MinMaxScaler
minmax_scale = preprocessing.MinMaxScaler().fit(df[['Alcohol', 'Malic acid']])
df_minmax = minmax_scale.transform(df[['Alcohol', 'Malic acid']])

print('Min-value after min-max scaling:\nAlcohol={:.2f}, Malic acid={:.2f}'
      .format(df_minmax[:,0].min(), df_minmax[:,1].min()))
print('\nMax-value after min-max scaling:\nAlcohol={:.2f}, Malic acid={:.2f}'
      .format(df_minmax[:,0].max(), df_minmax[:,1].max()))

# Visualization after Normalization
plt.figure(figsize=(8,6))
plt.scatter(df['Alcohol'], df['Malic acid'],
        color='maroon', label='input scale', alpha=0.3)
plt.scatter(df_std[:,0], df_std[:,1], color='red',
        label='Standardized', alpha=0.3)
plt.scatter(df_minmax[:,0], df_minmax[:,1],
        color='black', label='min-max scaled [min=0, max=1]', alpha=0.2)
plt.title('Alcohol and Malic Acid content of the wine dataset')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

df_std[:,0].mean()
df_minmax[:,0].max()

# Simple Python
x = [1,4,5,6,6,2,3]
mean = sum(x)/len(x)
std_dev = (1/len(x) * sum([ (x_i - mean)**2 for x_i in x]))**0.5
z_scores = [(x_i - mean)/std_dev for x_i in x]
print('Standard Scaler\n', z_scores)

# Min-Max scaling
minmax = [(x_i - min(x)) / (max(x) - min(x)) for x_i in x]
print('MinMax Scaler\n', minmax)

# Using Numpy
# Standardization
x_np = np.asarray(x)
z_scores_np = (x_np - x_np.mean()) / x_np.std()
print(z_scores_np)

# Min-Max scaling
np_minmax = (x_np - x_np.min()) / (x_np.max() - x_np.min())
print(np_minmax)

############################################
############################################
##
## 24) Machine Learning
##
############################################
############################################

# Python version
import sys
print('Python: {}'.format(sys.version))

# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))

# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))

# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))

# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Exploratory Data Analysis
print(dataset.shape)

# head
print(dataset.head(5))

# Statistical Summary
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# Data Visualization - Univariate Plots - box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# Multivariate Plots
scatter_matrix(dataset)
pyplot.show()

# Create a Validation dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Build Models - Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Select The Best Models
# Support Vector Machines (SVM) has the largest estimated accuracy score at about 0.98 or 98%.
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate Predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

############################################
############################################
##
## 25) Neural Networks / Deep Learning
## https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24
##
############################################
############################################
import numpy as np  # helps with the math
import matplotlib.pyplot as plt  # to plot error during training

# input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])


# create NeuralNetwork class
class NeuralNetwork:
    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []
    # activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=25000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# create neural network
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

# create two new examples to predict
example = np.array([[1, 1, 0]])
example_2 = np.array([[0, 1, 1]])

# print the predictions for both examples
print(NN.predict(example), ' - Correct: ', example[0][0])
print(NN.predict(example_2), ' - Correct: ', example_2[0][0])

# plot the error over the entire training duration
plt.figure(figsize=(15, 5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

############################################
############################################
##
## 26) Random Forest Bagging in Machine Learning
## https://www.facebook.com/1026788430809320/videos/366294271469058
##
############################################
############################################

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

x.shape
y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=100)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

############################################
############################################
##
## 27) Goodness-Of-Fit
## https://www.facebook.com/1026788430809320/videos/2636605556652321
##
############################################
############################################
# Importing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Creating sample points
np.random.seed(0)
x = np.arange(-5, 5, 0.1)
y = -x ** 4 + np.random.normal(0, 0.5, len(x))
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

len(x)

# Dividing the data into training and testing sets
x_train = x[0:80]
y_train = y[0:80]
x_test = x[80:]
y_test = y[80:]

# Creating the function for polynomial curve fitting
def polynomial_fit(degree=1):
    return np.poly1d(np.polyfit(x_train, y_train, degree))

# Another function to fit and display results
def plot_polyfit(degree=1):
    p = polynomial_fit(degree)
    plt.scatter(x_train, y_train, label="Training Set")
    plt.scatter(x_test, y_test, label="Testing Set")
    curve_x = np.arange(min(x), max(x), 0.01)
    plt.plot(curve_x, p(curve_x), label="Polynomial of Degree()".format(degree))
    plt.xlim(-5, 5)
    plt.ylim(-625, np.max(y)+0.1)

    plt.legend
    plt.plot

# Plotting the curve fitting on degree 1
# Underfitting Example
plot_polyfit(1)

# Plotting the curve fitting on degree 2
plot_polyfit(2)

# Plotting the curve fitting on degree 3
plot_polyfit(3)

# Goodness of Fit Example
plot_polyfit(4)

# Overfitting Example
plot_polyfit(10)

############################################
############################################
##
## 28) Boosting
## https://www.facebook.com/1026788430809320/videos/1012130032630138
##
############################################
############################################

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))

############################################
############################################
##
## 29) Gradient Boosting
## https://www.facebook.com/1026788430809320/videos/3062436150650270
##
############################################
############################################

from sklearn.ensemble import GradientBoostingRegressor as GB
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

(X,y) = load_boston(return_X_y=True)
X.shape
y.shape

X_train,X_val,y_train,y_val = train_test_split(X,y, test_size=0.1)
M = GB()
M.fit(X_train, y_train)
mse = mean_squared_error(y_val, M.predict(X_val))
print(mse)

############################################
############################################
##
## 30) Stacking
## https://www.facebook.com/1026788430809320/videos/2686715038308486
##
############################################
############################################

# ensemble learning, explain what is stacking and its
# connection to deep learning

# Multi-Level Stacking is Deep Learning.

############################################
############################################
##
## 31) Stock Price Prediction (RNN)
## https://www.facebook.com/1026788430809320/videos/1012130032630138
##
############################################
############################################

# Stock Market Prediction for Beginners - Part 1
# https://www.facebook.com/1026788430809320/videos/983388598804233

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pf = "ticker_lookup.csv"
df = pd.read_csv(pf, delimiter=",", usecols=["date", "open", "high ", "close "])
print(df.head)

plt.figure(figsize=(18,9))
plt.plot(range(df.shape[0]),df['Open'])
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)

plt.plot(df['date'], df['open'])
plt.show()

# Stock Market Prediction for Beginners - Part 2
# https://www.facebook.com/1026788430809320/videos/742438423023216
D = df.iloc[:,1:2].values
D.shape

N = 10000
Tr_set = D[:N]
Ts_set = D[N:]

Tr_set.shape
Ts_set.shape

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

sc = MinMaxScaler(feature_range=(0,1))

Tr_scaled = sc.fit_transform(Tr_set)
Ts_scaled = sc.transform(Ts_set)

seq_len = 100
X_train = []
y_train = []
for i in range(seq_len, len(Tr_scaled)-seq_len):
    X_train.append(Tr_scaled[i-seq_len:i,0])
    y_train.append(Tr_scaled[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)

X_train.shape
y_train.shape

seq_len = 100
X_test = []
y_test = []
for i in range(seq_len, len(Ts_scaled)-seq_len):
    X_test.append(Ts_scaled[i-seq_len:i,0])
    y_test.append(Ts_scaled[i,0])

X_test,y_test = np.array(X_test),np.array(y_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_train.shape

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
X_test.shape

############################################
############################################
##
## 32) Text Mining
##
############################################
############################################

# Text Mining Code
reason2 <- sum(grepl("baseline", as.character(chicagobooking[,13])))
biostatsTwitter <- grep("biostats",lineTwitter)
lineTwitter[biostatsTwitter]

rquote <- "workspace loaded from"
chars <- strsplit(rquote, split = "")[[1]]

# Print without quotes
cat(dollar(TotalClintonDisbursements), "\n")

##################################################
##################################################
##
## 33) Plot Creation - Data Visualization
##
##################################################
##################################################

# Python for Data Visualization Using #Seaborg - Part 1/3
# https://www.facebook.com/AISciencesLearn/videos/545301642826018
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()

import matplotlib.pyplot as plt
plt.style.available
plt.style.use('seaborn-whitegrid')

x = np.linspace(0, 10, 1000)
y = np.sin(x)
fig = plt.figure()
ax = plt.axes()
ax.plot(x, y)

x = np.linspace(0, 10, 1000)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = y1 + y2
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)

plt.plot(x,y1, linestyle='-', color='blue')
plt.plot(x,y2, linestyle=':', color='g')
plt.plot(x,y3, linestyle='-.', color=(0.5,0.4,0.8))

plt.plot(x,y1,'--b')
plt.plot(x,y2,':g')
plt.plot(x,y3,'-.r')
plt.xlim(-5,15)
plt.ylim(-3,2.5)

plt.plot(x, x+1, '-k')
plt.plot(x,y1,'--k')
plt.axis('tight')
plt.plot(x,y1,'--k')
plt.axis('equal')

plt.plot(x,y1,'-.r')
plt.xlim(10,0)
plt.ylim(1.5,-1.5)

plt.plot(x,y1,'--b',label='sin(x)')
plt.plot(x,y2,':g',label='cos(x)')
plt.plot(x,y3,'-.r',label='sin(x)+cos(x)')
plt.axis('tight')
plt.title('Sin and Cos Plots')
plt.xlabel('x')
plt.ylabel('sin(x),cos(x),sin(x)+cos(x)')
plt.legend()

x = np.linspace(0,10,30)
y1 = np.sin(x)
plt.plot(x,y1,'v',color='k')

markers = ['o','.',',','x','+','v','^','<','>','s','d']
for m in markers:
    plt.plot(np.random.rand(5),np.random.rand(5),m,label=m)
    plt.legend()

# Scatter Plots
plt.scatter(x,y1,color='red', s=100,alpha=0.3,cmap='viridis')
plt.colorbar()

plt.scatter(x,y1,c=np.random.rand(30),s=100*np.random.rand(30),alpha=0.5,cmap='viridis')
plt.colorbar()

# Contour Plots
x = np.linspace(0,5,50)
y = np.linspace(0,5,40)
X,Y = np.meshgrid(x,y)
Z = 10*np.sin(x)+2*Y
plt.contourf(X,Y,Z,30,cmap='RdGy')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot')

# Python for Data Visualization Using #Seaborn - Part 2/3
# https://www.facebook.com/AISciencesLearn/videos/546010789617301
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('classic')
x = np.linspace(0,100,1000)
y = np.cumsum(np.random.randn(1000,5),0)
y.shape
x.shape
plt.plot(x,y)
plt.legend('ABCDE',ncol=2,loc='upper left')

sns.set()
plt.plot(x,y)
plt.legend('ABCDE',ncol=2,loc='upper left')

X = 50*np.random.randn(5000)
Y = 200*np.random.randn(5000)
Z = 100*np.random.randn(5000)+500
plt.hist(X,density=True,alpha=0.5)
plt.hist(Y,density=True,alpha=0.5)
plt.hist(Z,density=True,alpha=0.5)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data = pd.DataFrame({'A':X,'B':Y,'C':Z})
for col in data.columns:
    plt.hist(data[col],density=True,alpha=0.5)
    sns.kdeplot(data[col],shade=True)

for col in data.columns:
    sns.distplot(data[col])

with sns.axes_style('white'):
    sns.jointplot('A','B',data,kind='kde')

iris = sns.load_dataset('iris')
iris.head(60)
iris.species.unique()
iris.shape
iris.columns
sns.pairplot(iris,hue='species')

from mpl_toolkits.mplot3d import Axes3D

plt.style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.gca(projection='3d')
r = np.linspace(-2,2,100)**2+1
theta = np.linspace(-4*np.pi,4*np.pi,100)
X = r*np.sin(theta)
Y = r*np.cos(theta)
Z = np.linspace(-2,2,100)
ax.plot(X,Y,Z,label="3D plot")

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

sns.set()
fig = plt.figure()
ax = fig.gca(projection='3d')
n = 100
X = 32-23*np.random.rand(n)+23
Y = 100*np.random.rand(n)
Z = (-25+50)*np.random.rand(n)-50
ax.scatter(X,Y,Z,c='r',marker='o',s=100)
X = 32-23*np.random.rand(n)+23
Y = 100*np.random.rand(n)
Z = (-25+50)*np.random.rand(n)-50
ax.scatter(X,Y,Z,c='b',marker='^',s=100)

plt.style.use('bmh')
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.gca(projection='3d')
X,Y,Z = axes3d.get_test_data(0.05)
ax.plot_surface(X,Y,Z,cmap='coolwarm',alpha=1)

# Python for Data Visualization Using Bokeh - Part 3/3
# https://www.facebook.com/1026788430809320/videos/185555456041662

from bokeh.plotting import figure,output_file,show
output_file("demo.html")
p = figure(plot_width=800,plot_height=400,title='line')
p.line([1,2,3,4,5],[6,7,8,9,10],line_width=2)
show(p)

p = figure(plot_width=800,plot_height=400,title='example',
           x_axis_label='X',
           y_axis_label='Y')
x = np.linspace(0,10,30)
y1 = np.sin(x)
y2 = np.cos(x)
p.line(x,y1,legend_label='sin(x)')
p.circle(x,x,legend_label='y=x',fill_color='green',size=5)
p.triangle(x,y2,legend_label='cos(x)',size=3)
show(p)

##################################################
##################################################
##
## 34) Cartography
##
##################################################
##################################################

library(leaflet)
# http://rstudio.github.io/leaflet/

# a map with the default DSM tile layer
m <- leaflet() %>% addTiles()
m

m <- leaflet() %>%
  addTiles() %>%  # Add default OpenStreetMap map tiles
  addMarkers(lng=174.768, lat=-36.852, popup="The birthplace of R")
m  # Print the map

# or
m <- leaflet()
m <- addTiles(m)
m <- addMarkers(m, lng=174.768, lat=-36.852,
                popup="The birthplace of R")
m

# Set bounds
m %>% fitBounds(0, 48, 18, 50)

# Move the center to Hall
m <- m %>% setView(-93.65, 42.0205, zoom=17)
m

# Add PopUps
m <- m %>% addPopups(-93.65, 42.0205, "Here is the <b>Department
                     of Statistics</b>, ISU")
m

# Add Markers
m <- m %>% addMarkers(-93.65, 42.0211)
m

library(maps)
mapStates <- map("state", fill=T, plot=F)
leaflet(data=mapStates) %>% addTiles() %>%
  addPolygons(fillColor=topo.colors(10, alpha=NULL), stroke=F)

m <- leaflet() %>% setView(lng=-71.0589, lat=42.3601, zoom=12)
m %>% addTiles()

##################################################
##################################################
##
## 35) Applications
##
##################################################
##################################################

##################################################
##################################################
##
## 36) Twitter Analysis
##
##################################################
##################################################

############################################
############################################
##
## 37) Correlation Analysis with Cross-Tabulation
##
############################################
############################################

##################################################
##################################################
##
## 38) Web Scraping
##
##################################################
##################################################

# Web Scraping and Data Mining for Beginners
# https://www.facebook.com/1026788430809320/videos/791498004733165

import requests
from bs4 import BeautifulSoup
url = 'https://www.fifa.com/fifa-world-ranking/ranking-table/men/'
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
print('HTML Data is loaded')

soup.find("h1")

soup.find_all("h2")

soup.find_all("h2")[0].text

soup.find_all("span", class_="fi-t__nText")

soup.find_all("span", class_="fi-t__nText")[0].text

t_names = [i.text for i in soup.find_all("span", class_="fi-t__nText")]
t_rankings = [i.text for i in soup.find_all("td", class_="fi-table__td fi-table__points")]
t_prev_rankings = [i.text for i in soup.find_all("td", class_="fi-table__td fi-table__prevpoints")]
print('Data is converted into lists')

d_ranks = dict(zip(t_names, t_rankings))
d_prev_ranks = dict(zip(t_names, t_prev_rankings))
print('Data is converted into dictionaries')

d_ranks['Belgium']

d_prev_ranks['Belgium']

url = 'https://www.worldcoinindex.com/trending/overview'
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
print('HTML Data is loaded')

soup.find_all("h1")

table = soup.find_all("div", class_="tradetable overview-table")
table2 = soup.find_all("table", id_="tablesorter")

############################################
############################################
##
## 37) Google Colab
##
############################################
############################################

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "GUI App.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ym3iCDSTS-iY",
        "outputId": "9ed7c8e1-8919-4871-ff4f-fb6d8f495bd3"
      },
      "source": [
        "#@title String fields\r\n",
        "\r\n",
        "text = 'values' #@param {type:\"string\"}\r\n",
        "dropdown = '1st option' #@param [\"1st option\", \"2nd option\", \"3rd option\"]\r\n",
        "text_and_dropdown = '2nd option' #@param [\"1st option\", \"2nd option\", \"3rd option\"] {allow-input: true}\r\n",
        "\r\n",
        "print(text)\r\n",
        "print(dropdown)\r\n",
        "print(text_and_dropdown)"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "valuesmy\n",
            "1st option\n",
            "2nd option\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QRSnnGD1TdA6"
      },
      "source": [
        "#@title Raw fields\r\n",
        "\r\n",
        "raw_input = None #@param {type:\"raw\"}\r\n",
        "raw_dropdown = raw_input #@param [1, \"raw_input\", \"False\", \"'string'\"] {type:\"raw\"}\r\n",
        "\r\n",
        "print(raw_input)\r\n",
        "print(raw_dropdown)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4U90oI3vTdRx"
      },
      "source": [
        "#@title Date fields\r\n",
        "date_input = '2018-03-22' #@param {type:\"date\"}\r\n",
        "\r\n",
        "print(date_input)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oLWhq2y3TdTG",
        "outputId": "36c47bd5-7a7d-469d-bf8f-edc48f19179c"
      },
      "source": [
        "#@title Number fields\r\n",
        "number_input = 10.0 #@param {type:\"number\"}\r\n",
        "number_slider = 0 #@param {type:\"slider\", min:-1, max:1, step:0.1}\r\n",
        "\r\n",
        "integer_input = 10 #@param {type:\"integer\"}\r\n",
        "integer_slider = 3 #@param {type:\"slider\", min:0, max:100, step:1}\r\n",
        "\r\n",
        "print(number_input)\r\n",
        "print(number_slider)\r\n",
        "\r\n",
        "print(integer_input)\r\n",
        "print(integer_slider)"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "10.0\n",
            "0\n",
            "10\n",
            "3\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VpOAUrnATdYG",
        "outputId": "c247c640-e075-43b0-85e7-9ab54a1940b0"
      },
      "source": [
        "#@title Boolean fields\r\n",
        "boolean_checkbox = False #@param {type:\"boolean\"}\r\n",
        "boolean_dropdown = True #@param [\"False\", \"True\"] {type:\"raw\"}\r\n",
        "\r\n",
        "print(boolean_checkbox)\r\n",
        "print(boolean_dropdown)"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "False\n",
            "True\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4VYMCUqoTdZe"
      },
      "source": [
        "#@title After running this cell manually, it will auto-run if you change the selected value. { run: \"auto\" }\r\n",
        "\r\n",
        "option2 = \"A\" #@param [\"A\", \"B\", \"C\"]\r\n",
        "print('You selected', option2)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mooxDOn_Tdd3"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "x4HX24DaTdfd"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "e_ecXDnVTdjq"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7WZwKIzmTdlM"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "miSNZ6LBTdpj"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZAdUzlzQTdr7"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}

