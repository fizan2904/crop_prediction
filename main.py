import datetime
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

def selection(data_id):
    if data_id == 1:
        data = pd.read_csv("./arjpaddy.csv")
        data_test = pd.read_csv("./arjpaddy_test.csv")
        title = "ARJUNANADHI PADDY"
    elif data_id == 2:
        data = pd.read_csv("./arjmaiz.csv")
        data_test = pd.read_csv("./arjmaiz_test.csv")
        title = "ARJUNANADHI MAIZ"
    elif data_id == 3:
        data = pd.read_csv("./arjcereals.csv")
        data_test = pd.read_csv("./arjcereals_test.csv")
        title = "ARJUNANADHI CEREALS"
    elif data_id == 6:
        data = pd.read_csv("./kscereals.csv")
        data_test = pd.read_csv("./kscereals_test.csv")
        title = "KOUSIKANADHI CEREALS"
    elif data_id == 5:
        data = pd.read_csv("./ksmaiz.csv")
        data_test = pd.read_csv("./ksmaiz_test.csv")
        title = "KOUSIKANADHI MAIZ"
    elif data_id == 4:
        data = pd.read_csv("./kspaddy.csv")
        data_test = pd.read_csv("./kspaddy_test.csv")
        title = "KOUSIKANADHI PADDY"
    else:
        return
    return predict(data, data_test, title)

def menu():
    print "\t\tAgriculture Prediction"
    print "\t\t\tMenu"
    print "\t\t1. ARJUNANADHI PADDY"
    print "\t\t2. ARJUNANADHI MAIZ"
    print "\t\t3. ARJUNANADHI CEREALS"
    print "\t\t4. KOUSIKANADHI PADDY"
    print "\t\t5. KOUSIKANADHI MAIZ"
    print "\t\t6. KOUSIKANADHI CEREALS"
    print "\t\tPress q or Q to quit"
    data_id = raw_input("Enter the index number of required crops prediction for 2013-2017: ")
    if data_id == "q" or data_id == "Q":
        return
    selection(int(data_id))

def predict(data, test_data, title):
    x_train = data[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', 'AREA UNDER CULTIVATION']]
    y_train = data[['YIELD']]

    x_test = test_data[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', 'AREA UNDER CULTIVATION']]
    y_test = test_data[['YIELD']]

    ols = linear_model.LinearRegression()
    model = ols.fit(x_train, y_train)

    y = model.predict(x_test)[0:5]
    dates = [2013, 2014, 2015, 2016, 2017]

    plt.plot(dates, y)
    plt.xlabel("Years(0 for 2013, 4 for 2017)")
    plt.ylabel("PCA Components 3D data")
    plt.title(title)
    plt.show()
    _ = raw_input("<Hit Enter To Continue>")
    plt.close()
    menu()

menu()