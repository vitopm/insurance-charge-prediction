import streamlit as st
import seaborn as sns
import os

def show_explore_page():
    st.header("Data exploratory Analysis")
    # st.subheader("Nothing is written here yet, hang on for a moment!")

    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv("insurance-cost.csv")
    st.write("First off all we have to check how the data looks like")
    st.write(df.head())
    st.write("Shape: {shape}".format(shape=df.shape))
    st.write("So there is **1338 records** with **7 columns** of data")

    st.write("Then we take a look at the missing data")
    st.write(df.isnull().sum())
    st.write("Here we can see that there is 2 missing data on each column bmi and smoker, in this step we can either remove the record that has the empty data, or we can fill them with the nearest value or their mean. But for now let's just drop these record with empty data.")

    df = df.dropna()
    st.write(df.isnull().sum())
    st.write("Shape: {shape}".format(shape=df.shape))
    st.write("After removing the records with empty value, we still have 1334 records, since we have 4 records dropped. The number of records that we still have is still plenty for us to do data exploratory analysis.")
    
    st.write("Let's see the numerical description of the current data")
    st.dataframe(df.describe())
    st.write("In the description above we still haven't see the summary of column sex, smoker, and region. This is because they are categorical data. Therefore we have to do some encoding to categorical data.")

    from sklearn.preprocessing import LabelEncoder

    st.write("We change the categorical data in column sex to numerical.")
    le_sex = LabelEncoder()
    st.write(df["sex"].unique())
    df["sex"] = le_sex.fit_transform(df["sex"])
    st.write(df["sex"].unique())

    st.write("We change the categorical data in column smoker to numerical.")
    le_smoker = LabelEncoder()
    st.write(df["smoker"].unique())
    df["smoker"] = le_smoker.fit_transform(df["smoker"])
    st.write(df["smoker"].unique())

    st.write("The last categorical data that we encode is region")
    le_region = LabelEncoder()
    st.write(df["region"].unique())
    df["region"] = le_region.fit_transform(df["region"])
    st.write(df["region"].unique())

    import numpy as np
    import matplotlib.pyplot as plt

    st.write("Now we are going to see the correlation between each variable")
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 4))
    fig = sns.displot(data = df, x = df['charges'], kde=True)
    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 4))
    fig = sns.displot(data = df.loc[df.smoker == 0], kde=True)
    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    
    
    x = np.random.normal(size = 1000)
    plt.hist(x, bins=50)
    plt.savefig("x")
    st.image("x.png")
    os.remove("x.png")


    
    X = df.iloc[:, :-1]
    X.head()

    y = df.iloc[:, -1]
    y.head()

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np 

    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, y)

    y_pred = regressor.predict(X)

    error = np.sqrt(mean_squared_error(y, y_pred))
    st.write(error)
