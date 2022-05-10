from turtle import width
import streamlit as st
import seaborn as sns
import os

def show_explore_page():
    st.header("Data exploratory Analysis")
    st.write("------------")

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

    st.write("Now we are looking at the result of the encoding.")
    st.write(df.head())

    import numpy as np
    import matplotlib.pyplot as plt

    st.write("Now we are going to see the correlation between each variable")
    st.write("### Here is the number of charges")
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 4))
    fig = sns.displot(data = df, x = df['charges'], kde=True)
    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


    st.write("### Correlation between charges and smoking")
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 4))
    fig = sns.displot(df.loc[df.smoker == 1]['charges'].values.tolist(), kde=True)
    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    st.write("### Correlation between charges and non-smoking")
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 4))
    fig = sns.displot(df.loc[df.smoker == 0]['charges'].values.tolist(), kde=True)
    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    
    st.write("### Age data")
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(10, 4))
    fig = sns.displot(data = df, x = df["age"], kde=True)
    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    
    st.write("### Charge vs Age")
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 4))
    sns.scatterplot(x= df['age'] , y= df['charges']).set_title('Charges vs Age without filter')
    st.write(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    
    st.write("### Charge vs Age filtered by sex")
    st.write("1 is for male and 0 for female.")
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 4))
    sns.scatterplot(x= df['age'] , y= df['charges'], hue='sex', data=df).set_title('Charges vs Age filter by sex')    
    st.write(fig)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    
    st.write("### Charge vs Age filtered by smoker")
    st.write("1 is smoker and 0 for non-smoker")
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 4))
    sns.scatterplot(x= df['age'] , y= df['charges'], hue='smoker', data=df).set_title('Charges vs Age filtered by smoker')
    st.write(fig)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    
    st.write("### Heatmap Corellation")
    fig, ax = plt.subplots()
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), ax=ax, annot=True, cmap = 'YlGnBu')
    st.write(fig)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()
    corr = df.corr()[['charges']].sort_values(by='charges', ascending=False)
    plt.figure(figsize=(8, 12))
    sns.heatmap(corr, vmin=-1, vmax=1, ax=ax, annot=True, cmap = 'BrBG')
    st.write(fig)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # x = np.random.normal(size = 1000)
    # plt.hist(x, bins=50)
    # plt.savefig("x")
    # st.image("x.png")
    # os.remove("x.png")

    st.write("## Machine Learning Modelling")
    st.write("We are using **Random Forest Regression** for making our prediction.")
    st.write("We have determined the **independent variables**:")
    X = df.iloc[:, :-1]
    for col in X.columns:
        st.write("- {col} ".format(col = col))
    # X.head()

    st.write("Also the **dependent variable**:")
    st.write("- charges")
    y = df.iloc[:, -1]
    # y.head()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    st.write("Then the data were split into 80% training data set and 20% test data set.")
    st.write("**X_train**")
    st.write(X_train.head())
    st.write("**X_test**")
    st.write(X_test.head())
    st.write("**y_train**")
    st.write(y_train.head())
    st.write("**y_test**")
    st.write(y_test.head())

    # from sklearn.tree import DecisionTreeRegressor
    # regressor = DecisionTreeRegressor(random_state=0)
    # regressor.fit(X, y)

    from sklearn.ensemble import RandomForestRegressor
    random_for_reg = RandomForestRegressor(random_state=0)
    random_for_reg.fit(X_train,y_train)
    y_pred = random_for_reg.predict(X_test)

    st.write("## Results")
    st.write("------------")

    fig, ax = plt.subplots()
    plt.scatter(random_for_reg.predict(X_train), y_train, edgecolor="k", c = "yellow",  label = "Training data")
    plt.scatter(random_for_reg.predict(X_test), y_test, edgecolor="k", c = "red",  label = "Test data")
    plt.title("Train data vs Test data")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.legend(loc = "upper left")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.scatter(range(len(y_test)), y_test, edgecolor="k", color='yellow', label= "actual value")
    plt.scatter(range(len(y_pred)), y_pred, edgecolor="k", color='red', label = "prediction")
    plt.title("y_test vs prediction")
    plt.xlabel("Range")
    plt.ylabel("Charge")
    plt.legend(loc = "upper left")
    st.pyplot(fig)

    from sklearn import metrics
    st.write('The R2 (Variance) is: ', metrics.r2_score(y_test, y_pred))
    st.write('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
    st.write('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
    st.write('Root Mean Squared Error (RMSE) is: ', metrics.mean_squared_error(y_test, y_pred, squared=False))

