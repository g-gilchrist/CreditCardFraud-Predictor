# Grant Gilchrist
# 06/08/2022 
# CS379-2203A-01 - Credit Fraud
# This is a supervised machine learning algorithm that utilizes data from an .arff file
# to predict credit fraud. The dataset has no null values and consists of both numerical 
# and categorical data. I chose a Random Forest algorithm to predict fraud. To simplify
# the preprocessing the categorical data was converted into multiple columns of booleans.



from scipy.io import arff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


#-------------------------------
# Load
#-------------------------------
# This is a function that will load .arff files returning a pandas dataframe

def load(file):
    
    # import the arff file
    data = arff.loadarff(file) 
    
    # turn the data into a pandas data frame
    df = pd.DataFrame(data[0]) 
    
    # ...return the dataframe
    return df


#-------------------------------
# Random Forest Algorithm
#-------------------------------
# This function will preform a randomForest Algorithm on the dataset provides
# as an argument and return the accuracy of the alogorithms predictions

def randomForest(df):

    # ” : ” means it will select all rows, “0:60” means that it will select columns all the columns with the index 0-60
    x = df.iloc[:, 0:60] 
    # ” : ” means it will select all rows, “61:62 ” means that it will ignore all columns except columns witht he 61 & 62 index
    # because this column was categorical with only two options I only have to use one column, the boolean will approprately classify the data
    y = df.iloc[:, 61] 

    # Split the Test and Training Data 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Creates a Scaler object
    sc = StandardScaler()
    
    # transforms the data to a scaled version to prevent bias
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # create regressor object
    # n_estimators is the number of trees in the forest
    # random_state is a seed for how the RanomForestClassifier is randomized
    classifier = RandomForestClassifier(n_estimators = 64, random_state = 5)
    
    # fit the regressor with x and y training data
    classifier.fit(X_train, y_train)
    
    # predicts a potential fraudulent account based on the X_test data
    y_pred = classifier.predict(X_test)
    
    # Computes the accuracy by comparing the actual outcome versus the predicted outcome
    accuracy = accuracy_score(y_test, y_pred )  

    # ...returns the variable accuracy which scores the algorthms... accuracy
    return accuracy

    
#-------------------------------
# Main
#-------------------------------
# This is the main function of the application

def main():
    
    # calls the load function to load the .arff file
    df = load('creditfraud.arff')
    
    print(df.info(), '\n')

    
    # converts categorical data to a column of booleans for each entry, do not use this if there are many different categories
    df = pd.get_dummies(df)  

    # check for null values 
    print(df.info(), '\n')
    
    # turning the accuracy into a percentage
    accuracy = randomForest(df) * 100
    
    # printing the alogirthms accuracy to the first decimal place
    print(f'Accuracy: {accuracy:.1f}%')

    
main()