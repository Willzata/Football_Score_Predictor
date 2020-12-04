import csv 
import requests 
from datetime import datetime
import tensorflow as tf
from tensorflow import keras 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# stored initial dataset in github 
URL = "https://raw.githubusercontent.com/Tom-P-Michaelis/Football_Data/master/combinedfootball.csv"


# reading csv from git repository 
df1 = pd.read_csv(URL)

# changing data types to assigned dtypes
Convert_Dict = {  "AELO": float , "HELO" : float, "AwayTeam": str, "HomeTeam" : str}

df1 = df1.astype(Convert_Dict)


df1["Date"]= pd.to_datetime(df1["Date"], format="%d/%m/%Y")

##drop Null Values:
df1 = df1.dropna(how='any',axis=0) 

# sorting by date 
df1=df1.sort_values(by=['Date'], ascending=False)

Data_Lists = df1.values.tolist()

# Creating data on last 5 games for each match - some data will be incomplete (cut later)
for Team in range(len(Data_Lists)):

  Home_Name = Data_Lists[Team][2]
  Away_Name = Data_Lists[Team][3]

  # resetting data counters
  GS = 0 
  GC = 0
  SS = 0
  SC = 0 
  SOTS = 0 
  SOTC = 0
  counter = 0

 # for each team iterate over rest of past data, if name is same home or away take note of data. 
  for Match_Team in range(Team +1, len(Data_Lists)):

    
    # if team was playing home in a previous match
    if Data_Lists[Match_Team][2] == Home_Name:

      # mining relevant data 
      GS += Data_Lists[Match_Team][4]
      GC += Data_Lists[Match_Team][5]

      SS += Data_Lists[Match_Team][10]
      SC +=  Data_Lists[Match_Team][11]

      SOTS += Data_Lists[Match_Team][12] 
      SOTC += Data_Lists[Match_Team][13]

      counter +=1

    # if team was playing away 
    if Data_Lists[Match_Team][3] == Home_Name:

      GS += Data_Lists[Match_Team][5]
      GC += Data_Lists[Match_Team][4]

      SS += Data_Lists[Match_Team][11]
      SC +=  Data_Lists[Match_Team][10]
      
      SOTS += Data_Lists[Match_Team][13] 
      SOTC += Data_Lists[Match_Team][12]

      counter +=1 
    
    if counter == 5:
      # append if the counter has reached five and then step out 
      # creating temporary row to append to  
      row = Data_Lists[Team]

      row.append(GS)
      row.append(GC)

      row.append(SS)
      row.append(SC)

      row.append(SOTS)
      row.append(SOTC)

      # replacing original list with augmented temporary row 
      Data_Lists[Team] = row 
      
      break 

##################### Same loop but for away team ##############################
for Team in range(len(Data_Lists)):

  Home_Name = Data_Lists[Team][2]
  Away_Name = Data_Lists[Team][3]

  # resetting data counters
  GS = 0
  GC = 0
  SS = 0
  SC = 0 
  SOTS = 0 
  SOTC = 0

  counter = 0 


 # for each team iterate over rest of past data, if name is same home or away take note of data. 
  for Match_Team in range(Team +1, len(Data_Lists)):

    
    # if team was playing home in a previous match
    if Data_Lists[Match_Team][2] == Away_Name:

      # mining relevant data 
      GS += Data_Lists[Match_Team][4]
      GC += Data_Lists[Match_Team][5]

      SS += Data_Lists[Match_Team][10]
      SC +=  Data_Lists[Match_Team][11]

      SOTS += Data_Lists[Match_Team][12] 
      SOTC += Data_Lists[Match_Team][13]

      counter +=1

    # if team was playing away 
    if Data_Lists[Match_Team][3] == Away_Name:

      GS += Data_Lists[Match_Team][5]
      GC += Data_Lists[Match_Team][4]

      SS += Data_Lists[Match_Team][11]
      SC +=  Data_Lists[Match_Team][10]
      
      SOTS += Data_Lists[Match_Team][13] 
      SOTC += Data_Lists[Match_Team][12]

      counter +=1 
    
    if counter == 5:
      # append if the counter has reached five and then step out 
      # creating temporary row to append to  

      row = Data_Lists[Team]
      row.append(GS)
      row.append(GC)

      row.append(SS)
      row.append(SC)

      row.append(SOTS)
      row.append(SOTC)

      # replacing original list with augmented temporary row 
      Data_Lists[Team] = row 
      
      break 

# # data structure is now [Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG[7],HTAG,HTR,HS,AS,HST,AST,HF[14],AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,
#HELO, AELO, HGSL5, HGCL5, HSSL5, HSCL5, HSOTSL5, HSOTCL5, AGSL5, AGCL5, ASSL5, ASCL5, ASOTSL5, ASOTCL5]

Dataset_All = np.array(Data_Lists) #converting to array - nested list is not as good 
columns_to_delete = [0,1,7,8,9,14,15,18,19,20,21,22,23,24] #getting rid of useless feature columns
Dataset_All = np.delete(Dataset_All,columns_to_delete,axis=1)
Dataset_All[1] 

#Data format is now [HomeTeam[0],AwayTeam[1],FTHG[2],FTAG[3],FTR[4],HS[5],AS[6],HST[7],AST[8],HC[9],AC[10],
#HELO[11], AELO[12], HGSL5[13], HGCL5[14], HSSL5[15], HSCL5[16], HSOTSL5[17], HSOTCL5[18], AGSL5[19], AGCL5[20], ASSL5[21], ASCL5[22], ASOTSL5[23], ASOTCL5[24]]

# **EDIT**
#Dropping home team & Away team as classifier cant work with strings. Might have to a categorical encode or some other trick.... 
#Lots of different teams though so not sure
names =[0,1]
Dataset_All = np.delete(Dataset_All,names, axis=1)


##Shuffle Data##
from sklearn.utils import shuffle
Dataset_All = shuffle(Dataset_All, random_state = 1)

#Remove Results - Labels

Results = Dataset_All[:,2]
Dataset_All = np.delete(Dataset_All,2,axis=1)

Dataset_All = np.asarray(Dataset_All).astype('float32') #ensuring right array format

#Data format is now [HomeTeam[0],AwayTeam[1],FTHG[2],FTAG[3],HS[4],AS[5],HST[6],AST[7],HC[8],AC[9],
#HELO[10], AELO[11], HGSL5[12], HGCL5[13], HSSL5[14], HSCL5[15], HSOTSL5[16], HSOTCL5[17], AGSL5[18], AGCL5[19], ASSL5[20], ASCL5[21], ASOTSL5[22], ASOTCL5[23]]

#*EDIT WITHOUT TEAM NAMES**

#FTHG[0],FTAG[1],HS[2],AS[3],HST[4],AST[5],HC[6],AC[7],
#HELO[8], AELO[9], HGSL5[10], HGCL5[11], HSSL5[12], HSCL5[13], HSOTSL5[14], HSOTCL5[15], AGSL5[16], AGCL5[17], ASSL5[18], ASCL5[19], ASOTSL5[20], ASOTCL5[21]]

#Data_Lists
#Take care missing data, replace with mean. 

#numerical_columns = [2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] #All columns now categorical

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(Dataset_All[:, :])
Dataset_All[:, :] = imputer.transform(Dataset_All[:, :])

##A = 0. D = 1 . H = 2 
#Label encoding for results column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Dataset_All, Results, test_size = 0.3, random_state = 1)

#Removing all the columns that show the stats of the game, not last 5.
##As those aren't known before a game happens.
##Would only be useful for a RT model which makes updated predictions during a game, as it happens.
X_train = X_train[:,8:] 
X_test = X_test[:,8:]

##X_Train/X_Test are now in format: HELO[0], AELO[1], HGSL5[2], HGCL5[3], HSSL5[4], HSCL5[5], HSOTSL5[6], HSOTCL5[7], AGSL5[8], AGCL5[9], ASSL5[10], ASCL5[11], ASOTSL5[12], ASOTCL5[13]]

##Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion ='entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

