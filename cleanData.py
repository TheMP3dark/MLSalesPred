import pandas as pd
from sklearn.preprocessing import Imputer as Imp

# read data from the sheet of our excel document that contains the data
data = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")

data = data.rename(columns={"Unnamed: 7": "First_Name"})

# fill missing values wherever possible, drop the rest
data = data.dropna(subset=["Agent_ID", "First_Name", "Last_Name", "Area_Code", "Sale"])

# make a new column to represent sale/gender values as 1(true) or 0(false)
saleNew = []
genderNew = []
for items in data.Sale:
    if items is True:
        saleNew.append(1)
    elif items is False:
        saleNew.append(0)
    elif items == "NA":
        saleNew.append(None)
    else:
        saleNew.append(None)
for items in data.Gender:
    if items == "Male":
        genderNew.append(1)
    elif items == "Female":
        genderNew.append(0)
    elif items == "Others":
        genderNew.append(2)
    else:
        genderNew.append(None)

# insert the new boolean sale column and drop the older Sale column
data["saleBool"] = saleNew
data["genderBool"] = genderNew
data = data.drop(columns=["Sale", "Gender"])

# using imputer (mode) to fill in missing data for gender
fillData = Imp(missing_values="NaN", strategy="most_frequent", axis=1)
fillData.fit([data.genderBool])
allFillData = fillData.transform([data.genderBool])
genderFillData = allFillData[0]
data["genderBool"] = genderFillData

# for columns in data:
#     print(data[f"{columns}"])
#     plt.scatter(data[f"{columns}"], data.saleBool)

# print(data.genderBool)

if __name__ == "__main__":
    # check for null values
    print(data.isnull().sum())
