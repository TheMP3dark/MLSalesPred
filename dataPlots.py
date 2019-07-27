import matplotlib.pyplot as plt
from cleanData import data

if __name__ == "__main__":
    print(data.columns)

    # we can plot graphs only for numeric columns
    print(data.dtypes)
    numericDataCols = ["Age", "Product_ID", "Timezone", "Phone_code", "Call_Count", "saleBool", "genderBool"]

    for cols in numericDataCols:
        print(cols)
        plt.scatter(data[f"{cols}"], data.saleBool)
        plt.show()
        plt.boxplot(data[f"{cols}"])
        plt.show()

# plots reveal that some attributes are not varying / have on effect on sales
# we drop these attributes

data = data.drop(columns=["Timezone", "Phone_code"])
