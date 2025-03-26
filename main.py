import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("superstore_sales.csv") # to read the dataset

# print(df.head().to_string()) # to get the first 5 rows of the dataset
# print(df.info()) # to get the information about the dataset
# print(df.describe()) # to get the summary of the dataset

# print(df.isnull().sum()) # to check the null values in the dataset

# print(df.duplicated()) # to check the duplicate values in the dataset
df.drop_duplicates(inplace=True) # to drop the duplicate values in the dataset



"""
Convert Data Types: Convert Order Date to a datetime format for time-based analysis.
Create New Columns: Add columns like Month and Year for deeper analysis.
"""
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y", dayfirst=True) # to convert the date column to datetime
df["Month"] = df["Order Date"].dt.month # to get the month column
df["Year"] = df["Order Date"].dt.year # to get the year column



"""
Sales Trends Over Time: This shows how sales have changed over the years.
"""
df.groupby("Year")["Sales"].sum().plot(kind="line") # to plot the sales by year
plt.title("Sales Trends Over Time") # to add the title
plt.xlabel("Year") # to add the x-axis label
plt.ylabel("Total Sales") # to add the y-axis label
# plt.show()
plt.close()



"""
Top-Performing Products: This identifies the best-selling products.
"""
top_products = df.groupby("Product Name")["Sales"].sum().nlargest(10) # to get the top 10 products by sales
top_products.plot(kind="bar", figsize=(10, 6)) # to plot the top 10 products by sales
plt.title("Top 10 Products by Sales") # to add the title
plt.xlabel("Product Name") # to add the x-axis label
plt.ylabel("Total Sales") # to add the y-axis label
# plt.show()
plt.close()



"""
Sales by Region: This shows which regions generate the most revenue
"""
region_sales = df.groupby("Region")["Sales"].sum() # to get the sales by region
region_sales.plot(kind="bar", figsize=(10, 6)) # to plot the sales by region
plt.title("Sales by Region") # to add the title
plt.xlabel("Region") # to add the x-axis label
plt.ylabel("Total Sales") # to add the y-axis label
# plt.show()
plt.close()



"""
Seasonality Analysis: This identifies seasonal trends (e.g., higher sales during holidays).
"""
monthly_sales = df.groupby("Month")["Sales"].sum() # to get the sales by month
monthly_sales.plot(kind="line", figsize=(10, 6)) # to plot the sales by month
plt.title("Monthly Sales Trends") # to add the title
plt.xlabel("Month") # to add the x-axis label
plt.ylabel("Total Sales") # to add the y-axis label
# plt.show()
plt.close()



"""
Heatmaps: Highlight correlations (e.g., sales by region and category).
"""
pvt_table = df.pivot_table(values="Sales", index="Region", columns="Category", aggfunc="sum") # to create a pivot table
sns.heatmap(pvt_table, annot=True, fmt=".1f", cmap="YlGnBu") # to plot the heatmap
plt.title("Sales by Region and Category") # to add the title
# plt.show()
plt.close()



"""
Predictive Analysis: Predictive analysis helps forecast future sales, which is valuable for planning.
Prepare Data: Use Order Month as the independent variable and Sales as the dependent variable.
Train Model: Train a linear regression model to predict Sales based on Order Month.
Evaluate Model: Evaluate the model's performance using Mean Squared Error.
"""
df["Order Month"] = df["Order Date"].dt.to_period("M").astype(int) # to get the order month column
X = df[["Order Month"]] # to get the independent variable
y = df["Sales"] # to get the dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # to split the data into training and testing sets
model = LinearRegression() # to create a linear regression model
model.fit(X_train, y_train) # to train the model
y_pred = model.predict(X_test) # to predict the sales
mse = mean_squared_error(y_test, y_pred) # to calculate the mean squared error
# print("Mean Squared Error:", mse) # to print the mean squared error

"----------------------------------------------------------------------------------------------------"

"""
Sales Distribution by Region: This shows how sales are distributed among different regions.
"""
df.groupby("Region")["Sales"].sum().sort_values(ascending=False).plot(kind="bar")
plt.title("Total Sales by Region")
plt.ylabel("Sales ($)")
# plt.show()
plt.close()

"""
Sales Distribution by Segment: This shows how sales are distributed among different segments.
"""
df.groupby("Segment")["Sales"].sum().plot(kind="pie", autopct="%1.1f%%") # to plot the sales by segment
plt.title("Sales Distribution by Segment") # to add the title
# plt.show()
plt.close()

"""
Top-Performing Products: This identifies the best-selling products.
"""
df.groupby("Product Name")["Sales"].sum().nlargest(10).plot(kind="barh")
plt.title("Top 10 Products by Sales")
plt.xlabel("Total Sales ($)")
# plt.show()
plt.close()

"""
Sales by Category and Sub-Category: This shows how sales are distributed among different categories and sub-categories.
"""
df.groupby(["Category", "Sub-Category"])["Sales"].sum().unstack().plot(kind="bar", stacked=True)
plt.title("Sales by Category and Sub-Category")
plt.ylabel("Sales ($)")
plt.legend(title="Sub-Category")
# plt.show()
plt.close()

"""
Sales Trends Over Time: This shows how sales have changed over the years.
"""
df["Order Date"] = pd.to_datetime(df["Order Date"])  # Ensure datetime format
monthly_sales = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trends")
plt.xlabel("Month")
plt.ylabel("Sales ($)")
plt.grid()
# plt.show()
plt.close()

"""
Yearly Sales Growth: This shows how sales have grown over the years.
"""
yearly_sales = df.groupby("Year")["Sales"].sum()
yearly_sales.plot(kind="line", marker="o")
plt.title("Yearly Sales Growth")
plt.ylabel("Sales ($)")
# plt.show()
plt.close()

"""
Top 5 Customers by Sales: This identifies the top 5 customers by sales.
"""
df.groupby("Customer Name")["Sales"].sum().nlargest(5).plot(kind="barh")
plt.title("Top 5 Customers by Sales")
plt.xlabel("Total Sales ($)")
# plt.show()
plt.close()
