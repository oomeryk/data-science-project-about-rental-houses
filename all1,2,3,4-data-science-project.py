################################################################################################################################

###   Rental house market project of Türkiye with selenium, sklearn, seaborn, matplotlib, pandas, numpy libraries.


# 1- Web Scraping -- line 18

# 2- Data Preprocessing -- line 198

# 3- Data Visualization -- line 593

# 4- Machine Learning -- line 711


######################################################################################################################################

#################################################################################################################################
# 1- Web Scraping
##########################################################################################################################################


# We should install and set up 'chrome driver' on computer.

# We should do 'pip install Selenium'
import selenium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary modules for web browsing with Selenium.
from selenium import webdriver
from selenium.webdriver.common.by import By

# Import functions for waiting.
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions

# Import exceptions for error handling.
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException



####################################################################
# LIST OF TURKIYE DISTRICTS "web scraping"
#########################################

# Create a Chrome WebDriver instance to open a new Chrome window.
driver = webdriver.Chrome()

# Create a WebDriverWait instance to wait for elements on the web page.
wait = WebDriverWait(driver, 4)

# Open the website.
driver.get("https://www.drdatastats.com/turkiye-il-ve-ilceler-listesi/")

# Create a list to store the information to be extracted from the website.
ilce_list = []

# Wait until the table is visible on the page.
wait.until(expected_conditions.visibility_of_element_located((By.XPATH, "/html/body/div[5]/div[2]/div/div[2]/div/div/article/div[3]/figure[1]/table/tbody")))

# Find all the rows in the table.
rows = driver.find_elements(By.XPATH, "/html/body/div[5]/div[2]/div/div[2]/div/div/article/div[3]/figure[1]/table/tbody/tr")

# Iterate through the rows in the table.
for i in range(len(rows)-1):
    
    # Extract the XPath for the district (ilçe).
    ilce_path = f"/html/body/div[5]/div[2]/div/div[2]/div/div/article/div[3]/figure[1]/table/tbody/tr[{i+2}]/td[3]"
    
    # Find and extract the district (ilçe) text.
    ilce = driver.find_element(By.XPATH, ilce_path)
    ilce = ilce.text
    
    # Check if the district is not "Merkez" and add it to the list directly.
    if ilce != "Merkez":
        ilce_list.append(ilce)
        
    # If it's "Merkez," also extract the province (il) and combine them.    
    else:
        il_path = f"/html/body/div[5]/div[2]/div/div[2]/div/div/article/div[3]/figure[1]/table/tbody/tr[{i+2}]/td[2]"
        il = driver.find_element(By.XPATH, il_path)
        il = il.text
        ilce_list.append((il + " " + ilce))

driver.quit()

#######################################
print(ilce_list[:10])
print(ilce_list[len(ilce_list)-1])





#####################################################################
# RENTAL HOUSING ADVERTISEMENTS IN EVERY DISTRICT "web scraping"
###############################################


# Open a Chrome window - create a driver instance.
driver = webdriver.Chrome()

# Create a WebDriverWait instance to wait for elements on the web page.
wait = WebDriverWait(driver, 5)

# Open the website.
driver.get("https://www.emlakjet.com/kiralik-konut/")

# Close the pop-up bar that appears at the bottom of the first page.
close_button = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[3]/div[5]/div[2]/div/button/span[1]")
close_button.click()

# Create a list to store the information to be extracted from the website.
list_before_excel = []

# Turn through the district list.
for j in range(446, len(ilce_list)):
    
        # Wait for the district input field to be clickable. Click and write district to input.
    wait.until(expected_conditions.element_to_be_clickable((By.XPATH, "/html/body/div/div/div[3]/div[1]/div/div[5]/div/div/div[2]/div[2]/div/input")))
    ilce_input = driver.find_element(By.XPATH, "/html/body/div/div/div[3]/div[1]/div/div[5]/div/div/div[2]/div[2]/div/input")
    ilce_input.click()                          
    ilce_input.send_keys(ilce_list[j])
 
    # Wait for the search button to be clickable and click.
    wait.until(expected_conditions.element_to_be_clickable((By.XPATH, "html/body/div/div/div[3]/div[1]/div/div[5]/div/div/button")))
    search_button = driver.find_element(By.XPATH, "html/body/div/div/div[3]/div[1]/div/div[5]/div/div/button")
    search_button.click()
    
    # Turn all pages for one district.
    for one_page in range(40):
        try:
            try:
                # Find elements with the class name of the advert listings on the first page.
                wait.until(expected_conditions.visibility_of_all_elements_located((By.CLASS_NAME, "manJWF")))
                elements = driver.find_elements(By.CLASS_NAME, "manJWF")
                
                # Turn through advert boxes on a page.
                for element in elements:
                    # Append advert information to list.
                    list_before_excel.append(element.text.split("\n"))

                # Click the "Next" button to go to the next page.      
                wait.until(expected_conditions.visibility_of_all_elements_located((By.CLASS_NAME, "_3au2n_.OTUgAO")))
                next_button = driver.find_element(By.CLASS_NAME, "_3au2n_.OTUgAO")
                next_button.click()                                  
                
            # if StaleElementReferenceException occurs refresh web page and run same codes above       
            except StaleElementReferenceException:
                driver.refresh()
                
                # Find elements with the class name of the property listings on the first page.
                wait.until(expected_conditions.visibility_of_all_elements_located((By.CLASS_NAME, "manJWF")))
                elements = driver.find_elements(By.CLASS_NAME, "manJWF")
                
                # Iterate through property listings on one page.
                for element in elements:
                    # Append property information to the list.
                    list_before_excel.append(element.text.split("\n"))

                # Find the 'next_button' again.   
                wait.until(expected_conditions.visibility_of_all_elements_located((By.CLASS_NAME, "_3au2n_.OTUgAO")))
                next_button = driver.find_element(By.CLASS_NAME, "_3au2n_.OTUgAO")
                next_button.click()
                
        # İf there aren't 'advert' or 'next_button' in page, move to next district.            
        # So if these errors occurs from the try block above; remove the (il, ilçe) and break loop    
        except (NoSuchElementException, TimeoutException, IndexError):
            try:
                # If (il, ilce) cancel buttons are not found(NoSuchElementException), change to second path and try again
                remove_ilce = driver.find_element(By.XPATH, "/html/body/div/div/div[3]/div[1]/div/div[5]/div/div/div[2]/div[2]/div/div[1]/i")
                remove_ilce.click()                     
                remove_il = driver.find_element(By.XPATH, "/html/body/div/div/div[3]/div[1]/div/div[5]/div/div/div[2]/div[2]/div/div/i")
                remove_il.click()                          
            except NoSuchElementException:
                remove_ilce = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[3]/div[1]/div/div[5]/div/div/div[2]/div[2]/div/div[1]/i")
                remove_ilce.click()                          
                remove_il = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[3]/div[1]/div/div[5]/div/div/div[2]/div[2]/div/div/i")
                remove_il.click()                
                
            break

###########################################
pddf = pd.DataFrame(list_before_excel)
pddf.to_excel("emlakjet.xlsx")





#############################################################################################################################################

#########################################################################################################################################
# 2- DATA PREPROCESSİNG
##################################################################################################################################################


import pandas as pd
import numpy as np
import seaborn as sns

###########################################
data = pd.read_excel("emlak_jet.xlsx")
df = data.copy()
df.head()

##########################################
len(df)

##########################################
df = df.drop(0, axis=0)
df = df.drop("Source.Name", axis=1)

df.head(4)


##########################################
# Reindex index number like starting from 0.
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')

df.head(4)


########################################
# Swipe cell that write '...' to one cell left
for i in range(len(df)):
    if df.iloc[i,2] == "...":
        j=2
        while j<16:
            df.iloc[i,j] = df.iloc[i,j+1]
            j+=1
df.head(4)


#########################################
# Drop rows except "Residence" and "Daire"
for i in range(len(df)):
    if df["Column4"][i] != "Residence" and df["Column4"][i] != "Daire":
        df = df.drop(i, axis=0)
        
df.head(4)  
df[''] = np.arange(len(df))
df = df.set_index('')
len(df)


###########################################
# Wiew the error of Column14 and Column16
df.iloc[35:38,:]

# Assign Column16 cell to Column14 cell
for i in range(len(df)):
    if df["Column14"][i]=="arrow_downward":
        df["Column14"][i] = df["Column16"][i]
        
# Error fixed
df.iloc[35:38,:]


#####################################
## Drop colums that not necessary 

# Create a list of the columns_to_keep with column(6,10,13,14)
columns_to_keep = [f"Column{i}" for i in [6, 10, 13, 14]]

for column in df.columns:
    if column not in columns_to_keep:
        df = df.drop(column, axis=1)
    
df.head()


#######################################3
# rename column name
df.rename(columns = {'Column6':  'room',   'Column10': 'm²',
                     'Column13': 'price',  'Column14': 'address'}, inplace = True)    
df.head(1)


#########################################3
# Delete extentions of "m²", "price", "address"
df["m²"     ] = df["m²"     ].str.slice(0, -3)
df["price"  ] = df["price"  ].str.slice(0, -3)
df["address"] = df["address"].str.slice(0, -4)

df.head()


#######################################
# Split "address" to tree column that "city", "district", "neighborhood"
df = df.assign(
    city         = df["address"].str.split(" - ").str[0],
    district     = df["address"].str.split(" - ").str[1],
    neighborhood = df["address"].str.split(" - ").str[2]
)

df.head()
df = df.drop("address", axis=1)


########################################
# how many rows that include null cell are there?

NaNdf = df[df.isnull().any(axis=1)]
print(f"Number of rows that include Null values: {len(NaNdf)}")  # = 0 row


#######################################
# Convert the "price" and "m²" to integer.
for i in range(len(df)):
    df["price"][i] = df["price"][i].replace("." , "") 

df["price"] = pd.to_numeric(df["price"])  # pandas to_numeric function convert string "to int64"
df["price"] = df["price"].astype(int)     # astype int function convert int64 "to int32"

df["m²"] = df["m²"].astype(int)

# 2.way
# but this code don't transform over 1.000.000 to float 
#df["price"] = df["price"].astype(float)
#df["price"] = df["price"].astype(int)

#########################
print(df["m²"].dtype)
print(df["price"].dtype)


############################################
# Drop outlier "m²" values.
# Drop limited to between 300 and 20  for now...
# if "m²">300 or "m²"<20 drop this row

for i in range(len(df)):
    if df["m²"][i]>300 or df["m²"][i]<20:
        df = df.drop(i, axis=0)
        
# Reindex index column.
df[''] = np.arange(len(df))
df = df.set_index('')

##################
# Wiew m² outliars in seaborn boxplot
m = df["m²"]
sns.boxplot(x = m)
df.describe().T

# learn index of m² outliars
outliars = df["m²"] > (145 + (3/2*(145-80)))
df[outliars].index

# drop outlier "m²" values
df = df.drop(df[outliars].index, axis=0)

# Reindex index column. 
df[''] = np.arange(len(df))
df = df.set_index('')


######################################
# wiew outliars m² in seaborn boxplot again.
m = df["m²"]
sns.boxplot(x = m)
df.describe().T

# learn index of m² outliars
outliars = df["m²"] > (140 + (3/2*(140-80)))


#####################
# drop outliers "m²" again.
df = df.drop(df[outliars].index, axis=0)

df[''] = np.arange(len(df))
df = df.set_index('')

# wiew m² outliars in seaborn boxplot 
m = df["m²"]
sns.boxplot(x = m)


###########################################
# Deleting outlier "price" values.
# Drop outlier "price" values between 2000 and 75000 for now...
# if "price">75000 or "price"<2000 drop that row.

for i in range(len(df)):
    if df["price"][i]>75000 or df["price"][i]<2000:
        df = df.drop(i, axis=0)
        
df[''] = np.arange(len(df))
df = df.set_index('')
df.describe().T


#######################################
# sort columns
df = df.reindex(["city", "district", "neighborhood", "room", "m²", "price"], axis=1)
df.head()


######################################
# replace "Stüdyo" to "1+0"
for i in range(len(df)):
    if df["room"][i]=="Stüdyo":
        df["room"][i]="1+0"


#######################################        
# Drop room numbers that not important like "2.5+1", "5+1", "2+0"
for i in range(len(df)):
    if df["room"][i] not in ["1+0", "1+1", "2+1", "3+1", "4+1"]:
        df = df.drop(i, axis=0)       
df[''] = np.arange(len(df))
df = df.set_index('')


#####################################
len(df)
##
df1 = df.copy()
df.to_excel("adverts1-1.xlsx")


#######################################
# Cleaning some outlier 'm²' values
####################################

# wiew and drop outliar "m²" for "1+0" room type
room0 = df[df["room"]=="1+0"]

sns.boxplot(x = room0["m²"])
room0["m²"].describe()
outliars = room0["m²"]> 54 + 3/2*(54-35)
room0[outliars].index
df = df.drop(room0[outliars].index, axis=0)
df[""]= np.arange(len(df))
df = df.set_index("")


########################################
# wiew and drop outliar "m²" for "1+1" room type
room1 = df[df["room"]=="1+1"]

sns.boxplot(x = room1["m²"])
room1["m²"].describe()
outliars = room1[room1["m²"] > 75 + 3/2*(75-55)].index
outliars2 = room1[room1["m²"] < 55 - 3/2*(75-55)].index
o = outliars.append(outliars2)

df = df.drop(o, axis=0)
df[""]= np.arange(len(df))
df = df.set_index("")


######################################
# wiew and drop outliar "m²" for "2+1" room type
room2 = df[df["room"]=="2+1"]

sns.boxplot(x = room2["m²"])
room2["m²"].describe()
outliars = room2[room2["m²"] > 110 + 3/2*(110-90)].index
outliars2 = room2[room2["m²"] < 90 - 3/2*(110-90)].index
o = outliars.append(outliars2)

df = df.drop(o, axis=0)
df[""]= np.arange(len(df))
df = df.set_index("")


#######################################
# wiew and drop outliar "m²" for "3+1" room type
room3 = df[df["room"]=="3+1"]

sns.boxplot(x = room3["m²"])
room3["m²"].describe()
outliars = room3[room3["m²"] > 160+3/2*(160-125)].index
outliars2 = room3[room3["m²"] < 125-3/2*(160-125)].index
o = outliars.append(outliars2)

df = df.drop(o, axis=0)
df[""]= np.arange(len(df))
df = df.set_index("")


#######################################
# wiew and drop outliar "m²" for "4+1" room type
room4 = df[df["room"]=="4+1"]

sns.boxplot(x = room4["m²"])
room4["m²"].describe()
outliars = room4[room4["m²"] < 170-3/2*(205-170)].index

df = df.drop(outliars, axis=0)
df[""]= np.arange(len(df))
df = df.set_index("")
len(df)
df2=df


###########################################
#  Cleaning some outlier "prices"
######################################

# wiew and drop "price" outliars for "1+0" room type
room0 = df[df["room"]=="1+0"]
price0 = room0["price"]

sns.boxplot(x = price0)
price0.describe()
outliars = price0[price0>12000+3/2*(12000-5000)].index

df = df.drop(outliars, axis=0)
df[""] = np.arange(len(df))
df = df.set_index("")


#########################################
# wiew and drop "price" outliars for "1+1" room type
room1 = df[df["room"]=="1+1"]
price1 = room1["price"]

sns.boxplot(x = price1)
price1.describe()
outliars = price1[price1>15000+5/2*(15000-8000)].index

df = df.drop(outliars, axis=0)
df[""] = np.arange(len(df))
df = df.set_index("")


##########################################
# wiew and drop "price" outliars for "2+1" room type
room2 = df[df["room"]=="2+1"]
price2 = room2["price"]

sns.boxplot(x = price2)
price2.describe()
outliars = price2[price2>17500+5/2*(17500-10000)].index

df = df.drop(outliars, axis=0)
df[""] = np.arange(len(df))
df = df.set_index("")


#######################################
# wiew and drop "price" outliars for "3+1" room type
room3 = df[df["room"]=="3+1"]
price3 = room3["price"]

sns.boxplot(x = price3)
price3.describe()
outliars = price3[price3>18500+3*(18500-11000)].index

df = df.drop(outliars, axis=0)
df[""] = np.arange(len(df))
df = df.set_index("")


########################################
# wiew and drop "price" outliars for "4+1" room type
room4 = df[df["room"]=="4+1"]
price4 = room4["price"]

sns.boxplot(x = price4)
price4.describe()
outliars = price4[price4>25000+3*(25000-14000)].index

df = df.drop(outliars, axis=0)
df[""] = np.arange(len(df))
df = df.set_index("")
len(df)


########################################
# save dataframe to excel
df.to_excel("adverts1-2.xlsx")

df3 = df





##########################################################################################################################################

##########################################################################################################################################
#  3- VISUALIZATION
##################################################################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("adverts1-2.xlsx")
df = data.copy()
df.head()
df = df.drop("Unnamed: 0", axis=1)


#####################################
# average price for each home type in Türkiye
y = df.groupby("room")["price"].mean()
x = np.array(["1+0", "1+1", "2+1", "3+1", "4+1"])

plt.grid(axis="y")
plt.title("Average price for each home type in Türkiye")
plt.bar(x,y)
plt.show()


#####################################3
# average m² for each home type
y = df.groupby("room")["m²"].mean()
x = np.array(["1+0", "1+1", "2+1", "3+1", "4+1"])

plt.grid()
plt.title("average m² for each home type")
plt.scatter(x,y, s=50)
plt.show()


#########################################
# count of home type
# the code below can also write using 'groupby' method!

unique_room_list = df["room"].unique()

twoone= [] # mean of "twoone" is "2+1" house
fourone= []
treeone= []
oneone= []
onezero= []

for i in range(len(df)):
    if unique_room_list[0] == df["room"][i]:
        twoone.append(df["room"][i])
    elif unique_room_list[1] == df["room"][i]:
        fourone.append(df["room"][i])
    elif unique_room_list[2] == df["room"][i]:
        treeone.append(df["room"][i])
    elif unique_room_list[3] == df["room"][i]:
        oneone.append(df["room"][i])
    else:
        onezero.append(df["room"][i])

print(f"count of 4+1 house: {len(fourone)}")
print(f"count of 3+1 house: {len(treeone)}")
print(f"count of 2+1 house: {len(twoone)}")
print(f"count of 1+1 house: {len(oneone)}")
print(f"count of 1+0 house: {len(onezero)}")


########################################
# count of home type
# determination x and y labels of pie chart 
x = np.array(["4+1", "3+1", "2+1", "1+1", "1+0"])
y = np.array([len(fourone), len(treeone), len(twoone), len(oneone), len(onezero)])

# create pie chart
plt.pie(y, labels=x)
plt.title("count of home type")
plt.show()


#########################################
# Average of rental house prices for each cities

average_prices = df.groupby("city")["price"].mean()
y = average_prices.values
x = average_prices.index


plt.figure(figsize=(6, 16))
plt.grid(linewidth=0.4,  axis="x")

plt.barh(x, y,  height=0.5,  color=["orange","blue"])

plt.title("average rent price of each cities")

plt.show()


############################################
# count of rental house in logarithmic scale
df["constant"] = 1

y = df.groupby("city")["constant"].sum()
x = pd.DataFrame(y).index

plt.figure(figsize=(10, 15))
plt.grid(axis="x")
plt.xscale("log")

plt.barh(x, y, height=0.4,  color=["red","y"])
plt.title("count of rental house in logarithmic scale")
plt.show()





######################################################################################################################################

######################################################################################################################################
#  4- MACHINE LEARNING
######################################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor  
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

########################################
data = pd.read_excel("adverts1-2.xlsx")
data

#######################################
data = data.drop("Unnamed: 0", axis=1)
df = data


#######################################
# Assing dummy variable to columns for each room type
df = pd.get_dummies(df, columns=["room"], prefix=["room"])

######################################
# learn how many advert for each room type using cplumns that assing dummy 
arr= [df["room_1+0"].sum(), df["room_1+1"].sum(), df["room_2+1"].sum(), df["room_3+1"].sum(), df["room_4+1"].sum()]
arr


######################################
# Apply LabelEencoder transformation for 'city', 'district', 'neighborhood'
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Encode the string column 'city'
df['city_le'] = le.fit_transform(df['city'])

# Encode the string column 'district'
df['district_le'] = le.fit_transform(df['district'])

# Encode the string column 'neighborhood'
df['neighborhood_le'] = le.fit_transform(df['neighborhood'])
df = df.drop(["city","district","neighborhood"], axis=1)
df.describe().T
df.head()


##########################################
# Define dependet and independet variables

y = df["price"]
x = df.drop("price", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# for learning rate"test_data/all_data" that 2+1 room type  
(x_test["room_2+1"].sum()) / (x_test["room_2+1"].sum() + x_train["room_2+1"].sum())
x_train


###############################################################

# Creating machine learning models

###################################
# KNN regression
knn_model = KNeighborsRegressor().fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
mse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
mse_knn

###################################
# Decision Tree regression
dec_tree = DecisionTreeRegressor().fit(x_train, y_train)
y_pred_tree = dec_tree.predict(x_test)
mse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
mse_tree

####################################
# Random Forest regression
rf = RandomForestRegressor().fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

mse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mse_rf

#####################################
# Cat Boost regression
cat_boost = CatBoostRegressor().fit(x_train, y_train)

y_pred_cat = cat_boost.predict(x_test)

mse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))
mse_cat

#####################################
# Bagged Trees regression
bag_model = BaggingRegressor(bootstrap_features = True).fit(x_train, y_train)

y_pred_bag = bag_model.predict(x_test)

mse_bag = np.sqrt(mean_squared_error(y_test, y_pred_bag))
mse_bag


####################################
#!pip install lightgbm
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
#########################
lgbm_model = LGBMRegressor().fit(x_train, y_train)

y_pred_lgbm = lgbm_model.predict(x_test)

mse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
mse_lgbm

####################################
# Gradient Boosting regression
gbm_model = GradientBoostingRegressor().fit(x_train, y_train)

y_pred_gbm = gbm_model.predict(x_test)

mse_gbm = np.sqrt(mean_squared_error(y_test, y_pred_gbm))
mse_gbm

#####################################
#!pip install xgboost
# XGBoost regression
xgb_model = XGBRegressor().fit(x_train, y_train)

y_pred_xgb = xgb_model.predict(x_test)

mse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mse_xgb


######################################
# mse comparison of all models
all_mse = pd.DataFrame({
    "x": ["mse_xgb","mse_bag","mse_cat","mse_gbm","mse_knn","mse_lgbm","mse_rf","mse_tree"],
    "y": [mse_xgb,mse_bag,mse_cat,mse_gbm,mse_knn,mse_lgbm,mse_rf,mse_tree]  
})

plt.grid(axis="x")
plt.barh(all_mse["x"], all_mse["y"])

# XGB showed the best performance in this dataset


#####################################################################
# XGB model tuning

xgb_grid = {
     'colsample_bytree': [0.5, 0.6, 0.7, 0.9],
     'subsample': [0.5, 1],
     'n_estimators':[100, 500, 900, 1000, 1100],
     'max_depth': [3, 5, 6, 7],
     'learning_rate': [0.1, 0.01, 0.2]
}

xgb = XGBRegressor()

xgb_cv = GridSearchCV(xgb, 
                      param_grid = xgb_grid, 
                      cv = 10, 
                      n_jobs = -1,
                      verbose = 2)


xgb_cv.fit(x_train, y_train)

xgb_cv.best_params_

##########################################
xgb_tuned = XGBRegressor(colsample_bytree = 0.6, 
                         learning_rate = 0.1, 
                         max_depth = 7, 
                         n_estimators = 500,
                         subsample = 1)

xgb_tuned = xgb_tuned.fit(x_train,y_train)

y_pred = xgb_tuned.predict(x_test)

np.sqrt(mean_squared_error(y_test, y_pred))




######################################################################################

# Show estimated price and many useful output using XGB model based on entered infos

def pred_price(city, district, neighborhood, room, metrekare):
  
    d = data
    if city!="" and city not in d["city"].tolist():
        print("--------------------------------\n\nBöyle bir il bulunamadı ya da bu ilde ilan yok!\n")
        print("Şehir ismini doğru yazdığınızdan emin olun!\nŞehrin baş harfini büyük yazmayı deneyin!\n")
        return None    
    if district!="" and district not in d["district"].tolist():
        print("--------------------------------\n\nBöyle bir ilçe bulunamadı ya da bu ilçede ilan yok!\n")
        print("İlçe ismini doğru yazdığınızdan emin olun!\nİlçenin baş harfini büyük yazmayı deneyin!\n")
        return None
    if neighborhood!="" and neighborhood not in d["neighborhood"].tolist():
        print("--------------------------------\n\nBöyle bir mahalle bulunamadı ya da bu mahallede ilan yok!\n")
        print("Mahalle ismini doğru yazdığınızdan emin olun!\nMahallenin baş harfini büyük yazmayı deneyin!\n")
        return None

####
    
    if metrekare=="" and room=="1+0": 
        metrekare=40
        #print(f"       m²={metrekare} olarak hesaba katıldı!")
    elif metrekare=="" and room=="1+1": 
        metrekare=65
        #print(f"       m²={metrekare} olarak hesaba katıldı!")
    elif metrekare=="" and room=="2+1": 
        metrekare=100
        #print(f"       m²={metrekare} olarak hesaba katıldı!")
    elif metrekare=="" and room=="3+1": 
        metrekare=130
        #print(f"       m²={metrekare} olarak hesaba katıldı!")
    elif metrekare=="" and room=="4+1": 
        metrekare=160
        #print(f"       m²={metrekare} olarak hesaba katıldı!")
    elif metrekare=="" and room=="": 
        metrekare=110
        #print(f"       m²={metrekare} olarak hesaba katıldı!")
    else: 
        pass

###    

    new = pd.DataFrame({"city":[city], "district":[district], "neighborhood":[neighborhood], "room":[room], "m²":[metrekare]})
    new["m²"] = pd.to_numeric(new["m²"])
       
    dff = d.drop("price", axis=1)
    dff = pd.concat([dff, new], axis=0)

    dff[""] = np.arange(len(dff))
    dff = dff.set_index("")

    # Get dummies the string column"room"
    dff = pd.get_dummies(dff, columns=["room"], prefix=["room"])

    # Encode the string column'city'
    dff['city_le'] = le.fit_transform(dff['city'])
    
    # Encode the string column'district'
    dff['district_le'] = le.fit_transform(dff['district'])
    
    # Encode the string column'neighborhood'
    dff['neighborhood_le'] = le.fit_transform(dff['neighborhood'])

    dff =dff.drop(["city","district","neighborhood"], axis=1)
   

    if room=="":
        dff = dff.drop("room_", axis=1)

###
    # len(data) = 27118 = index of last row
    train_data = dff.iloc[27118:,:]

    pred = xgb_tuned.predict(train_data)

###

    # write advertising infos according to inputs entered by user
    if city=="" and room!="":
        a=d[d["room"]==room]
        print(f"----------------------------\n\nTürkiye'de tahmini {room} kira fiyatı: {int(pred)} TL\n\n----------------------------")     
        
        print(f"\nTürkiye'de toplam {len(a)} tane {room} ilan var.\n\n----------------------------")
    
    elif district=="" and room!="":
        a=d[d["city"]==city]
        a=a[a["room"]==room]
        print(f"----------------------------\n\n{city} şehrinde tahmini {room} kira fiyatı: {int(pred)} TL\n\n----------------------------")     
        
        if len(a)>=5: 
            last_q = a["price"].quantile(0.75) / a["m²"].quantile(0.75)
            first_q = a["price"].quantile(0.25) / a["m²"].quantile(0.25)
            print(f"\nOrtalama metrekare fiyatı {int(first_q)}TL - {int(last_q)}TL aralığında.")
        else: pass

        print(f"\nEn düşük fiyat: {a['price'].min()} \nEn yüksek fiyat: {a['price'].max()}\n")
        print(f"{city} ilinde toplam {len(a)} tane {room} ilan var.\n\n----------------------------")
   
    elif neighborhood=="" and room!="":
        a=d[d["district"]==district]
        a=a[a["room"]==room]
        print(f"----------------------------\n\n{district} ilçesinde tahmini {room} kira fiyatı: {int(pred)} TL\n\n----------------------------")     
        
        if len(a)>=5: 
            last_q = a["price"].quantile(0.75) / a["m²"].quantile(0.75)
            first_q = a["price"].quantile(0.25) / a["m²"].quantile(0.25)
            print(f"\nOrtalama metrekare fiyatı {int(first_q)}TL - {int(last_q)}TL aralığında.")
        else: pass

        print(f"\nEn düşük fiyat: {a['price'].min()} \nEn yüksek fiyat: {a['price'].max()}\n")
        print(f"{district} ilçesinde toplam {len(a)} tane {room} ilan var.\n\n----------------------------")
        a=a.sort_values(by="price", ascending=True)
        print(f"\nGirilen bilgilere göre Emlakjet sitesindeki ilanlar 'Artan fiyata göre sıralı':\n\n{a.to_string(index=False)}")
   
    elif neighborhood != "" and room!="":
        a=d[d["district"]==district]
        a=a[a["neighborhood"]==neighborhood]
        a=a[a["room"]==room]
        print(f"----------------------------\n\n{neighborhood} mahallesinde tahmini {room} kira fiyatı: {int(pred)} TL\n\n----------------------------")     
        
        if len(a)>=5: 
            last_q = a["price"].quantile(0.75) / a["m²"].quantile(0.75)
            first_q = a["price"].quantile(0.25) / a["m²"].quantile(0.25)
            print(f"\nOrtalama metrekare fiyatı {int(first_q)}TL - {int(last_q)}TL aralığında.")
        else: pass

        print(f"\nEn düşük fiyat: {a['price'].min()} \nEn yüksek fiyat: {a['price'].max()}\n")
        print(f"{neighborhood} mahallesinde toplam {len(a)} tane {room} ilan var.\n\n----------------------------")
        a=a.sort_values(by="price", ascending=True)
        print(f"\nGirilen bilgilere göre Emlakjet sitesindeki ilanlar 'Artan fiyata göre sıralı':\n\n{a.to_string(index=False)}")

    
    elif room=="" and city=="":
        print(f"----------------------------\n\nTürkiye'de tahmini kira fiyatı: {int(pred)} TL\n\n----------------------------")     
        print(f"Türkiye'de toplam {len(d)} tane ilan var.\n\n----------------------------")
    
    elif room == "" and district=="":
        a=d[d["city"]==city]
        print(f"----------------------------\n\n{city} şehrinde tahmini kira fiyatı: {int(pred)} TL\n\n----------------------------")     
        
        if len(a)>=5: 
            last_q = a["price"].quantile(0.75) / a["m²"].quantile(0.75)
            first_q = a["price"].quantile(0.25) / a["m²"].quantile(0.25)
            print(f"\nOrtalama metrekare fiyatı {int(first_q)}TL - {int(last_q)}TL aralığında.")
        else: pass

        print(f"\nEn düşük fiyat: {a['price'].min()} \nEn yüksek fiyat: {a['price'].max()}\n")
        print(f"{city} ilinde toplam {len(a)} tane ilan var.\n\n----------------------------")
    
    elif room == "" and neighborhood=="":
        a=d[d["district"]==district]
        print(f"----------------------------\n\n{district} ilçesinde tahmini kira fiyatı: {int(pred)} TL\n\n----------------------------")     
        
        if len(a)>=5: 
            last_q = a["price"].quantile(0.75) / a["m²"].quantile(0.75)
            first_q = a["price"].quantile(0.25) / a["m²"].quantile(0.25)
            print(f"\nOrtalama metrekare fiyatı {int(first_q)}TL - {int(last_q)}TL aralığında.")
        else: pass

        print(f"\nEn düşük fiyat: {a['price'].min()} \nEn yüksek fiyat: {a['price'].max()}\n")
        print(f"{district} ilçesinde toplam {len(a)} tane ilan var.\n\n----------------------------")
        a=a.sort_values(by="price", ascending=True)
        print(f"\nGirilen bilgilere göre Emlakjet sitesindeki ilanlar 'Artan fiyata göre sıralı':\n\n{a.to_string(index=False)}")
    
    elif room == "" and neighborhood!="":
        a=d[d["city"]==city]
        a=a[a["neighborhood"]==neighborhood]
        print(f"----------------------------\n\n{neighborhood} mahallesinde tahmini kira fiyatı: {int(pred)} TL\n\n----------------------------")     
        
        if len(a)>=5: 
            last_q = a["price"].quantile(0.75) / a["m²"].quantile(0.75)
            first_q = a["price"].quantile(0.25) / a["m²"].quantile(0.25)
            print(f"\nOrtalama metrekare fiyatı {int(first_q)}TL - {int(last_q)}TL aralığında.")
        else: pass

        print(f"\nEn düşük fiyat: {a['price'].min()} \nEn yüksek fiyat: {a['price'].max()}\n")
        print(f"{neighborhood} mahallesinde toplam {len(a)} tane ilan var.\n\n----------------------------")
        a=a.sort_values(by="price", ascending=True)
        print(f"\nGirilen bilgilere göre Emlakjet sitesindeki ilanlar 'Artan fiyata göre sıralı':\n\n{a.to_string(index=False)}")
    
    else:
        pass



####################################################

print("İstenilen şekilde filtreleme yapilabilir.\n")

city = ""
city = input("City: ")
district = ""
district = input("District: ")
neighborhood = ""
neighborhood = input("Neighborhood: ")
room = ""
room = input("Rooms: ")
metrekare = ""
metrekare = input("Meter: ")

# Sent inputs to pred_price function
pred_price(city,district,neighborhood,room,metrekare)





##################################################################################################################################################

#########################################################################################################################################
