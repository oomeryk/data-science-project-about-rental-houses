### Rental house market project of Türkiye with using selenium, pandas, numpy, matplotlib in python

## 1- Web Scraping
############################################################################################

# We should install and set up 'chrome driver' on computer.

# We should do 'pip install Selenium and Pandas'
import pandas as pd
import numpy as np
import selenium
import matplotlib.pyplot as plt

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



#############################################################################################
###  LIST OF TURKIYE DISTRICTS

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

print(ilce_list[:10])
print(ilce_list[len(ilce_list)-1])



##############################################################################################
###  RENTAL HOUSING ADVERTISEMENTS IN EVERY DISTRICT IN TURKIYE

# Open a Chrome window - create a driver instance.
driver = webdriver.Chrome()

# Create a WebDriverWait instance to wait for elements on the web page.
wait = WebDriverWait(driver, 3)

# Open the website.
driver.get("https://www.emlakjet.com/kiralik-konut/")

# Close the pop-up bar that appears at the bottom of the first page.
close_button = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[3]/div[5]/div[2]/div/button/span[1]")
close_button.click()

# Create a list to store the information to be extracted from the website.
list_before_excel = []

# Turn through the district list.
for j in range(len(ilce_list)):
    
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
pddf = pd.DataFrame(list_before_excel)
pddf.to_excel("emlakjet.xlsx")



#########################################################################################
###  2- DATA ANALYSIS

# Cell 1 
d = pd.read_excel("emlak_jet.xlsx")

# Cell 2
df = d.copy()
df.head()

# Cell 3
len(df)

# Cell 4
df = df.drop(0, axis=0)
df = df.drop("Source.Name", axis=1)
df.head(4)

# Cell 5
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')
df.head(4)

# Cell 6
# Swipe cell that write '...' to one cell left
for i in range(len(df)):
    if df.iloc[i,2] == "...":
        j=2
        while j<16:
            df.iloc[i,j] = df.iloc[i,j+1]
            j+=1
df.head(4)

# Cell 7
# drop rows that not "Residence" or "Daire"
for i in range(len(df)):
    if df["Column4"][i] != "Residence" and df["Column4"][i] != "Daire":
        df = df.drop(i, axis=0)      
df.head(4)  

# Cell 8
len(df)

# Cell 9
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')

# Cell 10
# Create a list of the columns_to_keep with column(6,10,13,14)
columns_to_keep = [f"Column{i}" for i in [6, 10, 13, 14]]
for column in df.columns:
    if column not in columns_to_keep:
        df = df.drop(column, axis=1)   
df.head()

# Cell 11
# rename column name
df.rename(columns = {'Column6':  'room',   'Column10': 'm²',
                     'Column13': 'price',  'Column14': 'address'}, inplace = True)    
df.head(1)

# Cell 12
# delete extention of "m²", "price", "address"
df["m²"     ] = df["m²"     ].str.slice(0, -3)
df["price"  ] = df["price"  ].str.slice(0, -3)
df["address"] = df["address"].str.slice(0, -4)
df.head()

# Cell 13
# split "address" to tree column that "city", "district", "neighborhood"
df = df.assign(
    city         = df["address"].str.split(" - ").str[0],
    district     = df["address"].str.split(" - ").str[1],
    neighborhood = df["address"].str.split(" - ").str[2]
)
df.head()

# Cell 14
df = df.drop("address", axis=1)

# Cell 15
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')

# Cell 16
# to make "price" and "m²" integer.
for i in range(len(df)):
    df["price"][i] = df["price"][i].replace("." , "") 
# 2.way
df["price"] = df["price"].astype(float)
df["price"] = df["price"].astype(int)
df["m²"   ] = df["m²"   ].astype(int)
df.head()

# Cell 17
print(df["m²"].dtype)
print(df["price"].dtype)

# Cell 18
# if "price">75000 or "price"<2000 drop this row
for i in range(len(df)):
    if df["price"][i]>75000 or df["price"][i]<2000:
        df = df.drop(i, axis=0)

# Cell 19
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')

# Cell 20
df.head()

# Cell 21
df.describe().T

# Cell 22
# if "m²">300 or "m²"<20 drop this row
for i in range(len(df)):
    if df["m²"][i]>300 or df["m²"][i]<20:
        df = df.drop(i, axis=0)

# Cell 23
df.describe().T

# Cell 24
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')

# Cell 25
# sort column names
df = df.reindex(["city", "district", "neighborhood", "room", "m²", "price"], axis=1)
df.head()

# Cell 26
# replace "Stüdyo" to "1+0"
for i in range(len(df)):
    if df["room"][i]=="Stüdyo":
        df["room"][i]="1+0"

# Cell 27
# drop room numbers that not important 
for i in range(len(df)):
    if df["room"][i] not in ["1+0", "1+1", "2+1", "3+1", "4+1"]:
        df = df.drop(i, axis=0)

# Cell 28
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')

# Cell 29
df

# Cell 30
# drop rows that include null values  and show how many rows are droped
a = len(df)
df = df.dropna()
a-len(df)

# Cell 31
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')

# Cell 32
df.tail()

# Cell 33
# Show adverts that room number not equal to 1+0 and m² less than 30.
rows = df.loc[(df['room'] != "1+0") & (df['m²'] < 30)]
rows

# Cell 34
# Drop this rows.
df = df.drop([16751, 17365], axis=0)

# Cell 35
df.iloc[16748:16751]

# Cell 36
df.iloc[17362:17365]

# Cell 37
# Add list of new_index to the DataFrame as a new column and set new_index to real index numbers
df[''] = np.arange(len(df))
df = df.set_index('')

# Cell 38
df

# Cell 39
# save dataframe to excel
df.to_excel("adverts.xlsx")


############################################################################
## 3- Visulation

# Cell 40
data = pd.read_excel("adverts.xlsx")

# Cell 41
df = data.copy()
df.tail()

# Cell 42
df.drop("Unnamed: 0", axis=1, inplace=True)
df.tail()

#### How many house types are there according to the number of rooms? Showing in pie chart

# Cell 43
print(df["room"].unique())
len(df["room"].unique())

# the code below can also write using 'groupby' method!

# Cell 44
unique_room_list = df["room"].unique()

# Cell 45
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

# Cell 46
# determination x and y labels of pie chart 
x = np.array(["4+1", "3+1", "2+1", "1+1", "1+0"])
y = np.array([len(fourone), len(treeone), len(twoone), len(oneone), len(onezero)])

# Cell 47
# create pie chart
plt.pie(y, labels=x)
plt.title("count of room numbers")
plt.show()

# Cell 48
#### Average of rental house prices in bar plot
prices_mean_of_each_cities = df.groupby("city")["price"].mean()

# Cell 49
prices_mean_of_each_cities = pd.DataFrame(prices_mean_of_each_cities)
prices_mean_of_each_cities.head()

# Cell 50
x = prices_mean_of_each_cities.index
y = prices_mean_of_each_cities["price"]

# Cell 51
plt.figure(figsize=(6, 16))
plt.grid(linewidth=0.4,  axis="x")

plt.barh(x, y,  height=0.5,  color=["red","blue"])
plt.show()

# Cell 52
# compare m² and home type
y = df.groupby("room")["m²"].mean()
x = np.array(["1+0", "1+1", "2+1", "3+1", "4+1"])

plt.grid()
plt.title("compare m² and home type")
plt.scatter(x,y, s=50)
plt.show()

# Cell 53
# count of rental house in logarithmic scale
df["constant"] = 1

y = df.groupby("city")["constant"].sum()
x = pd.DataFrame(y).index

plt.figure(figsize=(10, 15))
plt.grid()
plt.xscale("log")

plt.scatter(y, x)
plt.title("count of rental house in logarithmic scale")
plt.show()







