#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostClassifier

file_name = "cat_reg.pkl"
model1 = pickle.load(open(file_name, "rb"))

# model1 = CatBoostClassifier()  
# model1.load_model('model_save')
d=pd.read_csv("uscities.csv")[["city","lat","lng"]]

with open('unique_value.pickle', 'rb') as handle:
    uni = pickle.load(handle)


with open('manufacture_model.pickle', 'rb') as handle:
    man_model = pickle.load(handle)

def welcome():
    return 'welcome all'


# In[6]:


def prediction(year, manufacturer, model, condition,cylinders,fuel,odometer,title_status,
               transmission,drive,size,typee,paint_color,lat,long):  
    
    year=float(year)
    condition=int(condition)
    cylinders=int(cylinders)
    odometer=float(odometer)
    lat=float(lat)
    long=float(long)
    
    test_case=pd.Series({'year': year, 'manufacturer': manufacturer, 'model': model, 'condition': condition, 
   'cylinders': cylinders, 'fuel': fuel, 'odometer': odometer, 'title_status': title_status, 
   'transmission': transmission, 'drive': drive, 'size': size, 'type': typee, 
   'paint_color': paint_color, 'lat': lat, 'long': long})
    
    import pickle
    with open('check.pickle', 'wb') as handle:
        pickle.dump(test_case, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("#####")
    print(test_case)
    
    y_hat_log = model1.predict(test_case)
    prediction = np.exp(y_hat_log)
    prediction= "$" + str(int(prediction))
    print(prediction)
    return prediction
      


# In[9]:


def main():
      # giving the webpage a title
    
    #st.title("Used car price prediction")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    st.set_page_config(page_title="Resale value calc", page_icon="🚗")


    html_temp = """
    <div style =blue;padding:8px">
    <h1 style ="color:blue;text-align:center;">Car Resale Valuation Prediction App</h1>
    </div>
    <div style =padding:0px">
    <br style ="color:blue;text-align:center;">*Please fill the details below</br>
    <br>
    </div>
    """
    
    with open('pic_hash.txt', 'r') as file:
        img_url = file.read().replace('\n', '')
        
    print(img_url)
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("{img_url}");
             
             background-size: cover;
            
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    st.markdown(
                    """
                <style>
                span[data-baseweb="tag"] {
                  background-color: blue !important;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    col1, col2= st.columns([1,1])

    with col1:
        year = st.text_input("YEAR", value=2022, key=1, help="Year in which you want to find the car's resale value")
    with col2:
        manufacturer = st.selectbox("MANUFACTURE", uni["manufacturer"],key="2",help="Brand")
    with col1:
        model = st.selectbox("MODEL", uni["model"],key="3")
    with col2:
        condition = st.selectbox("CONDITION", uni["condition"],key="4",help="rating based on the condition of the car")
    with col1:
        cylinders = st.selectbox("CYLINDERS", uni["cylinders"],key="5",help="number of cylinders in the engine")
    with col2:
        fuel = st.selectbox("FUEL", uni["fuel"],key="6",help="type of fuel used")
    with col1:
        odometer = st.text_input("ODOMETER RATING", 20220,key="7",help=" Distance travelled by the car in miles")
    with col2:
        title_status = st.selectbox("CAR STATUS", uni["title_status"],key="8",help="status of car")
    with col1:
        transmission = st.selectbox("TRANSMISSION", uni["transmission"],key="9")
    with col2:
        drive = st.selectbox("WHEEL DRIVE", uni["drive"],key="10",help="4wd: 4 wheel drive , fwd: forward wheel drive , rwd: rear wheel drive")
    with col1:
        size = st.selectbox("SIZE", uni["size"],key="11")
    with col2:
        typee = st.selectbox("TYPE", uni["type"],key="12")
    with col1:
        paint_color = st.selectbox("PAINT COLOR", uni["paint_color"],key="13")
    with col2:
        city = st.selectbox("City", uni["city"],key="15")
    
    add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)
    import time
    with st.sidebar:
            st.write("FAQs\n")
            st.write("What is car resale valuation?\n")
            st.write("- The price of any car depreciates over time and is dependent upon numerous attributes that determine the resale value of the car in  various cities around the USA. These attributes include year of resale,brand, model, miles driven ,overall condition of the car etc. Car Resale Valuation Prediction App incorporates all these factors into the pricing model that takes into account the previous car resale data to determine the most accurate price range for your car.\n")
            st.write("Do I have to pay or register for using Car resale Valuation APP?\n")
            st.write("- No, you do not have to register or pay to use the Car resale Valuation tool.\n")
    
    result =""
    lat=np.array(d[d["city"]==city]["lat"]).mean()
    long=np.array(d[d["city"]==city]["lng"]).mean()
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        if(model in man_model[manufacturer]):
            result = prediction(year, manufacturer, model, condition,cylinders,fuel,odometer,title_status,
                     transmission,drive,size,typee,paint_color,lat,long)
            st.success('Car is expected to be sold at {}'.format(result))
        else:
            original_manufacture = [k for k,v in man_model.items() if  model in v]
            result = "Car Manufacturer and car model mismatch. " + "Model "+ str(model) + " is a product of manufacturer " +str(original_manufacture[0])
            st.success(format(result))

# In[10]:


if __name__=='__main__':
    main()


# In[11]:



# In[ ]:




