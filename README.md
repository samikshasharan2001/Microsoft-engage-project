# Microsoft Engage - Car Resale valuation prediction App
<a href="https://share.streamlit.io/samikshasharan2001/microsoft-engage-project/main"> Link To App : <a href="https://www.youtube.com/watch?v=lVmN0fCsq7Q&ab_channel=SamikshaSharan"> Link to Demo

---
 # About The Project
 ---
  The price of any car depreciates over time and is dependent upon  numerous attributes that determine the resale value of the car in  various cities around the USA. These attributes include year of resale,brand, model, miles driven ,overall condition of the car etc. <br>
  Car Resale Valuation Prediction App incorporates all these factors into the pricing model that takes into account the  previous car resale data to determine the most accurate price for your car.
 <details>
<summary> More About the Project </summary>
Car Resale Valuation Prediction App is Macchine Learning Model that uses the concepts of pattern recognition, as well as other forms of predictive algorithms, to make judgments on incoming data. 
</details>
  

 # Table Of Content
<details>
<summary>Table Of Contents</summary>
<ul><li><a href="https://github.com/samikshasharan2001/Microsoft-engage-project/edit/main/README.md#about-the-project">About The Project</a></li>
<li><a href="https://github.com/samikshasharan2001/Microsoft-engage-project/edit/main/README.md#tech-stack">Tech Stack</a></li>
 <li><a href="https://github.com/samikshasharan2001/Microsoft-engage-project/edit/main/README.md#getting-started">Getting Started</a></li>
 <li><a href="https://github.com/samikshasharan2001/Microsoft-engage-project/edit/main/README.md#run-locally">Run Locally</a></li></ul>
</details>
  
  
  ## Tech Stack

**Frontend:** `streamlit framework`
 
**Backend:** `python` 
 
 ## Getting Started
### Installation

CatBoost (python installation )


```bash
 pip install catboost
```
 ```bash
  pip install streamlit
```
 Other Dependencies are: `numpy` `pandas` `seaborn` `os` `pickle` `SciPy` `scikit-learn`
 
#### Use a python IDE to implement predict-car-price-by-catboost-2.py file.
 
 
## Run Locally
#### To separately run the  application on your local host, perform the following steps:

Clone the project

```bash
  git clone https://github.com/samikshasharan2001/Microsoft-engage-project.git
```

Go to the project directory

```bash
  cd microsoft-engage-project
```
Run the following commands to start the server side.
```bash
 streamlit run streamlit_app.py
```
 ## Deployment

the rest API is deployed on streamlit global server and the url is a public address:

Link to the app:- https://share.streamlit.io/samikshasharan2001/microsoft-engage-project/main
 
---
 # More About the project
 ## Roadmap

-  Data acquisition - Required data is downloaded from open-sourced data on the Kaggle - https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data?select=vehicles.csv kaggle the dataset contains most all relevant information that Craigslist provides on car sales including columns like price, condition, manufacturer and 19 other categories. 


- Data cleaning and feature engineering- the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning.
- Model Training - With the help of catboost library, I have used a boosted decision tree algorithm to model resale value w.r.t. vehicle's independent features.<br>
 `catboost library` - is a machine learning library used to handle categorical (CAT) data automatically.
 This library contains the `CatBoostRegressor` and `CatBoostClassifier` models with a similar interface as scikit-learn models.<br>
 ### Model training
 <img width="936" align="centre" alt="Screenshot 2022-05-29 at 1 12 57 PM" src="https://user-images.githubusercontent.com/105884271/170885881-d3ed93cd-1fab-4c61-8361-822d9ea77267.png">
I took 60-70 random data points from the test dataset as indicated by the y_hat(blue) while y_val(red) are the values predicted by the trained model.
 As one can see the predicted values closely follow the original values, thereby indicating that the model is trained well.
 
 <br>
 
 - Streamlit for deploying light ML models- Streamlit is an open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as scikit-learn, NumPy, pandas, Matplotlib etc.
 
 ## Future Improvement 
 - As soon as a car or model is selected, the dropdown and the value that could be selected for a feature should be updated depending upon the attributes of the car or manufactured selected.
 - Although categorical boosting seems to be the best algorithm provided the dataset. Other tree based or SVMs could be tried out. More intensive hyper parameter search could be done using grid search.

 
 


  


