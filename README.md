# Health Insurance Cross Sell Prediction 

## App Link Heroku: https://cross-sell-prediction-heroku.herokuapp.com/
## App Link AWS: http://cross.dotslashai.com/
## Objectives

<br>- Our client is an Insurance company that has provided Health Insurance to its customers now they need your help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company. 

<br>- Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimize its business model and revenue. 

<br>- In order to predict, whether the customer would be interested in Vehicle insurance, we have information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc.

<br>- We  need to build a model to identify right set of customers who would buy the vehicle insurance.

## Data Set
https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction

## Final Deliverables

<br>- A webpage where an analyst enters customer details
<br>- Result: Final suggestion to run campaign against this customer or not.

## Docker deployment commands:

docker build -t cross-sell .
docker run -d --name cs -p 80:80 cross-sell


## Technologies used

<br>- Python3
<br>- Pandas
<br>- Numpy
<br>- Scikit-learn
<br>- Streamlit
<br>- Heroku
<br>- AWS EC2 and docker
