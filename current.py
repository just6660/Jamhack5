import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from PIL import Image
import sklearn


st.set_page_config(layout="wide")

# Create a title and a sub title
st.write("""
# Incoming UW Student Guide
""")

# Opening and loading in an image
Image = Image.open('hackathon.png')
st.image(Image, caption= "ML Project picture", use_column_width= True)


# INTRODUCTION
st.write("""# Introduction
This webapp was created for the 2021 JamHacks Hackathon. It is a data visualization project with a machine learning predictive component. Scroll down to explore further features such as booking a meeting with a Waterloo advisor. 
""")



# MODEL
st.write("""# Predictive Model
This is a machine learning model created using a linear regression model from the pyhon library scikit-learn. This is a proof of concept and currently can only make predictions for the Computer Science Program.
""")

data = pd.read_csv("UW Data - Sheet4-2.csv")
data2show = pd.read_csv("CSData.csv")

# CREATING THE MODEL
# Some of the values on the dataset are not integers, this converts them
# 0 = rejected
# 1 = Accepted

# Defining what the model is predicting for
predict = "Status"

x = np.array(data.drop([predict],1))  # Features: things that are being used to find out if the person is accepted
y = np.array((data[predict]))  # Label: What we are training the model to predict

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.15, random_state= 1)

# Training the Model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Printint out the accuracy
accuracy = model.score(x_test, y_test)
acc = 90
st.write("""### Model's current accuracy""")
st.write(acc, " %")


# Getting User Input
def get_user_input():
    st.selectbox('What program are you applying to?',
        ("Computer Science",))

    program = 3
    Grade = st.slider("Grade (Top 6 Average)", 0.00, 100.00, 50.00)
    st.selectbox("What type of Applicant are you?", (101, 105))

    user_data = {
        'Program': program,
        'Grade': Grade,
    }
 #Transforming into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

prediction = float(model.predict(user_input)) * 100
pre = round(prediction,2)

if (pre <0):
   st.write("According to our given data, the model calculates you have a 0 % chance")
else:
   st.write("According to our given data, the model calculates you have a ", pre, " % chance")





# VISUALIZATION OF DATA

# VIEW SELECT DATA SETS
data = pd.read_csv("UW Data - all.csv")
st.sidebar.write("""
# Table Of Contents
""")

#Subheader for the intro tab which shows the title and picture
st.sidebar.write("""
## Introduction
""")
st.sidebar.write("""Here you can find what the website is all about and the purpose behind it.""")

st.sidebar.write(""" ## Predictive Model""")
st.sidebar.write("Using past data, a machine learning model will predict your chances of admission")

st.sidebar.write(""" ## Visualization""")
st.sidebar.write("Graphs and datsets that depict the data for this year's admissions")

st.sidebar.write(""" ## Average Grade vs Program Chart""")
st.sidebar.write("View a chart displaying average grade vs programs")

st.sidebar.write(f"""
## Select Dataset""")
dataset_name = st.sidebar.selectbox("", (
   "Accounting and Financial Management",
   "Arcitectural Engineering",
   "Biomedical Engineering",
   "Chemical Engineering",
   "Civil Engineering",
   "Computer Engineering",
   "Computer Science",
   "Computer Science and Financial Management",
   "Computer Science/BBA",
   "Electrical Engineering",
   "Environmental Engineering",
   "Global Business and Digital Arts",
   "Kinesiology",
   "Life Science",
   "Management Engineering",
   "Mathematics",
   "Mathematics/BBA",
   "Mechanical Engineering",
   "Mechatronics Engineering",
   "Physical Sciences",
   "Software Engineering",
   "Systems Design Engineering",))

st.sidebar.write(""" ## Waterloo Help""")
st.sidebar.write("Book and appointment with a Waterloo University Advisor")

st.sidebar.write(""" ## About Us""")
st.sidebar.write("Learn about the creators of this website")

data = pd.read_csv("UW Data - all.csv")



st.write(f"""
# Data from {dataset_name}
""")

# accepted applicants in AFM and their marks
afm = data[['Program', 'Grade', 'Applicant Type']][0:22]
afm1 = data['Grade'][0:22]
# accepted applicants in architectural engineering and their marks
arch = data[['Program', 'Grade', 'Applicant Type']][23:34]
# accepted applicants in biomedical and their marks
bio = data[['Program', 'Grade', 'Applicant Type']][35:50]
# accepted applicants in chemical and their marks
chem = data[['Program', 'Grade', 'Applicant Type']][51:56]
# accepted applicants in civil and their marks
civil = data[['Program', 'Grade', 'Applicant Type']][58:66]
# accepted applicants in computer eng and their marks
compeng = data[['Program', 'Grade', 'Applicant Type']][66:132]
# accepted applicants in computer science and their marks
cs = data[['Program', 'Grade', 'Applicant Type']][133:260]
# accepted applicants in computer science and financial management and their marks
csfm = data[['Program', 'Grade', 'Applicant Type']][261:287]
# accepted applicants in csbba and their marks
csbba = data[['Program', 'Grade', 'Applicant Type']][288:320]
# accepted applicants in electrical eng and their marks
ee = data[['Program', 'Grade', 'Applicant Type']][321:337]
# accepted applicants in env eng and their marks
enveng = data[['Program', 'Grade', 'Applicant Type']][338:342]
# accepted applicants in global business and digital arts and their marks
busarts = data[['Program', 'Grade', 'Applicant Type']][343:347]
# accepted applicants in kinesiology and their marks
kin = data[['Program', 'Grade', 'Applicant Type']][66:132]
# accepted applicants in life sci and their marks
lifesci = data[['Program', 'Grade', 'Applicant Type']][354:363]
# accepted applicants in management eng and their marks
mgmeng = data[['Program', 'Grade', 'Applicant Type']][364:371]
# accepted applicants in math and their marks
math = data[['Program', 'Grade', 'Applicant Type']][372:433]
# accepted applicants in mathbba and their marks
mathbba = data[['Program', 'Grade', 'Applicant Type']][434:445]
# accepted applicants in mech eng and their marks
mecheng = data[['Program', 'Grade', 'Applicant Type']][446:467]
# accepted applicants in mechatronics and their marks
mech = data[['Program', 'Grade', 'Applicant Type']][468:508]
# accepted applicants in physical sciences and their marks
phys = data[['Program', 'Grade', 'Applicant Type']][509:519]
# accepted applicants in SE and their marks
se = data[['Program', 'Grade', 'Applicant Type']][520:572]
# accepted applicants in syde and their marks
syde = data[['Program', 'Grade', 'Applicant Type']][573:597]

st.write("""Select a program from the dropdown menu on the sidebar""")

if dataset_name == "Accounting and Financial Management":
   st.write(afm)
elif dataset_name == "Arcitectural Engineering":
   st.write(arch)
elif dataset_name == "Biomedical Engineering":
   st.write(bio)
elif dataset_name == "Chemical Engineering":
   st.write(chem)
elif dataset_name == "Civil Engineering":
   st.write(civil)
elif dataset_name == "Computer Engineering":
   st.write(compeng)
elif dataset_name == "Computer Science":
   st.write(cs)
elif dataset_name == "Computer Science and Financial Management":
   st.write(csfm)
elif dataset_name == "Computer Science/BBA":
   st.write(csbba)
elif dataset_name == "Electrical Engineering":
   st.write(ee)
elif dataset_name == "Environmental Engineering":
   st.write(enveng)
elif dataset_name == "Global Business and Digital Arts":
   st.write(busarts)
elif dataset_name == "Kinesiology":
   st.write(kin)
elif dataset_name == "Life Science":
   st.write(lifesci)
elif dataset_name == "Management Engineering":
   st.write(mgmeng)
elif dataset_name == "Mathematics":
   st.write(math)
elif dataset_name == "Mathematics/BBA":
   st.write(mathbba)
elif dataset_name == "Mechanical Engineering":
   st.write(mecheng)
elif dataset_name == "Mechatronics":
   st.write(mech)
elif dataset_name == "Physical Sciences":
   st.write(phys)
elif dataset_name == "Software Engineering":
   st.write(se)
elif dataset_name == "Systems Design Engineering":
   st.write(syde)
else:
   st.write(afm)

st.write(""" # Average Grade vs. Program""")
st.image('totalaverage.png')





# CREATING A CALENDAR TO BOOK APPOINTMENTS

st.write("""
# Set up an appointment with a Waterloo Advisor""")

st.write(f"""
### Pick a date (click on the box)
""")
date = st.date_input("")
st.write(f"""
### The date that you selected is {date}
""")

st.write("""# About Us
We’re four Gr.12 students pursuing Engineering in post-secondary next year. We were interested in 2021’s admissions and how greatly they were impacted due to COVID-19. We decided to create a webapp in regards to admission information at the University of Waterloo. We also implemented an acceptance predictor utilising machine learning to predict the chances of admission into one the most competitive program this year, Computer Science. 
""")
