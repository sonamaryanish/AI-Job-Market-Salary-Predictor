
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("salary_model.pkl", "rb"))

print("loaded model")

@app.route("/")


def home():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])


def predict():

    if request.method=='POST':
        try:
            # Get form inputs

            title = request.form['job_title']
            print(title)

            level= request.form['level']
            print(level)

            emptype = request.form['employment_type']
            print(emptype)
            
            location = request.form['location']
            print(location)    

            size = request.form['company_size']
            print(size)

            ratio = request.form['remote_ratio']
            print(ratio)

            skills = request.form['skills']
            print(skills)

            education = request.form['education']
            print(education)     

            experience = request.form['experience']
            print(experience)

            industry = request.form['industry']
            print(industry)

            encoders = pickle.load(open("encoders.pkl", "rb"))

            title = encoders['job_title'].transform([title])[0]
            level = encoders['experience_level'].transform([level])[0]
            emptype = encoders['employment_type'].transform([emptype])[0]
            location = encoders['company_location'].transform([location])[0]
            size = encoders['company_size'].transform([size])[0]
            skills = encoders['required_skills'].transform([skills])[0]
            education = encoders['education_required'].transform([education])[0]
            industry = encoders['industry'].transform([industry])[0]


            # Prepare data 
            details =[title, level,emptype,location,size, ratio, skills,education,experience,industry]
            print("Details are ",details)

            data_out=np.array(details).reshape(1,-1)
            print(data_out)
            print(data_out.shape)

            scaled = pickle.load(open('scaling.pkl','rb'))
            data_scaled = scaled.transform(data_out)
            print("Scaled ", data_scaled)


            # Predict salary
            prediction = model.predict(data_scaled)
            print("Predicted Salary is : ",float(round(prediction[0],2)))

            return render_template('index.html', prediction_text=f'Estimated Salary in USD: {float(round(prediction[0],2))}')
        

        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)