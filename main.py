import pickle
from flask import Flask,render_template,request

app=Flask(__name__)
model= pickle.load(open('model.pkl','rb'))

#url
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    #prediction=model.predict([[request.form.get("temperature")]])
    prediction=model.predict([[20]])

    output=round(prediction[0],2)
    print(output)
    return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True)
