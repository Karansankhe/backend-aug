from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('loan_approval_rf_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.json
    features = np.array([data['Gender'], data['Married'], data['Dependents'], data['Education'],
                         data['Self_Employed'], data['ApplicantIncome'], data['CoapplicantIncome'],
                         data['LoanAmount'], data['Loan_Amount_Term'], data['Credit_History'],
                         data['Property_Area']])
    
    # Reshape the data and make prediction
    prediction = model.predict([features])
    
    # Return the prediction as JSON
    output = {'Loan_Status': 'Approved' if prediction[0] == 1 else 'Not Approved'}
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
