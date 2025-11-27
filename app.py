from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model đã train
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Lấy dữ liệu từ form
        new_user_data = {
            'Gender': [int(request.form['Gender'])],
            'Age': [int(request.form['Age'])],
            'Occupation': [int(request.form['Occupation'])],
            'Stay_In_Current_City_Years': [int(request.form['Stay_In_Current_City_Years'])],
            'Marital_Status': [int(request.form['Marital_Status'])],
            'Product_Category_1': [int(request.form['Product_Category_1'])],
            'Product_Category_2': [int(request.form['Product_Category_2'])],
            'Product_Category_3': [int(request.form['Product_Category_3'])],
            'City_Category_A': [int(request.form['City_Category_A'])],
            'City_Category_B': [int(request.form['City_Category_B'])],
            'City_Category_C': [int(request.form['City_Category_C'])]
        }

        # Chuyển thành DataFrame
        df = pd.DataFrame(new_user_data)

        # Dự đoán
        pred = model.predict(df)[0]
        prediction = f"Số tiền chi tiêu dự đoán: {pred:,.2f} VNĐ"

    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
