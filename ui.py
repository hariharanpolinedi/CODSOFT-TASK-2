import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

def load_model(file_path):
    """Load trained model from file."""
    return joblib.load(file_path)

def preprocess_new_data(new_data, preprocessor):
    """Preprocess new data using preprocessor."""
    return preprocessor.transform(new_data)

def predict_churn(model, new_data):
    """Predict churn for new data."""
    return model.predict(new_data)

def predict():
    # Load trained model
    model_path = 'customer_churn_model.pkl'
    model = load_model(model_path)
    
    # Load preprocessor
    preprocessor_path = 'preprocessor.pkl'
    preprocessor = joblib.load(preprocessor_path)
    
    # Get user inputs
    credit_score = float(credit_score_entry.get())
    geography = geography_var.get()
    gender = gender_var.get()
    age = float(age_entry.get())
    tenure = float(tenure_entry.get())
    balance = float(balance_entry.get())
    num_of_products = float(num_of_products_entry.get())
    has_cr_card = float(has_cr_card_var.get())
    is_active_member = float(is_active_member_var.get())
    estimated_salary = float(estimated_salary_entry.get())
    
    # Preprocess new data
    new_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    
    new_data_processed = preprocess_new_data(new_data, preprocessor)
    
    # Predict churn for new data
    churn_prediction = predict_churn(model, new_data_processed)
    churn_status = "Churn" if churn_prediction[0] == 1 else "No Churn"
    messagebox.showinfo("Churn Prediction", f"The predicted churn status is: {churn_status}")

# Create tkinter window
window = tk.Tk()
window.title("Customer Churn Prediction")

# Styling
window.configure(bg='#f0f0f0')
label_font = ('Helvetica', 10)
entry_font = ('Helvetica', 10)
button_font = ('Helvetica', 10)

# Credit Score
credit_score_label = tk.Label(window, text="Credit Score:", font=label_font, bg='#f0f0f0')
credit_score_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
credit_score_entry = tk.Entry(window, font=entry_font)
credit_score_entry.grid(row=0, column=1, padx=10, pady=5, sticky='w')

# Geography
geography_label = tk.Label(window, text="Geography:", font=label_font, bg='#f0f0f0')
geography_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
geography_var = tk.StringVar(window)
geography_var.set("France")
geography_option = tk.OptionMenu(window, geography_var, "France", "Germany", "Spain")
geography_option.grid(row=1, column=1, padx=10, pady=5, sticky='w')

# Gender
gender_label = tk.Label(window, text="Gender:", font=label_font, bg='#f0f0f0')
gender_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
gender_var = tk.StringVar(window)
gender_var.set("Male")
gender_option = tk.OptionMenu(window, gender_var, "Male", "Female")
gender_option.grid(row=2, column=1, padx=10, pady=5, sticky='w')

# Age
age_label = tk.Label(window, text="Age:", font=label_font, bg='#f0f0f0')
age_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
age_entry = tk.Entry(window, font=entry_font)
age_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')

# Tenure
tenure_label = tk.Label(window, text="Tenure:", font=label_font, bg='#f0f0f0')
tenure_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
tenure_entry = tk.Entry(window, font=entry_font)
tenure_entry.grid(row=4, column=1, padx=10, pady=5, sticky='w')

# Balance
balance_label = tk.Label(window, text="Balance:", font=label_font, bg='#f0f0f0')
balance_label.grid(row=5, column=0, padx=10, pady=5, sticky='w')
balance_entry = tk.Entry(window, font=entry_font)
balance_entry.grid(row=5, column=1, padx=10, pady=5, sticky='w')

# Number of Products
num_of_products_label = tk.Label(window, text="Number of Products:", font=label_font, bg='#f0f0f0')
num_of_products_label.grid(row=6, column=0, padx=10, pady=5, sticky='w')
num_of_products_entry = tk.Entry(window, font=entry_font)
num_of_products_entry.grid(row=6, column=1, padx=10, pady=5, sticky='w')

# Has Credit Card
has_cr_card_label = tk.Label(window, text="Has Credit Card:", font=label_font, bg='#f0f0f0')
has_cr_card_label.grid(row=7, column=0, padx=10, pady=5, sticky='w')
has_cr_card_var = tk.StringVar(window)
has_cr_card_var.set("1")
has_cr_card_option = tk.OptionMenu(window, has_cr_card_var, "0", "1")
has_cr_card_option.grid(row=7, column=1, padx=10, pady=5, sticky='w')

# Is Active Member
is_active_member_label = tk.Label(window, text="Is Active Member:", font=label_font, bg='#f0f0f0')
is_active_member_label.grid(row=8, column=0, padx=10, pady=5, sticky='w')
is_active_member_var = tk.StringVar(window)
is_active_member_var.set("1")
is_active_member_option = tk.OptionMenu(window, is_active_member_var, "0", "1")
is_active_member_option.grid(row=8, column=1, padx=10, pady=5, sticky='w')

# Estimated Salary
estimated_salary_label = tk.Label(window, text="Estimated Salary:", font=label_font, bg='#f0f0f0')
estimated_salary_label.grid(row=9, column=0, padx=10, pady=5, sticky='w')
estimated_salary_entry = tk.Entry(window, font=entry_font)
estimated_salary_entry.grid(row=9, column=1, padx=10, pady=5, sticky='w')

# Predict Button
predict_button = tk.Button(window, text="Predict Churn", font=button_font, command=predict, bg='#4CAF50', fg='white')
predict_button.grid(row=10, column=0, columnspan=2, padx=10, pady=10)

window.mainloop()
