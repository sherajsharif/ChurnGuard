# ChurnGuard: Customer Churn Prediction System

## Project Overview
ChurnGuard is a sophisticated machine learning application that helps businesses predict and analyze customer churn patterns. Built with Python and Streamlit, this tool provides an interactive web interface for uploading customer data and getting instant predictions and insights to improve customer retention strategies.

## Author
**Sheraj Sharif**
- LinkedIn: [Sheraj Sharif](https://www.linkedin.com/in/sheraj-sharif-652723250/)

## Features

- 🔄 Interactive data upload and analysis
- 📊 Real-time visualization of churn patterns
- 🤖 Advanced machine learning model training
- 📈 Feature importance analysis
- 💾 Downloadable trained model
- 📱 Mobile-responsive design

## Technologies Used

- Python 3.9+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib

## Quick Start

1. Clone this repository:
```bash
git clone <repository-url>
cd ChurnGuard
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Data Requirements

Your CSV file should include these columns:
- `customerID`: Unique customer identifier
- `gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Whether customer is a senior citizen (0/1)
- `Partner`: Whether customer has a partner (Yes/No)
- `Dependents`: Whether customer has dependents (Yes/No)
- `tenure`: Number of months with the company
- `PhoneService`: Whether customer has phone service (Yes/No)
- `MultipleLines`: Whether customer has multiple lines (Yes/No/No phone service)
- `InternetService`: Customer's internet service provider (DSL/Fiber optic/No)
- `OnlineSecurity`: Whether customer has online security (Yes/No)
- `OnlineBackup`: Whether customer has online backup (Yes/No)
- `DeviceProtection`: Whether customer has device protection (Yes/No)
- `TechSupport`: Whether customer has tech support (Yes/No)
- `StreamingTV`: Whether customer has streaming TV (Yes/No)
- `StreamingMovies`: Whether customer has streaming movies (Yes/No)
- `Contract`: Contract term (Month-to-month/One year/Two year)
- `PaperlessBilling`: Whether customer has paperless billing (Yes/No)
- `PaymentMethod`: Payment method
- `MonthlyCharges`: Monthly charges
- `TotalCharges`: Total charges
- `Churn`: Whether customer churned (Yes/No)

## Model Details

The application uses a Random Forest Classifier to predict customer churn. The model:
- Automatically handles missing values
- Performs feature scaling
- Encodes categorical variables
- Provides feature importance analysis
- Shows model performance metrics

## Deployment

This application can be deployed for free on Streamlit Cloud:
1. Fork this repository
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy the app with a single click

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset inspired by IBM's Telco Customer Churn dataset
- Built with Streamlit's amazing framework
- Special thanks to the open-source community

---
Made with ❤️ by Sheraj Sharif 