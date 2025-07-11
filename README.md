# 🧠 Customer Churn Prediction using ANN

A robust Artificial Neural Network (ANN)-based solution designed to predict whether a customer is likely to leave a bank. The model is trained on a real-world customer dataset and deployed as an interactive web application using **Streamlit** for real-time predictions.

---

## 🚀 Features

- **Machine Learning Model**: ANN with two hidden dense layers and ReLU activation functions.
- **Performance**:
  - Achieved **Validation Accuracy** of **86.35%**
  - Achieved **Validation Loss** of **0.3425** by **Epoch 3**
- **Data Preprocessing**:
  - `StandardScaler` for feature normalization
  - `LabelEncoder` and `OneHotEncoder` for encoding categorical variables
- **Model Training**:
  - Utilized **EarlyStopping** to prevent overfitting
  - Integrated **TensorBoard** for training visualization
- **Deployment**:
  - Built a responsive web interface using **Streamlit**
  - Allows real-time predictions based on user inputs

---

## 🧱 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Pandas, NumPy, scikit-learn**
- **Streamlit** (for frontend deployment)
- **Matplotlib / Seaborn** (for visualizations)
- **TensorBoard** (for monitoring)

---

## 🧪 Model Architecture

```python
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=number_of_features))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
Loss Function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy

## Sample Results
| Epoch | Validation Accuracy | Validation Loss |
| ----- | ------------------- | --------------- |
| 1     | 83.42%              | 0.5128          |
| 2     | 85.27%              | 0.4101          |
| 3     | **86.35%**          | **0.3425**      |


How to Use
🔧 Local Setup
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/customer-churn-ann.git
cd customer-churn-ann
Install dependencies:

bash
Copy
Edit
pip install -r app/requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app/app.py
