🌿 AI-Driven Crop Disease Prediction and Management System
👨‍🌾 Empowering Farmers with AI for Early Crop Disease Detection

This project uses Deep Learning and Random Forest Models to identify plant diseases from images and predict potential diseases based on environmental data (temperature & humidity).
It aims to help farmers detect plant diseases early and take the right preventive measures to protect their crops and maximize yield.

🚀 Features

🧠 Image-Based Disease Recognition:
Upload a leaf image — the trained CNN (TensorFlow/Keras) model classifies it into one of 38 disease categories.

🌦️ Environment-Based Prediction:
The system also predicts potential future diseases using Random Forest models trained on environmental factors.

💡 Smart Recommendations:
For each disease detected, the app provides actionable suggestions and treatment guidance.

💻 Streamlit Dashboard:
An interactive, easy-to-use UI for farmers, researchers, and agritech innovators.

🧰 Tech Stack
Component	Technology
Frontend	Streamlit
Backend	TensorFlow, Scikit-learn, Pandas, Numpy
Model Storage	Pickle (.pkl)
Image Processing	TensorFlow Keras Preprocessing
Language	Python 3.x
🧩 How It Works

Upload an Image:
Go to the Disease Recognition page and upload a crop leaf image.

AI Prediction:
The app uses a trained CNN model to predict the disease class.

Environmental Analysis:
Enter current temperature and humidity — the Random Forest model predicts possible upcoming diseases.

Recommendations:
The app displays treatment suggestions and preventive measures for detected diseases.

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/HritikChandravnshi/kishan1.git
cd kishan1

2️⃣ Create and Activate a Virtual Environment
python3 -m venv env
source env/bin/activate  # On Mac/Linux

3️⃣ Install Required Libraries
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run app.py


Replace app.py with your actual file name if different.

📁 Project Structure
kishan1/
│
├── Model/
│   ├── trained_model.keras
│   ├── rf_model_<plant_name>.pkl
│
├── app.py
├── requirements.txt
└── README.md

🌱 Dataset Info

The dataset consists of 87,000+ RGB images of healthy and diseased crop leaves.

Categorized into 38 classes across multiple crops.

Split into Train (80%) and Validation (20%), with 33 test images for prediction evaluation.

💬 Example Output

When an image of a diseased leaf is uploaded:

Predicted Disease from image: Tomato___Late_blight
Suggested Solution: Use resistant varieties and apply fungicides.
Predicted Future Disease based on Environmental Data: Tomato___Early_blight
Suggested Solution based on Environment: Apply fungicides and avoid overhead watering.

👨‍💻 Author

Hritik Chandravanshi
Computer Science Engineer | Web & AI Enthusiast
🌐 GitHub Profile

⭐ Future Improvements

🌾 Integrate real-time weather APIs for live disease forecasting.

🛰️ Add satellite-based soil moisture tracking.

📱 Develop a farmer-friendly mobile app interface.
