# FlightPricePrediction_ML
This project predicts flight ticket prices using machine learning. It includes a Streamlit web app where users can enter flight details such as airline, source city, destination city, departure time, arrival time, number of stops, duration, class, and days left to travel. The model then predicts the expected ticket price and  Premium category.
# ✈️ Flight Price Prediction

## 🔗 Live Demo
👉 [https://flightpricepredictionml-a3vdf3qls2upkvtofrefkb.streamlit.app/]

---

## 💡 About This Project

As part of my Data Science and Machine Learning training, I built this project to predict flight prices and categorize them as Budget, Mid, or Premium using a real dataset of over 300,000 flight bookings.

This project covers both Decision Tree and Random Forest algorithms, applied to the same dataset for both regression and classification tasks.

---

## 🔍 What This App Does

- Predicts the exact **flight price** using Decision Tree Regressor and Random Forest Regressor
- Classifies flights into **Budget / Mid / Premium** using Decision Tree Classifier and Random Forest Classifier
- Lets you switch between all 4 models and compare predictions instantly

---

## 🛠️ Tools Used

- Python, Pandas, NumPy
- Scikit-learn
- Streamlit

---

## 📊 Model Results

| Model | Task | Score |
|---|---|---|
| Decision Tree | Price (Regression) | R² = [0.98] |
| Random Forest | Price (Regression) | R² = [0.98] |
| Decision Tree | Price Category (Classification) | Accuracy = [0.962] |
| Random Forest | Price Category (Classification) | Accuracy = [0.962] |

---

## 👩‍💻 About Me

I am Komal, an AIML student at PCTE Group of Institutes, currently doing a Data Science and Machine Learning training program. This project helped me understand the practical difference between single Decision Trees and ensemble methods like Random Forest.
