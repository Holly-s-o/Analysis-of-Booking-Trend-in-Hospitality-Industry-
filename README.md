# Analysis-of-Booking-Trend-in-Hospitality-Industry-
This project analyses hotel booking data to uncover patterns and factors influencing booking cancellations and customer behaviour in the hospitality industry. The goal was to build a machine learning model capable of predicting whether a booking would be cancelled or not — and to extract actionable insights for better decision-making.

## Objectives

* Identify key factors contributing to booking cancellations.
* Explore booking trends across customer types, lead times, and market segments.
* Build a predictive model to classify bookings as *Canceled* or *Not Canceled*.
* Provide data-driven recommendations to reduce cancellations and improve customer retention.

## Dataset

The dataset contains details such as:

* Number of adults and children
* Weekend and weekday nights
* Meal plan type
* Room type
* Lead time before check-in
* Market segment (online/offline)
* Special requests
* Average price per night
* Repeat guest indicator
* Booking status (Cancelled/Not Cancelled)

**Target Variable:** `booking_status`

## Data Preprocessing

Steps performed include:

1. Handling missing values and dropping irrelevant features.
2. Converting `date_of_reservation` into a  datetime format.
3. Encoding categorical columns (`type_of_meal`, `room_type`, `market_segment_type`) using LabelEncoder.
4. Scaling numeric features (`lead_time`, `average_price`, etc.) for model efficiency.

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
```

## Exploratory Data Analysis (EDA)

### Key Insights:

* **Lead Time:** Longer lead times show higher cancellation likelihood.
* **Market Segment:** Online bookings are cancelled more often than offline ones.
* **Repeat Guests:** Repeat customers are far less likely to cancel.
* **Average Price:** Higher prices slightly increase cancellations.

## Model Building

A **Random Forest Classifier** was selected for its accuracy and interpretability.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Model Performance

| Metric        | Score |
| :------------ | ----: |
| **Accuracy**  |  ~92% |
| **Precision** |   92% |
| **Recall**    |   92% |
| **F1 Score**  |   92% |

The model effectively predicts booking outcomes with high reliability.

## Feature Importance

Top predictors influencing booking status:

1. Market Segment Type 
2. Special requests
3. Lead Time
4. Car Parking Space 
5. Reservation for the year

These reveal that *Marketing, customer intent,* and *timing* drive most booking behaviours.

## Insights & Recommendations

1. **Reduce Lead-Time Cancellations**

   * Offer flexible cancellation options for early bookers.
   * Send timely reminders or discounts closer to the check-in date.

2. **Reward Loyalty and attend more to special requests**

   * Incentivise repeat guests with loyalty programs.
   * Personalise offers to returning customers.
   * Intentionality towards special requests. 

3. **Optimize Online Channels**

   * Improve online user experience.
   * Use chatbots or follow-up emails for booking confirmations.

4. **Dynamic Pricing Strategy**

   * Adjust prices based on demand forecasts to reduce dropouts.

## Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn
* **Model:** Random Forest Classifier
* **Environment:** Jupyter Notebook

## Author

**Holiness Segun-Olufemi**
Public Policy Professional | Data Scientist | Researcher





