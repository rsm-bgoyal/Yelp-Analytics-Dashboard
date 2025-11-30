# 🍽️ Yelp Analytics Dashboard  
### Interactive Analysis of Ratings, Prices, Sentiment & Cuisines

This dashboard analyzes Yelp business reviews across California with a focus on **restaurants, cafes, bars**, and other food-related businesses.

It uses a cleaned dataset derived from the Yelp Open Dataset combined with additional sentiment scoring.  
The Streamlit interface enables fast exploration of trends, cuisine performance, geographic patterns, and customer sentiment.

---

## 🚀 Live App  
👉 https://yelp-analytics-dashboard-e877zsqi6mcjvdqcrrtvxa.streamlit.app/

---

## 📦 Features

- Interactive global filters  
- City, date, price, rating, and sentiment slicing  
- Cuisine-level insights  
- Geo-spatial restaurant mapping  
- Sentiment and review language analytics  
- Outlier detection (hidden gems, overpriced spots)  
- Exportable filtered data table  

---

## 🧹 Data Cleaning Summary

The raw Yelp categories include many unrelated business types (e.g., gyms, churches, salons, nutritionists, grocery stores).  
A custom cleaning pipeline was applied:

### ✔ Food-business classification  
Each business is kept **only if** category tokens indicate it is a food place:

- restaurants, cafes, bars, coffee, bakery, brunch, pubs, breweries, etc.  
- **and** does *not* contain any unrelated tags  
  (e.g., church, salon, fitness, weight loss, grocery, nutritionist, school, etc.)

### ✔ Token-level category cleaning  
Remaining category strings are cleaned to remove non-food tokens.

### ✔ Additional processing  
- Converted time columns  
- Generated sentiment labels  
- Created exploratory variables  
- Downcasted numeric types for memory efficiency  
- Saved a cleaned, GitHub-safe dataset



