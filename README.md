# 🛒 Brazilian E-Commerce Price Optimization  
### Data Science & Data Analysis Integrated Project  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![ML](https://img.shields.io/badge/ML-Regression%2C%20RandomForest%2C%20XGBoost-orange) ![DataScience](https://img.shields.io/badge/Data_Science-Price%20Prediction%2C%20Modeling-yellowgreen)  
![License](https://img.shields.io/badge/License-MIT-green) ![GitHub](https://img.shields.io/badge/Repo%20Style-Multi%20Track-brightgreen)

This repository presents a **data-driven project** aimed at optimizing product pricing strategies in Brazil’s e-commerce market, with a focus on the top 10 product categories.  
The project integrates **Data Science** (for predictive modeling) and **Data Analysis** (for extracting actionable insights) under one unified business problem:

> **How can sellers optimize product pricing strategies to remain competitive and profitable across Brazil’s e-commerce top 10 product categories?**

---

## 🎯 Business Context  

Brazil’s online marketplace is diverse, with thousands of sellers offering various products with significant price variability. Sellers struggle to maintain competitive pricing while ensuring profitability.

The analysis explores key factors:
- Product characteristics (weight, dimensions, description length)
- Seller and customer locations (logistics impact)
- Delivery performance (on-time rate, delays)
- Customer satisfaction (reviews, ratings)

---

## 🧩 Project Objectives  

1. **Price Prediction (Data Science Track)**  
   - Develop regression models to predict product prices.  
   - Evaluate models using **MedAE**, **MAE**, **R²**, and **MAPE** metrics.  
   - Identify key price influencers such as product weight, description length, and seller location.

2. **Market Insights (Data Analyst Track)**  
   - Explore customer behavior, review sentiment, and pricing patterns across different product categories.  
   - Investigate the impact of delivery speed, customer reviews, and payment preferences on product pricing and satisfaction.  

3. **Affiliate Program Design**  
   - Simulate a business model where sellers subscribe to data-driven pricing tools.  
   - Evaluate the impact of the affiliate program on seller performance and marketplace revenue.

---

## 📊 Dataset Overview  

The dataset is based on the **Brazilian E-Commerce Public Dataset (Olist)**, containing tables with order details, product attributes, reviews, payments, and customer data.

You can access the dataset on [Kaggle here](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) for more details.

| Table | Description |
|-------|-------------|
| `order_items` | Details of each item in the order (price, seller, product attributes) |
| `orders` | Order lifecycle details (timestamps, status) |
| `products` | Product features (weight, dimensions, description, category) |
| `reviews` | Customer feedback (rating, sentiment) |
| `payments` | Payment methods, installments, values |
| `customers` | Customer demographics and geographic info |
| `sellers` | Seller locations and postal codes |

---

## 🧠 Data Science Pipeline  

1. **Data Merging & Cleaning**  
   - Merged multiple data tables into a single comprehensive dataset.  
   - Addressed missing values and retained significant outliers like premium products and rare delivery events.

2. **Feature Selection**  
   - Chose key features: product weight, dimensions, description length, seller location, and review score.  
   - Target variable: `price`

3. **Preprocessing**  
   - **Encoding:** OneHotEncoder for categorical features (seller city, product category).  
   - **Scaling:** RobustScaler for numerical features (product size, weight).  
   - **Log Transformation:** Applied to `price` to normalize distribution.

4. **Modeling & Evaluation**  
   - Models: `RandomForest`, `LightGBM`, `XGBoost`, `LinearRegression`  
   - Cross-validation: 5-fold  
   - Evaluation metrics: **MedAE**, MAE, R², Max Error, MAPE  

5. **Best Model: RandomForest (Log-Transformed)**  
   - Test results:  
     - **MedAE:** 2.03  
     - **MAE:** 14.71  
     - **R²:** 0.81  
     - **MAPE:** 11.5%  
   - Key predictors: `product_weight_g`, `product_description_length`, `product_height_cm`, `seller_state`.

6. **Deployment**  
   - Model saved using `joblib` for future deployment.  
   - Streamlit integration for live price prediction dashboard.

---

## 📈 Data Analyst Insights  

### 1. Pricing & Profitability  
- Premium products tend to have longer descriptions and higher-quality images.  
- Freight costs correlate with higher product prices, especially for heavier items.  
- Higher prices generally lead to better review scores.

### 2. Delivery Performance  
- Delayed deliveries result in lower customer satisfaction.  
- São Paulo & Rio de Janeiro show the fastest delivery times.  
- Delivery speed is the strongest predictor of review scores.

### 3. Customer Sentiment (WordCloud Insights)  
- Positive reviews: *“entrega rápida”, “ótimo produto”, “recomendo”*  
- Negative reviews: *“não recebi”, “veio errado”, “faltando”*  
- Fast, accurate, and well-packaged deliveries lead to higher customer satisfaction.

### 4. Regional Pricing  
- States like **Rondônia (RO)** and **Rio Grande do Norte (RN)** show higher average prices, likely due to logistic costs and regional demand.  
- **São Paulo** remains competitive with lower pricing due to high supply.

---

## 🛠️ Tech Stack  

| Area               | Tools                                            |
|--------------------|--------------------------------------------------|
| Data Wrangling     | Python, Pandas, NumPy                            |
| Data Visualization | Matplotlib, Seaborn, Plotly, Tableau             |
| Machine Learning   | Scikit-learn, LightGBM, XGBoost                  |
| Sentiment Analysis | WordCloud, NLP Preprocessing                     |
| Deployment         | Streamlit (interactive dashboard)                |
| Project Management | GitHub, Jupyter Notebooks                        |

---

## 📚 Folder Structure  

```

📂 Brazil-Ecommerce-Price-Optimization/
│
├── 📁 data/                # Raw & cleaned datasets
├── 📁 notebooks/           # Jupyter notebooks (EDA, modeling, sentiment)
├── 📁 models/              # Saved ML models (.joblib)
├── 📁 visuals/             # Charts, plots, and wordcloud images
├── 📄 requirements.txt     # Library dependencies for Streamlit
├── 📄 how_to_run.md        # Instructions on how to run the Streamlit app
├── 🖥️ app.py               # Streamlit app (for deployment)
└── 📖 README.md            # Project overview

```

---

## 🧾 Key Results Summary  

| Track | Focus | Key Output |
|--------|--------|-------------|
| **Data Science** | Predictive Modeling | Log-Transformed Tuned RandomForest (R²=0.81) |
| **Data Analyst** | Business Insights | Cross-category sentiment, pricing, delivery, & customer trends |
| **Integration** | Business Strategy | Seller Affiliation Program with +R$115K profit/month |

---

## 🚀 Deployment  

This project is deployed using **Tableau** for visualizing the data and insights from the price optimization analysis.

- **Tableau Dashboard**: [Link to Tableau Dashboard]([https://your-tableau-dashboard-link](https://public.tableau.com/app/profile/uriel.siboro/viz/BrazilianE-CommerceFINAL/Dashboard2?publish=yes))  
- **Google Drive**: [Link to Google Drive]([https://your-google-drive-link](https://drive.google.com/drive/folders/1ujXQRfHrQEJQd03RCn6wDSkgamlikWOH?usp=sharing))

These links provide access to the interactive visualizations and detailed reports generated during the project, allowing users to explore the results and insights further.

---

## 🏁 License  
This project is licensed under the MIT License — feel free to use and adapt with attribution.
