# JKN Mobile User Reviews 2024 Dashboard 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jkn-mobile-reviews-2024.streamlit.app/) 

## 🔍 About the project

This Streamlit application empowers users to deep dive into JKN Mobile Users Reviews, under the period of January 01, 2024, until October 01, 2024.

By scraping user reviews from Google Playstore, analyzing the sentiment of the text along with the undelying topic, this dashboard will provide valuable insights into the app’s strengths, weaknesses, and potential areas for improvement.

This project involved several key steps:

1. Data Collection: Scraping user reviews from Google Playstore using 'google-play-scraper' library.
2. Data Cleaning: Remove duplicate reviews, handle missing value.
3. Text Cleansing: Remove punctuations, emoji, stopwords, handle slang words, ensuring the text is clean to proceed further.
4. Exploratory Data Analysis: The data was thoroughly examined to understand its characteristics and identify potential patterns.
3. Sentiment Analysis Labeling: Labeling the reviews using the [Indonesia-bert-sentiment-classification](https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification) model available in Hugging Face.
4. Text Analysis: Drawing bigrams, trigrams, word clouds from each sentiment category to unleash pattern.
5. Topic Modeling: Employing LDA to identify the main themes and topics discussed in the reviews.
6. Dashboard creation: For interactive visualization and easy customization, Streamlit was used to create dashboard.
7. Create narrative report: Present the findings and comprehensive analysis in a narrative format [medium article](https://medium.com/@roissyahfernanda/understanding-jkn-mobile-user-experiences-a-sentiment-analysis-and-topic-modeling-approach-9f0b6d9f8177)


## ⚒️ Setup environment
```
conda create --name jkn-mobile-2024-dashboard python=3.9
conda activate jkn-mobile-2024-dashboard
pip install requirements.txt
```

## 🚀 Run streamlit app
```
streamlit run app.py
```