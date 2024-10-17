# import library
import streamlit as st
import pandas as pd
import plotly.express as px
import helper_functions as hf

# ignore warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# add title
st.set_page_config(page_title="Unveiling hidden insights from user reviews",
                   page_icon=":phone:", layout="wide")
st.title('JKN Mobile User Reviews Dashboard')
st.write('Understanding user experiences of JKN Mobile App: Sentiment Analysis and Topic Modeling Approach')

# adjust the layout of the app
adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

# read the csv
df = pd.read_csv('df_result(17-okt).csv')

# add sidebar for filter
with st.sidebar:

    # add notes
    st.markdown(
        """
        This dashboard seeks to display the underlying sentiments and concerns expressed by JKN Mobile users,
        under the period of January 01, 2024, until October 01, 2024.
        """
    )

    # add 2 columns for summary stats
    col1, col2 = st.columns(2)

    with col1:
        total_reviews = df['Text'].nunique()
        st.metric('Total reviews', value=total_reviews)

    with col2:
        avg_rating = df.score.mean()
        st.metric('Average rating', value=round(avg_rating,3))

    # add notes
    st.markdown(
        """
        **Legend:**
        - Green: Positive sentiment
        - Grey: Neutral sentiment
        - Red: Negative sentiment
        """
    )

    st.markdown("Created by Roissyah Fernanda")

# add 2 charts
col3, col4 = st.columns(2)

with col3:
    # sentiment proportion
    st.subheader('Sentiment Distribution')
    sentiment_plot = hf.plot_sentiment(df)
    sentiment_plot.update_layout(height=350, title_x=0.5)
    st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)

with col4:
    # rating distribution
    st.subheader('Rating Distribution')
    ratings = df['score'].value_counts().reset_index()
    ratings.columns = ['score', 'count']
    fig = px.bar(ratings, y='count', x='score')
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

# stacked bar-chart: number of reviews by month-year, for each sentiment 
# group by month and sentiment, then count the number of reviews
monthly_sentiment_counts = df.groupby(['month_name', 'sentiment'])['sentiment'].count().unstack().fillna(0)

# sort the months in ascending order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']
monthly_sentiment_counts = monthly_sentiment_counts.reindex(month_order)

# create a new DataFrame in long format for Plotly
df_plotly = monthly_sentiment_counts.reset_index().melt(id_vars='month_name', var_name='sentiment', value_name='count')

# define a color mapping for sentiments
color_map = {'positive': 'green', 'neutral': 'grey', 'negative': 'red'}

fig = px.bar(df_plotly, x='month_name', y='count', color='sentiment',
             color_discrete_map=color_map,
             )

fig.update_layout(xaxis_title='Month', yaxis_title='Number of Reviews', showlegend=False)
st.subheader("Monthly Review Counts by Sentiment")
st.plotly_chart(fig, theme='streamlit', use_container_width=True)

# add options, to choose sentiment
options = df['sentiment'].unique()
default_var = "positive"
sentiment_selection = st.selectbox("Choose a sentiment category:", options)

# define color mapping for sentiments
color_map = {
    "positive": "green",
    "neutral": "grey",
    "negative": "red"
}

# get the appropriate color based on the selected sentiment
selected_color = color_map.get(sentiment_selection, "blue")  # Default to blue if not found

# add more 3 charts
col6, col7, col8 = st.columns(3)

with col6:
    st.subheader(f'Bigrams of {sentiment_selection} sentiment')
    fig = hf.create_bigram_barplot(df, 'Text', 'sentiment', f'{sentiment_selection}', selected_color)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

with col7:
    st.subheader(f'Trigrams of {sentiment_selection} sentiment')
    fig = hf.create_trigram_barplot(df, 'Text', 'sentiment', f'{sentiment_selection}', selected_color)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

with col8:
    # wordcloud
    st.subheader(f'Word cloud of {sentiment_selection} sentiment')
    fig = hf.create_wordcloud(df, 'Text', 'sentiment', f'{sentiment_selection}')
    st.pyplot(fig)

# add more 2 charts
col9, col10 = st.columns(2)

with col9:
    # group the data by topic and sentiment, then count the occurrences
    topic_sentiment_counts = df.groupby(['Topic-Interpretation', 'sentiment'])['sentiment'].count().unstack().fillna(0)
    # sort values by the total number of sentiments in descending order
    topic_sentiment_counts['Total'] = topic_sentiment_counts.sum(axis=1)
    topic_sentiment_counts = topic_sentiment_counts.sort_values('Total', ascending=True).drop('Total', axis=1)

    fig = px.bar(
    topic_sentiment_counts,
    x=topic_sentiment_counts.columns.tolist(),
    y=topic_sentiment_counts.index.tolist(),
    orientation='h',
    barmode='stack',
    #title='Topic Distribution by Sentiment',
    labels={'value': 'Count', 'Topic-Interpretation': 'Topic Interpretation', 'variable': 'Sentiment'},
    color_discrete_map={'positive': 'green', 'neutral': 'grey', 'negative': 'red'}
    )

    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        legend_title='Sentiment',
        width=1000,
        height=800,
        showlegend=False
    )

    st.subheader('Topic Distribution by Sentiment')
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

with col10:
    # table of original reviews, its score, its sentiment, sentiment-score
    df_display = df.sample(frac=0.001).reset_index()
    st.subheader('Reviews Table (sample)')
    st.dataframe(df_display[['content', 'score', 'sentiment', 'sentiment-score']].style.applymap(
        hf.sentiment_color, subset='sentiment'
    ))