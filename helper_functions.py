# import library
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# function to draw pie plot of "sentiment distribution"
def plot_sentiment(dataframe):
    # count the number tweets based on the sentiment
    sentiment_count = dataframe["sentiment"].value_counts()

    # plot the sentiment distribution in a pie chart
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        #title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        # set the color of each sentiment class
        color_discrete_map={"positive": "green", "neutral": "grey", "negative": "red"},
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=False)
    return fig

# function to draw bigrams (per sentiment category)
def create_bigram_barplot(dataframe, text_column, sentiment_column, sentiment_class, color):
    """
    Generate and save a bigram horizontal barplot from a specified column in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the bigram from.
    sentiment_column (str): The name of the column containing the sentiment labels.
    sentiment_class (str): The sentiment class to filter the DataFrame (positive, neutral, or negative).
    color (str): Color of the barchart
    
    Returns:
    None
    """

    # Filter the DataFrame for the specified sentiment class
    df_filtered = dataframe.loc[dataframe[sentiment_column] == sentiment_class].copy()

    # Tokenization
    df_filtered['tokens'] = df_filtered[text_column].apply(lambda x: x.split())

    # Create Bigrams
    df_filtered['bigrams'] = df_filtered['tokens'].apply(lambda x: [x[i] + " " + x[i+1] for i in range(len(x)-1)])

    # Counting bigram frequency
    bigram_freq = Counter([item for sublist in df_filtered['bigrams'] for item in sublist])

    # Creating DataFrame for top 15 bigrams
    bigrams_freq_df = pd.DataFrame(bigram_freq.most_common(15), columns=['Bigram', 'Frequency'])

    # Creating bar plot
    fig = px.bar(
        bigrams_freq_df, y=bigrams_freq_df['Bigram'], x=bigrams_freq_df['Frequency'],
        color_discrete_sequence=[color]
    )

    return fig

# function to draw trigrams (per sentiment category)
def create_trigram_barplot(dataframe, text_column, sentiment_column, sentiment_class, color):
  """
    Generate and save a trigram horizontal barplot from a specified column in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the bigram from.
    sentiment_column (str): The name of the column containing the sentiment labels.
    sentiment_class (str): The sentiment class to filter the DataFrame (positive, neutral, or negative).
    barchart_title (str): The title of the barchart.
    color (str): The color of the bars in the barchart.

    Returns:
    None
  """

  # Filter the DataFrame for the specified sentiment class
  df = dataframe[dataframe[sentiment_column] == sentiment_class].copy()

  # Tokenization
  df['tokens'] = df[text_column].apply(lambda x: x.split())

  # Trigram (change)
  df['trigrams'] = df['tokens'].apply(lambda x: [x[i] + " " + x[i+1] + " " + x[i+2] for i in range(len(x)-2)])

  # Counting trigram frequency
  trigram_freq = Counter([item for sublist in df['trigrams'] for item in sublist])

  # Creating DataFrame for top bigrams
  trigrams_freq_df = pd.DataFrame(trigram_freq.most_common(15), columns=['Trigram', 'Frequency'])

  # Creating bar plot
  fig = px.bar(
        trigrams_freq_df, y=trigrams_freq_df['Trigram'], x=trigrams_freq_df['Frequency'],
        color_discrete_sequence=[color]
    )
  
  return fig

# function to draw word clouds (per sentiment category)
def create_wordcloud(dataframe, text_column, sentiment_column, sentiment_class):
    """
    Generate and save a word cloud from a specified column in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the word cloud from.
    sentiment_column (str): The name of the column containing the sentiment labels.
    sentiment_class (str): The sentiment class to filter the DataFrame (positif, netral, or negatif).

    Returns:
    None
    """
    # Filter the DataFrame for the specified sentiment class
    df = dataframe[dataframe[sentiment_column] == sentiment_class]

    # Choose dataset column
    text_data = df[text_column]

    # Generate word cloud
    all_text = ' '.join(text_data.tolist())
    word_cloud = WordCloud(max_words=100, background_color='white',
                           random_state=100, colormap='seismic').generate(all_text)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(word_cloud, interpolation='bilinear')
    #plt.title(f'WordCloud of Frequently Used Words in {sentiment_class}', fontsize=20)
    plt.axis("off")
    #plt.show()

    return fig

# function to change background column in a df, according to its sentiment category
def sentiment_color(sentiment):
    if sentiment == "positive":
        return "background-color: green; color: white"
    elif sentiment =='neutral':
        return "background-color: grey"
    else:
        return "background-color: red; color: white"
