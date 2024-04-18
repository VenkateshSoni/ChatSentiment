import streamlit as st 
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import textblob
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from gradio_client import Client
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from getpass import getpass
import os
import nltk
from nltk.corpus import brown

"""Downloading punkt for textblob sentiment analysis."""
nltk.download('punkt')

HUGGINGFACEHUB_API_TOKEN: st.secrets["HUGGINGFACEHUB_API_TOKEN"]

def init_page() -> None:
    """
    Initializes the Streamlit page configuration.

    Sets the page title to "ChatSentiment", adds a header "ChatSentiment",
    and sets the title of the sidebar to "Options".
    """
    st.set_page_config(
        page_title="ChatSentiment"
    )
    st.header("ChatSentiment")
    st.sidebar.title("Options")

def init_messages() -> None:
    """
    Initializes the messages session state.

    Adds a clear conversation button to the sidebar. If the button is clicked
    or the session state does not contain messages, it adds a default system message
    to the messages session state.
    """
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in markdown format."
            )
        ]

def generate_answer(question):
    """
    Generates an answer for the given question using the Hugging Face Llama model.

    Parameters:
        question (str): The input question.

    Returns:
        str: The generated answer.
    """
    client = Client("huggingface-projects/llama-2-7b-chat")
    result = client.predict(
        message=question,
        request=question,
        param_3=1024,
        param_4=0.6,
        param_5=0.9,
        param_6=50,
        param_7=1.2,
        api_name="/chat"
    )
    return result

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using TextBlob.

    Parameters:
        text (str): The text to analyze.

    Returns:
        str: The sentiment of the text (Positive, Neutral, or Negative).
    """
    tb = TextBlob(text)
    polarity = round(tb.polarity, 2)

    # Determine emotion based on polarity
    if polarity > 0:
        emotion = "Positive"
    elif polarity == 0:
        emotion = "Neutral"
    else:
        emotion = "Negative"

    return emotion

def extract_keywords(messages):
    """
    Extracts keywords from the given messages.

    Parameters:
        messages (list): A list of messages.

    Returns:
        list: A list of extracted keywords.
    """
    keywords = []
    for message in messages:
        if isinstance(message, AIMessage):
            # Tokenize the message content and extract keywords
            blob = TextBlob(message.content)
            keywords.extend([word.lower() for word in blob.words if len(word) > 2 and word.isalpha()])
    return keywords

def render_sentiment_report(messages):
    """
    Renders a sentiment analysis report based on the given messages.

    Parameters:
        messages (list): A list of messages.
    """
    if not messages:
        st.subheader("Sentiment Analysis Report")
        st.write("No messages to analyze.")
        return

    sentiments = []
    message_lengths = []
    response_times = []
    keywords = extract_keywords(messages)

    for message in messages:
        if isinstance(message, AIMessage):
            sentiment = analyze_sentiment(message.content)
            sentiments.append(sentiment)
            message_lengths.append(len(message.content.split()))
            # Add logic to calculate response times

    # Sentiment Analysis
    if sentiments:
        sentiment_counts = pd.Series(sentiments).value_counts()
        fig1 = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                    labels={'x': 'Sentiment', 'y': 'Count'}, color=sentiment_counts.index,
                    )

        fig2 = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                                    title='Distribution of Sentiments')])

        st.subheader("Sentiment Analysis Report")
        st.subheader("Count of Sentiments:")
        st.plotly_chart(fig1, use_container_width=True)
        st.subheader("Distribution of Sentiments")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.subheader("Sentiment Analysis Report")
        st.write("No messages to analyze.")

    # Keyword Frequency
    if keywords:
        keyword_counts = Counter(keywords)
        most_common_keywords = keyword_counts.most_common(10)
        keyword_df = pd.DataFrame(most_common_keywords, columns=["Keyword", "Frequency"])
        st.subheader("Keyword Frequency Analysis")
        st.write("Top 10 most frequent keywords:")
        st.table(keyword_df)

    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(' '.join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.subheader("Word Cloud")
    st.pyplot(plt)

def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    init_page()
    init_messages()

    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Bot is typing ..."):
            answer = generate_answer(user_input)
            print(answer)
        st.session_state.messages.append(AIMessage(content=str(answer)))

    if st.sidebar.button("Get Sentiment Report"):
        render_sentiment_report(st.session_state.messages)

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

if __name__ == "__main__":
    main()