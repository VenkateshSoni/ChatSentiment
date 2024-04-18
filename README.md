# ChatSentiment

ChatSentiment is a Streamlit web application for analyzing the sentiment of chat messages. It uses the Hugging Face Llama model for generating responses and TextBlob for sentiment analysis.

## Features

- **Sentiment Analysis:** Analyzes the sentiment of chat messages and generates a report showing the distribution of positive, negative, and neutral sentiments.
- **Keyword Frequency Analysis:** Extracts keywords from chat messages and displays the top 10 most frequent keywords.
- **Word Cloud Visualization:** Generates a word cloud visualization based on the extracted keywords.
- **Clear Conversation:** Allows users to clear the conversation history.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/VenkateshSoni/ChatSentiment.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ChatSentiment
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running with Streamlit

Run the following command to start the Streamlit app:

```bash
streamlit run app.py
```

Once the app is running, you can input your questions in the chat interface and view the sentiment analysis report, keyword frequency analysis, and word cloud visualization in real-time.

### Running with Docker

Alternatively, you can run the application using Docker. First, pull the Docker image from the repository:

```bash
docker pull venkateshsoni/chatsentiment:1.1
```

Then, run the Docker container:

```bash
docker run -p 80:80 venkateshsoni/chatsentiment:1.1
```

This will start the Streamlit app inside a Docker container, and you can access it by navigating to [http://localhost](http://localhost) in your web browser.

## Documentation

### Functions

- **`init_page()`:** Initializes the Streamlit page configuration, setting the page title to "ChatSentiment", adding a header "ChatSentiment", and setting the title of the sidebar to "Options".
- **`init_messages()`:** Initializes the messages session state by adding a clear conversation button to the sidebar. If the button is clicked or the session state does not contain messages, it adds a default system message to the messages session state.
- **`generate_answer(question)`:** Generates an answer for the given question using the Hugging Face Llama model.
- **`analyze_sentiment(text)`:** Analyzes the sentiment of the given text using TextBlob.
- **`extract_keywords(messages)`:** Extracts keywords from the given messages.
- **`render_sentiment_report(messages)`:** Renders a sentiment analysis report based on the given messages, including sentiment analysis, keyword frequency analysis, and word cloud visualization.
- **`main()`:** Main function to run the Streamlit app, which initializes the page, messages, gets user input, generates responses, and displays the sentiment report and chat messages.

## Contributing

Contributions are welcome! If you have any suggestions, enhancements, or bug fixes, please feel free to open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
```

With these instructions, users can choose to run the application either directly using Streamlit or through Docker.
