 Movie Recommender Bot

This project is a Movie Recommender Bot that suggests movies based on user queries. It uses a CSV file containing top-rated movies and leverages natural language processing (NLP) techniques to understand user input and provide relevant movie recommendations.

## Project Structure

```
main.py
top_rated_movies.csv
.vscode/
    settings.json
```

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/movie-recommender-bot.git
    cd movie-recommender-bot
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download NLTK resources:**
    The necessary NLTK resources will be downloaded automatically when you run the script for the first time.

## Usage

1. **Run the script:**
    ```sh
    python main.py
    ```

2. **Interact with the bot:**
    The bot will prompt you to enter your movie queries. You can ask for movie recommendations based on genres, similar movies, or top-rated movies.

## Example Queries

- "Recommend movies like Inception"
- "Show me good action movies"
- "Best comedy movies"
- "I want to watch a scary movie"


## Project Details

- **Data Source:** The bot uses a CSV file (`top_rated_movies.csv`) containing 10,000 top-rated movies.
- **NLP Techniques:** The bot uses NLTK for text processing and scikit-learn for building a TF-IDF model and calculating cosine similarity.
- **Error Handling:** The bot includes robust error handling to manage various edge cases, such as missing data or invalid user input.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
