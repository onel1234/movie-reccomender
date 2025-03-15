data_path = 'top_rated_movies.csv' 


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import os

# First, create the NLTK data directory if it doesn't exist
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Download necessary NLTK data properly
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Import NLTK components after downloading resources
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class MovieRecommender:
    def __init__(self, data_path):
        """Initialize the recommender with data from a CSV file."""
        try:
            print(f"Attempting to read file: {data_path}")
            
            # Try different separators to read the CSV correctly
            self.df = None
            for sep in ['\t', ',', ';', '|']:
                try:
                    temp_df = pd.read_csv(data_path, sep=sep, encoding='utf-8', engine='python', nrows=5)
                    if len(temp_df.columns) > 1:
                        self.df = pd.read_csv(data_path, sep=sep, encoding='utf-8', engine='python')
                        print(f"Successfully read file with separator: '{sep}'")
                        break
                except Exception as e:
                    print(f"Failed with separator '{sep}': {e}")
            
            # If standard separators fail, try more permissive reading
            if self.df is None or self.df.empty:
                self.df = pd.read_csv(data_path, encoding='utf-8', engine='python', on_bad_lines='skip')
                print("Used permissive reading settings to read the file")
            
            # Display column information for debugging
            print(f"Columns found in the dataset: {self.df.columns.tolist()}")
            print(f"Dataset shape: {self.df.shape}")
            
            # Handle case-sensitive column names or whitespace issues
            self._standardize_columns()
            
            # Preprocess the data and build the recommendation model
            self.preprocess_data()
            self.build_recommendation_model()
            
            # Define genre keywords for query matching
            self._initialize_genre_keywords()
            
            print("MovieRecommender initialized successfully.")
            
        except Exception as e:
            print(f"Error initializing recommender: {e}")
            raise
    
    def _standardize_columns(self):
        """Handle column name variations and ensure required columns exist."""
        required_columns = ['overview', 'original_title', 'release_date', 'vote_average', 'vote_count']
        
        # Create mapping for column names (handle case and whitespace variations)
        column_mapping = {}
        for req_col in required_columns:
            for col in self.df.columns:
                if col.strip().lower() == req_col.lower():
                    column_mapping[col] = req_col
                    break
        
        # Rename columns if needed
        if column_mapping:
            self.df = self.df.rename(columns=column_mapping)
        
        # Add missing columns with empty values if they don't exist
        for col in required_columns:
            if col not in self.df.columns:
                print(f"Warning: Adding missing column: {col}")
                self.df[col] = ''
    
    def _initialize_genre_keywords(self):
        """Initialize keywords for genre identification in user queries."""
        self.genre_keywords = {
            'funny': ['comedy', 'humor', 'laugh', 'hilarious', 'amusing', 'comical', 'funny'],
            'action': ['action', 'thriller', 'adventure', 'explosive', 'fight', 'chase', 'exciting'],
            'romantic': ['romance', 'love', 'relationship', 'passion', 'romantic', 'couple', 'date'],
            'scary': ['horror', 'thriller', 'terrifying', 'scary', 'frightening', 'supernatural', 'ghost'],
            'dramatic': ['drama', 'emotional', 'powerful', 'intense', 'moving', 'tragic', 'serious'],
            'family': ['family', 'children', 'kids', 'animation', 'cartoon', 'disney', 'pixar'],
            'crime': ['crime', 'detective', 'murder', 'mystery', 'police', 'criminal', 'investigation'],
            'documentary': ['documentary', 'true story', 'real', 'history', 'actual', 'facts', 'documentary'],
            'scifi': ['science fiction', 'sci-fi', 'space', 'future', 'alien', 'robot', 'technology']
        }
        
    def preprocess_data(self):
        """Clean and preprocess the data."""
        try:
            # Handle missing values
            self.df['overview'] = self.df['overview'].fillna('')
            
            # Convert release_date to datetime with multiple format attempts
            try:
                self.df['release_date'] = pd.to_datetime(self.df['release_date'], errors='coerce')
            except Exception as e:
                print(f"Error converting release_date: {e}")
                print("Trying alternative date formats...")
                
                for date_format in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']:
                    try:
                        self.df['release_date'] = pd.to_datetime(self.df['release_date'], 
                                                                format=date_format,
                                                                errors='coerce')
                        if not pd.isna(self.df['release_date']).all():
                            print(f"Successfully parsed dates with format: {date_format}")
                            break
                    except Exception:
                        continue
            
            # Extract year from release_date
            self.df['release_year'] = pd.to_datetime(self.df['release_date'], errors='coerce').dt.year
            
            # Convert numeric columns to appropriate types
            for col in ['vote_average', 'vote_count']:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                except Exception as e:
                    print(f"Error converting {col} to numeric: {e}")
                    self.df[col] = 0
            
            # Clean text in overview
            self.df['cleaned_overview'] = self.df['overview'].apply(self.clean_text)
            
            # Verify we have usable content
            non_empty = self.df['cleaned_overview'].str.strip().ne('').sum()
            print(f"Processed {non_empty} movie overviews with usable content.")
            
            print("Data preprocessing completed successfully.")
        except Exception as e:
            print(f"Error in preprocess_data: {e}")
            raise
        
    def clean_text(self, text):
        """Clean and normalize text with proper error handling."""
        try:
            if pd.isna(text) or text == '':
                return ''
            
            # Convert to lowercase and remove special characters
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize using proper NLTK function
            try:
                # Use explicit call to word_tokenize to avoid punkt_tab issues
                tokens = word_tokenize(text)
                
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]
                
                # Lemmatize
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
                
                return ' '.join(tokens)
            except LookupError as e:
                # If NLTK resource error, try a simpler approach
                print(f"NLTK resource error: {e}. Using simple tokenization.")
                return ' '.join(text.split())
            except Exception as e:
                print(f"Text processing error: {e}. Using original text.")
                return text
                
        except Exception as e:
            print(f"Error in clean_text: {e}")
            return str(text)  # Return original text if cleaning fails
    
    def build_recommendation_model(self):
        """Build the TF-IDF model for content-based filtering."""
        try:
            # Check if we have enough data to build the model
            if len(self.df) == 0:
                print("Warning: No data available to build recommendation model")
                # Create dummy data to avoid errors
                self._create_dummy_model()
                return
                
            # Check if we have usable text content
            non_empty_overviews = self.df['cleaned_overview'].str.strip().ne('')
            if non_empty_overviews.sum() == 0:
                print("Warning: No text content available in cleaned_overview")
                self._create_dummy_model()
                return
            
            # Use only rows with non-empty overviews for the model
            valid_df = self.df[non_empty_overviews].copy()
            
            if len(valid_df) == 0:
                print("Warning: No valid movie descriptions available")
                self._create_dummy_model()
                return
                
            print(f"Building recommendation model with {len(valid_df)} movies...")
            
            # Create TF-IDF vectorizer - limit features to avoid memory issues
            self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            
            # Fit and transform the cleaned overviews
            self.tfidf_matrix = self.tfidf.fit_transform(valid_df['cleaned_overview'])
            
            # Keep track of valid indices for later use
            self.valid_indices = valid_df.index.tolist()
            
            # Compute cosine similarity matrix
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            
            print(f"Recommendation model built successfully with {self.tfidf_matrix.shape[0]} movies and {self.tfidf_matrix.shape[1]} features.")
        except Exception as e:
            print(f"Error in build_recommendation_model: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a minimal model to prevent errors when the real one can't be built."""
        print("Creating fallback recommendation model.")
        dummy_texts = ['action movie', 'comedy film', 'drama story', 'horror scary', 'romance love']
        self.tfidf = TfidfVectorizer(max_features=10)
        self.tfidf_matrix = self.tfidf.fit_transform(dummy_texts)
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.valid_indices = list(range(len(dummy_texts)))
    
    def _index_to_movie_id(self, matrix_index):
        """Convert matrix index to movie dataframe index."""
        if not hasattr(self, 'valid_indices'):
            return matrix_index
        
        # Check bounds
        if matrix_index < 0 or matrix_index >= len(self.valid_indices):
            return None
            
        return self.valid_indices[matrix_index]
    
    def _movie_id_to_index(self, movie_id):
        """Convert movie dataframe index to matrix index."""
        if not hasattr(self, 'valid_indices'):
            return movie_id
            
        try:
            return self.valid_indices.index(movie_id)
        except ValueError:
            return None
    
    def get_recommendations_by_title(self, title, n=5):
        """Get recommendations based on a movie title."""
        try:
            # Find the movie with the most similar title (case-insensitive)
            title_lower = title.lower()
            self.df['title_match'] = self.df['original_title'].str.lower().apply(
                lambda x: self._string_similarity(x, title_lower)
            )
            
            # Sort by similarity and get the best match
            similar_titles = self.df.sort_values('title_match', ascending=False)
            
            if len(similar_titles) == 0 or similar_titles['title_match'].iloc[0] < 0.6:
                return f"Movie '{title}' not found in the database. Please check the spelling or try another title."
                
            matched_movie = similar_titles.iloc[0]
            movie_id = matched_movie.name  # Get the index as movie_id
            
            # Convert to matrix index
            idx = self._movie_id_to_index(movie_id)
            
            if idx is None:
                return f"Sorry, '{matched_movie['original_title']}' was found but doesn't have enough content for recommendations."
                
            # Get pairwise similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Sort movies by similarity
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top n+1 (including the movie itself)
            sim_scores = sim_scores[:n+1]
            
            # Skip the first result if it's the input movie itself
            if sim_scores[0][1] > 0.99:  # Almost identical, probably the same movie
                sim_scores = sim_scores[1:n+1]
            else:
                sim_scores = sim_scores[:n]
            
            # Convert matrix indices to movie dataframe indices
            movie_indices = [self._index_to_movie_id(i[0]) for i in sim_scores]
            movie_indices = [idx for idx in movie_indices if idx is not None]
            
            # Get the recommended movies
            recommendations = self.df.loc[movie_indices][['original_title', 'overview', 'vote_average', 'vote_count']]
            
            if recommendations.empty:
                return f"No similar movies found for '{title}'."
                
            return recommendations
        except Exception as e:
            print(f"Error in get_recommendations_by_title: {e}")
            return f"Error finding recommendations for '{title}': {str(e)}"
    
    def _string_similarity(self, str1, str2):
        """Calculate string similarity ratio for fuzzy title matching."""
        # Simple method using longest common substring
        if not str1 or not str2:
            return 0
            
        if str1 in str2 or str2 in str1:
            return 0.9
            
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        return matches / max(len(str1), len(str2))
    
    def get_recommendations_by_genre(self, genre_query, n=5, min_votes=50):
        """Get recommendations based on genre keywords."""
        try:
            # Identify which genre keywords match the query
            matching_genres = []
            for genre, keywords in self.genre_keywords.items():
                for keyword in keywords:
                    if keyword in genre_query.lower():
                        matching_genres.append(genre)
                        break
            
            if not matching_genres:
                return "Could not identify a specific genre from your query."
            
            print(f"Identified genres: {matching_genres}")
            
            # Filter movies with minimum votes
            # First ensure vote_count is numeric
            self.df['vote_count'] = pd.to_numeric(self.df['vote_count'], errors='coerce').fillna(0)
            filtered_df = self.df[self.df['vote_count'] >= min_votes]
            
            if filtered_df.empty:
                print(f"No movies found with at least {min_votes} votes. Reducing threshold.")
                # If no movies with the minimum votes, try with fewer votes
                min_votes = 0
                filtered_df = self.df
            
            # Create a search query from the genre keywords
            search_query = ' '.join([' '.join(self.genre_keywords[genre]) for genre in matching_genres])
            
            # Transform the search query using the fitted TF-IDF vectorizer
            query_vector = self.tfidf.transform([self.clean_text(search_query)])
            
            # Compute cosine similarity between the query and all movies
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Create a dataframe with similarity scores
            sim_scores = list(enumerate(cosine_similarities))
            
            # Filter to only include movies from the filtered dataframe
            sim_scores = [(i, score) for i, score in sim_scores if i in filtered_df.index]
            
            # Sort by similarity score
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top n recommendations
            top_n = sim_scores[:n]
            
            if not top_n:
                return f"No recommendations found for '{genre_query}'."
                
            # Get the indices of the top recommendations
            movie_indices = [i[0] for i in top_n]
            
            # Return the top n movies
            recommended_movies = self.df.iloc[movie_indices][['original_title', 'overview', 'vote_average', 'vote_count']]
            
            if recommended_movies.empty:
                return f"No matching movies found for '{genre_query}'."
                
            return recommended_movies
        except Exception as e:
            print(f"Error in get_recommendations_by_genre: {e}")
            return f"Error finding recommendations for '{genre_query}': {str(e)}"
    
    def format_recommendation(self, movie):
        """Format a movie recommendation as a string."""
        try:
            title = movie['original_title']
            overview = movie['overview']
            rating = movie['vote_average']
            votes = movie['vote_count']
            
            sentiment = self.get_sentiment_description(rating)
            
            formatted = f"Title: {title}\n"
            formatted += f"Rating: {rating}/10 ({votes} votes)\n"
            formatted += f"What people say: {sentiment}\n"
            formatted += f"Overview: {overview}\n"
            
            return formatted
        except Exception as e:
            print(f"Error in format_recommendation: {e}")
            return "Error formatting recommendation."
    
    def get_sentiment_description(self, rating):
        """Get a sentiment description based on the rating."""
        if rating >= 8.5:
            return "Universally acclaimed as a masterpiece by critics and audiences alike."
        elif rating >= 8.0:
            return "Highly praised by viewers, considered an excellent film."
        elif rating >= 7.5:
            return "Very well received, audiences generally love this movie."
        elif rating >= 7.0:
            return "Quite popular, most viewers enjoy this film."
        elif rating >= 6.5:
            return "Generally favorable reviews, though with some mixed opinions."
        elif rating >= 6.0:
            return "Mixed reviews, but more positive than negative."
        elif rating >= 5.5:
            return "Somewhat mixed reviews, appeal may vary."
        elif rating >= 5.0:
            return "Average ratings, audiences are divided on this one."
        else:
            return "Below average ratings, not widely recommended."
    
    def recommend(self, user_query):
        """Provide recommendations based on user query."""
        # Check if the query is about a specific movie
        title_match = re.search(r'like\s+(.*?)(?:\s+movie|\s+film|\s*$)', user_query, re.IGNORECASE)
        
        if title_match:
            # Get recommendations similar to the specified movie
            movie_title = title_match.group(1).strip()
            recommendations = self.get_recommendations_by_title(movie_title)
            
            if isinstance(recommendations, str):
                return recommendations
            
            results = []
            for _, movie in recommendations.iterrows():
                results.append(self.format_recommendation(movie))
            
            return f"Based on your interest in '{movie_title}', here are some recommendations:\n\n" + "\n\n".join(results)
        else:
            # Get recommendations based on genre
            recommendations = self.get_recommendations_by_genre(user_query)
            
            if isinstance(recommendations, str):
                return recommendations
            
            results = []
            for _, movie in recommendations.iterrows():
                results.append(self.format_recommendation(movie))
            
            return f"Based on your query '{user_query}', here are some recommendations:\n\n" + "\n\n".join(results)

# Example usage
if __name__ == "__main__":
    # Create a recommender instance
    recommender = MovieRecommender('top_rated_movies.csv')
    
    # Example queries
    queries = [
        "What is a funny movie I can watch?",
        "Recommend a movie like The Godfather",
        "I want to see a romantic comedy",
        "What's a good action movie?",
        "Suggest a movie with high ratings"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        print(recommender.recommend(query))
        print("\n" + "="*50 + "\n")