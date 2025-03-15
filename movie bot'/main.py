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
            
            # Initialize common query patterns
            self._initialize_query_patterns()
            
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
    
    def _initialize_query_patterns(self):
        """Initialize common query patterns for chatbot matching."""
        self.query_patterns = {
            r'(?:recommend|suggest)\s+(?:a|some)\s+(?:movie|film)(?:s)?\s+like\s+([\w\s]+)': self._handle_similar_movie,
            r'(?:what|any|some)\s+(?:good|great|nice)\s+([\w\s]+)\s+(?:movie|film)(?:s)?': self._handle_genre_query,
            r'(?:show|give|recommend)\s+(?:me)?\s+(?:a|some)?\s+([\w\s]+)\s+(?:movie|film)(?:s)?': self._handle_genre_query,
            r'(?:i\s+want\s+to\s+(?:watch|see))\s+(?:a|some)?\s+([\w\s]+)\s+(?:movie|film)(?:s)?': self._handle_genre_query,
            r'(?:i\'m|i am)\s+in\s+the\s+mood\s+for\s+(?:a|some)?\s+([\w\s]+)\s+(?:movie|film)(?:s)?': self._handle_genre_query,
            r'(?:best|top|highest\s+rated)\s+([\w\s]+)\s+(?:movie|film)(?:s)?': self._handle_top_rated_genre,
            r'(?:movie|film)(?:s)?\s+similar\s+to\s+([\w\s]+)': self._handle_similar_movie,
            r'help': self._handle_help,
            r'(?:how\s+to\s+use|instructions|commands)': self._handle_help,
            r'(?:exit|quit|bye|goodbye)': self._handle_exit
        }
        
        # More generic patterns to catch other types of queries
        self.fallback_patterns = {
            r'([\w\s]+)\s+(?:movie|film)(?:s)?': self._handle_generic_query,
            r'': self._handle_unknown_query  # Empty pattern as fallback
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
            # Convert indices to movie IDs and filter out the input movie
            rec_movie_indices = [self._index_to_movie_id(i[0]) for i in sim_scores]
            rec_movie_indices = [i for i in rec_movie_indices if i != movie_id]
            
            # Get the top n recommendations
            recommendations = self.df.loc[rec_movie_indices].head(n)
            
            # Format the recommendations
            result = f"Based on '{matched_movie['original_title']}', here are some recommendations:\n\n"
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                year = int(row['release_year']) if not pd.isna(row['release_year']) else 'Unknown'
                rating = round(row['vote_average'], 1) if not pd.isna(row['vote_average']) else 'N/A'
                result += f"{i}. {row['original_title']} ({year}) - Rating: {rating}/10\n"
            
            return result
            
        except Exception as e:
            print(f"Error in get_recommendations_by_title: {e}")
            return "Sorry, I couldn't generate recommendations based on that title. Please try another movie."
    
    def _string_similarity(self, str1, str2):
        """Calculate string similarity between 0 and 1."""
        # Simple fuzzy matching
        str1, str2 = str1.lower(), str2.lower()
        
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Check if one is contained in the other
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Compute word-level overlap
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0
            
        overlap = len(words1.intersection(words2))
        return overlap / max(len(words1), len(words2))
    
    def get_recommendations_by_genre(self, genre, n=5):
        """Get recommendations based on a genre keyword."""
        try:
            # Create a genre-specific query
            genre_terms = self.genre_keywords.get(genre.lower(), [genre.lower()])
            
            # Create a combined query text
            query_text = ' '.join(genre_terms)
            
            # Clean the query
            cleaned_query = self.clean_text(query_text)
            
            # Transform the query with our existing TF-IDF vectorizer
            query_vec = self.tfidf.transform([cleaned_query])
            
            # Calculate similarity with all movies
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get indices of movies sorted by similarity
            sim_indices = sim_scores.argsort()[::-1]
            
            # Convert matrix indices to movie IDs
            movie_indices = [self._index_to_movie_id(idx) for idx in sim_indices[:n+5]]
            
            # Filter out any None values (if any conversion failed)
            movie_indices = [idx for idx in movie_indices if idx is not None]
            
            # Get recommendations with a minimum threshold on vote count and rating
            recommendations = self.df.loc[movie_indices]
            recommendations = recommendations[recommendations['vote_count'] >= 20]
            recommendations = recommendations.sort_values('vote_average', ascending=False).head(n)
            
            # Format the recommendations
            result = f"Top {genre} movies you might enjoy:\n\n"
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                year = int(row['release_year']) if not pd.isna(row['release_year']) else 'Unknown'
                rating = round(row['vote_average'], 1) if not pd.isna(row['vote_average']) else 'N/A'
                result += f"{i}. {row['original_title']} ({year}) - Rating: {rating}/10\n"
            
            return result
            
        except Exception as e:
            print(f"Error in get_recommendations_by_genre: {e}")
            return f"Sorry, I couldn't find good {genre} movie recommendations. Please try another genre."
    
    def get_top_rated_movies(self, genre=None, n=5, min_votes=100):
        """Get top rated movies, optionally filtered by genre."""
        try:
            # Start with all movies that have sufficient votes
            qualified = self.df[self.df['vote_count'] >= min_votes].copy()
            
            if len(qualified) == 0:
                return "Not enough rated movies in the database to make recommendations."
            
            # If genre is specified, filter by genre
            if genre:
                # Create a genre-specific query
                genre_terms = self.genre_keywords.get(genre.lower(), [genre.lower()])
                query_text = ' '.join(genre_terms)
                
                # Clean and vectorize the query
                cleaned_query = self.clean_text(query_text)
                query_vec = self.tfidf.transform([cleaned_query])
                
                # Find movies similar to the genre query
                sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                
                # Get valid indices above a threshold
                valid_indices = [i for i, score in enumerate(sim_scores) if score > 0.1]
                valid_movie_ids = [self._index_to_movie_id(idx) for idx in valid_indices]
                
                # Filter the qualified movies by these IDs
                qualified = qualified[qualified.index.isin(valid_movie_ids)]
            
            # Sort by rating
            top_movies = qualified.sort_values('vote_average', ascending=False).head(n)
            
            # Format the results
            if genre:
                result = f"Top rated {genre} movies:\n\n"
            else:
                result = "Top rated movies across all genres:\n\n"
                
            for i, (_, row) in enumerate(top_movies.iterrows(), 1):
                year = int(row['release_year']) if not pd.isna(row['release_year']) else 'Unknown'
                rating = round(row['vote_average'], 1) if not pd.isna(row['vote_average']) else 'N/A'
                votes = int(row['vote_count']) if not pd.isna(row['vote_count']) else 0
                result += f"{i}. {row['original_title']} ({year}) - Rating: {rating}/10 ({votes} votes)\n"
            
            return result
            
        except Exception as e:
            print(f"Error in get_top_rated_movies: {e}")
            return "Sorry, I couldn't retrieve top rated movies at this time."
    
    def process_query(self, query):
        """Process a natural language query and return recommendations."""
        query = query.lower().strip()
        
        # Try to match the query against defined patterns
        for pattern, handler in self.query_patterns.items():
            match = re.search(pattern, query)
            if match:
                return handler(match)
        
        # If no match found, try fallback patterns
        for pattern, handler in self.fallback_patterns.items():
            match = re.search(pattern, query)
            if match:
                return handler(match)
        
        # Ultimate fallback
        return self._handle_unknown_query(None)
    
    def _handle_similar_movie(self, match):
        """Handle queries asking for movies similar to a specific title."""
        movie_title = match.group(1).strip()
        return self.get_recommendations_by_title(movie_title)
    
    def _handle_genre_query(self, match):
        """Handle queries asking for movies of a specific genre."""
        genre_term = match.group(1).strip()
        
        # Check if the genre term matches any of our genre keywords
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in genre_term for keyword in keywords):
                return self.get_recommendations_by_genre(genre)
        
        # If no specific genre matched, use the term directly
        return self.get_recommendations_by_genre(genre_term)
    
    def _handle_top_rated_genre(self, match):
        """Handle queries asking for top rated movies in a genre."""
        genre_term = match.group(1).strip()
        
        # Check if the genre term matches any of our genre keywords
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in genre_term for keyword in keywords):
                return self.get_top_rated_movies(genre=genre)
        
        # If no specific genre matched, use the term directly
        return self.get_top_rated_movies(genre=genre_term)
    
    def _handle_generic_query(self, match):
        """Handle generic movie-related queries."""
        query_term = match.group(1).strip()
        
        # Check if it might be a movie title
        if len(query_term.split()) <= 4:  # Likely a movie title if few words
            return self.get_recommendations_by_title(query_term)
        else:
            # Try to extract meaningful terms and treat as genre
            return self.get_recommendations_by_genre(query_term)
    
    def _handle_help(self, _):
        """Handle help requests."""
        return """
Movie Recommender Help:
- Ask for movies similar to one you like: "Recommend movies like Inception"
- Ask for movies in a specific genre: "Show me good action movies"
- Ask for top rated movies: "Best comedy movies"
- Just type a movie title to get similar recommendations

Try phrases like:
- "I want to watch a scary movie"
- "Recommend movies like The Matrix"
- "What are some good romantic films?"
- "Top rated sci-fi movies"
        """
    
    def _handle_exit(self, _):
        """Handle exit requests."""
        return "Goodbye! Come back when you need more movie recommendations."
    
    def _handle_unknown_query(self, _):
        """Handle queries that don't match any patterns."""
        return """
I'm not sure what kind of movie you're looking for. You can:
- Ask for movies similar to one you like
- Ask for movies in a specific genre
- Ask for top rated movies
- Type 'help' for more information
        """


# Example usage
if __name__ == "__main__":
    recommender = MovieRecommender(data_path)
    
    print("\nMovie Recommender Initialized!")
    print("Type your movie queries or 'exit' to quit.\n")
    
    while True:
        user_input = input("What kind of movie would you like to watch? ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the Movie Recommender. Goodbye!")
            break
            
        response = recommender.process_query(user_input)
        print("\n" + response + "\n")