import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed  

def compute_similarity(input_file, output_file, user_threshold=1, n_jobs=-1):
    movie_user_matrix, movie_means = preprocess_data(input_file)
    
    movie_pairs = [(i, j) for i in range(len(movie_user_matrix)) for j in range(i + 1, len(movie_user_matrix))]

    results = Parallel(n_jobs=n_jobs)(delayed(process_pair)(movie_user_matrix[i], movie_user_matrix[j], movie_means[i], movie_means[j], i, j, user_threshold) for i, j in movie_pairs)
    
    results = [result for result in results if result is not None]
    
    result_df = pd.DataFrame(results, columns=['movie_a', 'movie_b', 'similarity', 'common_ratings'])
    
    print(result_df.head(20))       # printing 20 lines to show
    
    # results saved to csv
    result_df.to_csv(output_file, index=False)
    
    # create 20 lines output
    abridged_df = result_df.head(20)
    abridged_output_file = "output.csv"             # output file saved
    abridged_df.to_csv(abridged_output_file, index=False)
    print("Abridged DataFrame (first 20 lines):")
    print(abridged_df)

def preprocess_data(input_file):
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = pd.read_csv(input_file, sep='\t', names=cols, usecols=['user_id', 'movie_id', 'rating'])

    print("Loaded Data:")
    print(data.head(20))

    movie_user_matrix = data.pivot_table(index='movie_id', columns='user_id', values='rating').values

    movie_means = np.nanmean(movie_user_matrix, axis=1)

    return movie_user_matrix, movie_means

def process_pair(movie_a, movie_b, mean_a, mean_b, idx_a, idx_b, user_threshold):
    common_users = ~np.isnan(movie_a) & ~np.isnan(movie_b)
    num_common_users = np.sum(common_users)

    if num_common_users >= user_threshold:
        ratings_a = movie_a[common_users]
        ratings_b = movie_b[common_users]

        similarity = calculate_similarity(ratings_a, ratings_b, mean_a, mean_b)

        if similarity is not None:
            return (idx_a, idx_b, similarity, num_common_users)
    return None

def calculate_similarity(ratings_a, ratings_b, mean_a, mean_b):
    centered_a = ratings_a - mean_a
    centered_b = ratings_b - mean_b

    numerator = np.sum(centered_a * centered_b)
    denominator = np.sqrt(np.sum(centered_a ** 2)) * np.sqrt(np.sum(centered_b ** 2))

    if denominator == 0:
        return None

    return numerator / denominator

if __name__ == "__main__":
    input_file = "u.data"         #we can update the input file
    output_file = "output.csv"  

    t1 = time.time()

    compute_similarity(input_file, output_file, user_threshold=1, n_jobs=-1)
    
    print(f"Finished computing movie similarity in {time.time() - t1:.2f} seconds")  
