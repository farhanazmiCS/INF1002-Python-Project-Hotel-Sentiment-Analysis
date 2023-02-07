import sys
import os
import pandas as pd

def read_file(file_name: str) -> pd.DataFrame:
    """ Reads the specified CSV file. Returns dataframe object. """
    data_frame = pd.read_csv(file_name)
    return data_frame

def reviews_no_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Remove duplicate entries, based on all columns. Returns dataframe object. """
    reviews = dataframe.drop_duplicates()
    return reviews

def classify_review(score: float or int) -> str:
    """ Based on the review score, returns 'Negative' if review is bad,
        otherwise 'Positive'.
    """
    if score < 5:
        return 'Negative'
    return 'Positive'

def double_score(score: int) -> int:
    """ For actual dataset, score is out of 5. Hence, to fit inside the model, 
        multiply the score by 2. 
    """
    return score * 2

def divide_score(score: int) -> int:
    """ Divides the score after splitting the dataset into positive and negative. """
    return score / 2

def clean_review_content(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ For Hotel_Reviews.csv: 

        1. Reviews with 'No Negative' in the Negative_Review column would have the value of Positive_Review inside reviewContent
        2. Reviews with 'No Positive' in the Positive_Review column would have the value of Negative_Review inside reviewContent

        Returns a DataFrame object.
    """
    dataframe.loc[(dataframe['Reviewer_Score'] >= 5), 'reviewContent'] = dataframe['Positive_Review']
    dataframe.loc[(dataframe['Reviewer_Score'] < 5), 'reviewContent'] = dataframe['Negative_Review']
    return dataframe

def filter_training_reviews(dataframe: pd.DataFrame, sentiment: str) -> pd.DataFrame:
    """ Removes any reviews that has a score between 4 and 9. Returns DataFrame object. """
    if sentiment == 'good':
        return dataframe[dataframe['Reviewer_Score'] == 10]
    elif sentiment == 'bad':
        return dataframe[dataframe['Reviewer_Score'] < 4]
    else:
        print('Invalid sentiment!')

def output_csv(dataframe: pd.DataFrame, filename: str, sentiment: str, path: str) -> None:
    if sentiment == 'good':
        dataframe = dataframe[dataframe['sentiment'] == 'Positive']
        dataframe.to_csv(f'{path}/{filename[:-4]}_good.csv')
        print(f'{filename[:-4]}_good.csv created successfully!')
    elif sentiment == 'bad':
        dataframe = dataframe[dataframe['sentiment'] == 'Negative']
        dataframe.to_csv(f'{path}/{filename[:-4]}_bad.csv')
        print(f'{filename[:-4]}_bad.csv created successfully!')

    else:
        print('Invalid sentiment!')

def all_func(dataframe: pd.DataFrame, filename: str, sentiment: str) -> None:
    """ Calls all prior functions on either cases (Training or actual data)
    """
    dataframe = reviews_no_duplicates(dataframe)
    if filename == 'Hotel_Reviews.csv':
        path = 'processed_training_and_test_data'
        dataframe = dataframe.copy()
        dataframe = clean_review_content(dataframe)
        dataframe['sentiment'] = dataframe['Reviewer_Score'].apply(classify_review)
        dataframe = filter_training_reviews(dataframe, sentiment)
        output_csv(dataframe, filename, sentiment, path)
    else:
        path = 'processed_tripadvisor_data'
        dataframe = dataframe.copy()
        dataframe['reviewRating'] = dataframe['reviewRating'].apply(double_score)
        dataframe['sentiment'] = dataframe['reviewRating'].apply(classify_review)
        dataframe['reviewRating'] = dataframe['reviewRating'].apply(divide_score)
        output_csv(dataframe, filename, sentiment, path)

if len(sys.argv) != 3:
    print('Usage: python3 clean_data.py [file_name].csv [bad/good]')
else:
    file_name = sys.argv[1]
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data'
    try:
        dataframe = read_file(f'{path}/{file_name}')
    except FileNotFoundError:
        print('Error! File not found.')
    else:
        sentiment = sys.argv[2]
        all_func(dataframe, file_name, sentiment)
