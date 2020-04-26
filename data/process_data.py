import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ load csv file containing messages based on the file path provided by user
    
    Output - Dataframe containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='left', on='id')
    
    return df


def clean_data(df):
    """Cleaning the data by:
        - Dropping duplicates
        - Removing missing classes
        - clean category columne
        
    Output: Cleaned dataframe with split of category
    """

    categories = df.categories.str.split(';', expand=True)
    row = categories[:1]

    # get category names
    category_colnames = row.applymap(lambda s: s[:-2]).iloc[0, :].tolist()
    categories.columns = category_colnames

    # get only the last value in each value as an integer
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 

    # add the categories back to the original df
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # clean up the final data
    df.drop_duplicates(subset='message', inplace=True)
    df.dropna(subset=category_colnames, inplace=True)

    return df

    


def save_data(df, database_filename):
    """ Save resulting data into sqlite database"""
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')
    engine.dispose()


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()