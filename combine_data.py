import pandas as pd
import numpy as np
import glob
#'*scrape.csv'
def read_files(filenames):
    '''
    read in all scraped files
    '''
    file_list = [f for f in glob.glob(filenames)]
    files = {}
    for i,file in enumerate(file_list):
        files[file] = pd.read_csv(file)
    return files

def check_keywords(keywords, string_list):
    '''
    see if keywords are in a string - useful for data cleaning
    '''
    keyword_list = list(keywords.split(','))
    check = True
    for word in keyword_list:
        if word not in string_list:
            check = False
    return check

def clean_file(keywords, file):
    #make sure all keywords in file

    file['Job Title'] = file['Job Title'].apply(lambda x: str(x).lower())
    file['Job Title'] = file['Job Title'].apply(lambda x: x.replace(',', '').replace('(', '').replace(')', ''))
    file['Job Title'] = file['Job Title'].apply(lambda x: x.replace('/', ' ').split(' '))

    #filter out non keywords

    file['Job Title'] = file['Job Title'].apply(lambda x: x if check_keywords(keywords, x) == True else np.NaN)

    file = file.dropna()
    #remove word 'senior'
    file['Job Title'] = file['Job Title'].apply(lambda x: np.NaN if x[0] == 'senior' else x)
    file['keyword'] = keywords

    return file.dropna()

def main():
    files = read_files('Scraped_Data/*scrape.csv')

    df_list = []


    keywords = ['ux,designer', 'data,scientist', 'data,analyst', 'project,manager', 'product,manager',
     'account,manager', 'consultant','marketing', 'sales']
    for i,f in enumerate(files.keys()):
        df = clean_file(keywords[i], files[f])
        print(files[f])
        df_list.append(df)

    concated_df = pd.concat(df_list, ignore_index = True)
    concated_df.to_csv('jobs.csv')

main()
