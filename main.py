'''
@author: Sergei Savin

Comments:
    - more control for the user over the parameters can be implemented (thresholds, plots, etc.)
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import norm, chi2_contingency, chi2


def clean_outliers(df, threshold = 1.5, method = 'IQR', side = 'right'):

    ''' Removes outliers based on IQR rule.
        Can be extended to other methods '''

    if method == 'IQR':
        Q1 = df.Characters_Translated.quantile(0.25)
        Q3 = df.Characters_Translated.quantile(0.75)
        IQR = Q3 - Q1

        if side == 'right':
            df_clean = df[df.Characters_Translated <= Q3 + threshold * IQR]
        elif side == 'left':
            df_clean = df[df.Characters_Translated >= Q1 - threshold * IQR]
        else:
            df_clean = df[(df.Characters_Translated >= Q1 - threshold * IQR) &
                          (df.Characters_Translated <= Q3 + threshold * IQR)]

    return df_clean


def load_data(path, keep_outliers = True):
    ''' Loads the .csv data, cleans from outliers (if False) and returns pandas data frame '''

    df = pd.read_csv(path)
    df.columns = ['Session_ID', 'Variant_ID', 'Conversion', 'Characters_Translated']

    if not keep_outliers:
        df = clean_outliers(df)

    return df


def df_obs_print(df):

    ''' Summarizes the data '''

    num_sessions = df.Conversion.value_counts()
    avg_char = df.groupby('Conversion').Characters_Translated.mean()
    std_char = df.groupby('Conversion').Characters_Translated.std()

    df_obs = {'Conversion': [0, 1], 'Num_Sessions': [num_sessions[0], num_sessions[1]],
                'Avg_Characters_Translated': [avg_char[0], avg_char[1]],
                'Std_Characters_Translated': [std_char[0], std_char[1]]}
    df_obs = pd.DataFrame(df_obs)
    print(df_obs)


def show_statistics(df):

    # Summarize the data
    df_A = df[df.Variant_ID == 0]
    df_B = df[df.Variant_ID == 1]
    print('Control group')
    df_obs_print(df_A)
    print('\n')
    print('Test group')
    df_obs_print(df_B)
    print('\n')

    # Descriptive statistics
    print('Descriptive statistics \n')
    print('converted')
    print(df[df.Conversion == 1].Characters_Translated.describe(), '\n')
    print('not converted')
    print(df[df.Conversion == 0].Characters_Translated.describe(), '\n')

    plt.ion() #unblocks plt.show()
    # Box plots w/o outliers
    fig1, axes = plt.subplots(1, 1, sharey=True, figsize=(7, 7))
    plt.suptitle('w/o outliers')
    sns.boxplot(x = 'Conversion', y='Characters_Translated', data=df, showfliers=False);

    # Box plots w/ outliers
    fig2, axes = plt.subplots(1, 1, sharey=True, figsize=(7, 7))
    plt.suptitle('w/ outliers')
    sns.boxplot(x='Conversion', y='Characters_Translated', data=df, showfliers=True);

    fig3, axes = plt.subplots(1, 1, sharey=True, figsize=(7, 4))
    plt.suptitle('Translated characters vs. conversion: version A')
    X = np.array([[x] for x in df_A.Characters_Translated.tolist()])
    y = np.array(df_A.Conversion.tolist())
    plt.scatter(X, y, color='black')

    fig4, axes = plt.subplots(1, 1, sharey=True, figsize=(7,4))
    plt.suptitle('Translated characters vs. conversion: version B')
    X = np.array([[x] for x in df_B.Characters_Translated.tolist()])
    y = np.array(df_B.Conversion.tolist())
    plt.scatter(X, y, color='black')

    plt.show()


def Z_test_two_sample(mean, std, n, version='onesided'):

    ''' Two-sample Z-test '''

    Z = (mean[1] - mean[0]) / np.sqrt(std[0] ** 2 / n[0] + std[1] ** 2 / n[1])
    alternative = ''
    if Z > 0:
        if version == 'onesided':
            alternative = 'right'
    else:
        Z=-Z
        if version == 'onesided':
            alternative = 'left'
    p_val = norm.sf(Z)

    if version == 'twosided':
        alternative = 'twosided'
        p_val = p_val * 2

    return p_val, Z, alternative


def test_conversion(df, plot=False):
    ''' Chi-square test for the null hypothesis: the conversion is independent on versions A/B.
        Returns the corresponding p-value and the test-score. '''

    # Summarize the data
    num_sessions = df.Variant_ID.value_counts()
    converted_obs = df.groupby('Variant_ID').Conversion.sum()

    df_obs = {'Version': [0, 1], 'Num_Sessions': [num_sessions[0], num_sessions[1]],
              'Converted': [converted_obs[0], converted_obs[1]]}
    df_obs = pd.DataFrame(df_obs)
    df_obs['Not_Converted'] = df_obs.Num_Sessions - df_obs.Converted

    # Perform the test
    cont_table_observed = np.array([df_obs.Converted.tolist(), df_obs.Not_Converted.tolist()]).T
    test_result = chi2_contingency(cont_table_observed, correction=False)
    test_score = test_result[0]
    p_val = test_result[1]

    if plot:
        x = np.arange(0, 5, 0.1)
        plt.plot(x, chi2.pdf(x, df=1))
        plt.fill_between(x[x > test_score], chi2.pdf(x[x > test_score], df=1))
        plt.show()

    return p_val, test_score


def test_characters_translated(df, version = 'onesided' , plot=False):

    ''' Significance test for the null hypothesis: the average number of translated characters is the same
        for versions 0 and 1. Returns the corresponding p-value, the test score and the alternative hypothesis. '''

    # Summarize the data
    num_sessions = df.Variant_ID.value_counts()
    avg_char = df.groupby('Variant_ID').Characters_Translated.mean()
    std_char = df.groupby('Variant_ID').Characters_Translated.std()

    df_obs = {'Version': [0, 1], 'Num_Sessions': [num_sessions[0], num_sessions[1]],
              'Avg_Characters_Translated': [avg_char[0], avg_char[1]],
              'Std_Characters_Translated': [std_char[0], std_char[1]]}
    df_obs = pd.DataFrame(df_obs)

    # Perform the test
    mean = df_obs.Avg_Characters_Translated.tolist()
    std = df_obs.Std_Characters_Translated.tolist()
    sample_size = df_obs.Num_Sessions.tolist()
    p_val, test_score, alternative = Z_test_two_sample(mean, std, sample_size, version = version)

    if plot:
        z = np.arange(-3, 3, 0.1)
        plt.plot(z, norm.pdf(z))
        plt.fill_between(z[z > test_score], norm.pdf(z[z > test_score]))
        plt.show()

    return  p_val, test_score, alternative


def correlation_conversion_translated_characters(df, group = -1, version = 'onesided', plot=False):
    ''' Z-test for the null hypothesis: Converted and non-converted users translate the same amount of characters on average
    group values are: -1 (both A and B versions), 0 (the control version), 1 (the test version).
    Returns the corresponding p-value, the test score and the alternative hypothesis. '''

    # Summarize the data
    if group == 0:
        df = df[df.Variant_ID == 0]
    elif group == 1:
        df = df[df.Variant_ID == 1]

    num_sessions = df.Conversion.value_counts()
    avg_char = df.groupby('Conversion').Characters_Translated.mean()
    std_char = df.groupby('Conversion').Characters_Translated.std()

    df_obs = {'Conversion': [0, 1], 'Num_Sessions': [num_sessions[0], num_sessions[1]],
                'Avg_Characters_Translated': [avg_char[0], avg_char[1]],
                'Std_Characters_Translated': [std_char[0], std_char[1]]}
    df_obs = pd.DataFrame(df_obs)

    mean = df_obs.Avg_Characters_Translated.tolist()
    std = df_obs.Std_Characters_Translated.tolist()
    sample_size = df_obs.Num_Sessions.tolist()

    p_val, test_score, alternative = Z_test_two_sample(mean, std, sample_size, version = version)

    if plot:
        z = np.arange(-3, 3, 0.1)
        plt.plot(z, norm.pdf(z))
        plt.fill_between(z[z > test_score], norm.pdf(z[z > test_score]))
        plt.show()

    return p_val, test_score, alternative


def main(path, keep_outliers):

    # Load the data
    df = load_data(path, keep_outliers = keep_outliers)

    print('\n')
    print('----------------------------------------------------------------------------------------------------------')
    print('     Hypothesis testing')
    print('----------------------------------------------------------------------------------------------------------')

    # Significance test for the conversion
    print('The conversion is independent on versions A/B')
    p_val, test_score = test_conversion(df, plot=False)
    print('The corresponding p-value is: ', p_val, '\n')

    # Significance test for the average number of translated characters
    print('The average number of translated characters is the same for the vesrions A/B')
    p_val, test_score , alternative = test_characters_translated(df, version = 'onesided', plot=False)
    print('The alternative hypothesis is', alternative, 'with the corresponding p-value: ', p_val, '\n')

    # Significance test for the conversion vs. average number of translated characters
    print('Converted and non-converted users translate the same amount of characters on average: the control group only')
    p_val, test_score, alternative = correlation_conversion_translated_characters(df, group = 0, version = 'onesided',
                                                                                  plot=False)
    print('The alternative hypothesis is', alternative, 'with the corresponding p-value: ', p_val, '\n')

    print('Converted and non-converted users translate the same amount of characters on average: the test group only')
    p_val, test_score, alternative = correlation_conversion_translated_characters(df, group=1, version='onesided',
                                                                                  plot=False)
    print('The alternative hypothesis is', alternative, 'with the corresponding p-value: ', p_val, '\n')


if __name__ == '__main__':
    # Load the data

    path = ''
    exist = os.path.exists(path)
    while not exist:
        path = input('Please, provide the path to the data: ')
        exist = os.path.exists(path)

    outliers = ''
    while outliers not in ['yes', 'no', 'idk']:
        outliers = input('Would you like to keep the outliers: idk (some statistics will be shown), yes, no: ')
    if outliers == 'yes':
        keep_outliers = 1
        main(path, keep_outliers)
    elif outliers == 'no':
        keep_outliers = 0
        main(path, keep_outliers)
    else:
        print('\n')
        print('Here is some statistics wh you to choose \n')
        show_statistics(load_data(path))

        stop = 0
        while stop != 1:
            stop = input('Stop: yes, no: ')
            if stop == 'yes':
                stop = 1
            else:
                stop = 0

        if stop:
            plt.close('all')

 # ./data/session_data.csv


