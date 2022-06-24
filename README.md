# Case Study Data Scientist

## Note
This is a test assignment to one of the companies I applied to. I was offered a job position. 

## Data

We provide you some dummy data to provide an AB test analysis on. Even though the data is synthetically generated, it is representative of what we are dealing with in reality. However in this case study we anaylze only a tiny subset of metrics we are interested in and the amount of data provided is only a fraction of what we are actually working with.

We consider a hypothetical AB test setup which resulted in the data you find in [data/session_data.csv](./data).

The fields in that csv file are:

- *session_id*: unique identifier of a user session.
- *variant_id*: identifier of the test group the session was assigned to. Values are 0 and 1. A value of 0 means the session was assigned to the control group, i.e. the user was provided the status quo of our website, without the new feature added. A value of 1 means the session was assigned to the test group, i.e. the new feature was activated. It may be helpful to illustrate this with a naive example: If the feature corresponds to change of the color of a button on the website from blue to green, then all sessions with variant_id 0 will see a blue button, while all sessions with variant_id 1 see a green button.
- *conversion*: Identifies if a session resulted in a subscription by the user. 0 means no subscription, 1 means the user subscripbed within the session.
- *characters_translated*: Total number of characters the user has translated within the session.

## Your tasks

1. Based on the data described in the previous section, we ask you to provide a manual analysis of the data. Imagine the following szenario: We have run the AB test for a while and provide you with the data we collected so far in order to suggest an action point to your Data Science Colleagues. Is it helping our business and we should accept it (i.e. release it as the new default)? Should we reject it? Should we extend the testing period? 

    As mentioned above, our production code for AB testing is based on Python. Therefore we would like you to use Python for your solution of this task (preferably Python 3). All your code should be placed within this directory.
    In terms of presentation of results, we use Jupyter Notebook for our analyses since it also works great for providing a presentation of the results to techincal colleagues along with the code. If you prefer something else, that's also fine. What's important to us is that we can easily follow your thinking and conclusions and your code is well comprehensible. Please provide all the code that is required to perform your analysis.
    As a guideline: If you are comfortable with the topic of AB testing this task can be completed within one or two hours. 

    For Bonus Points: In the file [INSTRUCTIONS.md](./INSTRUCTIONS.md), please provide instructions for us in order to execute the code you provided in order to reproduce your analysis. If you like it's fine to base this on a tool like Anaconda, poetry, Docker, make, etc.. Main aspect is that it should be easy for us to reproduce your results.


2. So far we looked at 2 metrics. Which 3 other metrics would you include in the analysis to measure the impact of a change? Please give reasons for your choice. You can provide your answer in the file [task2.txt](./task2.txt).
