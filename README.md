# WNS_ML
WNS Machine Learning # WNS Machine Learning Challenge

WNS Machine Learning Challenge
The hackathon was for the purpose of carring out an analysis within the HR area. Training set of employee data with below data
department region education gender recruitment_channel no_of_trainings age previous_year_rating length_of_service KPIs_met >80% awards_won? avg_training_score
The Algorithm should predict is_promoted variable.

This step is data preparation - Varible creation We observed a null values for Education and Previous_Year_rating. We need to handle them
When we drew a chart for Education and found that Bachelor degree is mean for all the department
Similar chart showed mean rating was 3. Hence we did null handelling 
Next up, we observed the varibles department and region were categorical variable. So we wanted to create dummy variables.
We also created addional variable like what is the % of training an employee has completed as compared to regional average or compared to department

On comparing the is_promoted column. We observed the data set is biased. We need to handle this. We have two ways of doing it
Upsample minority class 
During upsampling random data sets are generated to closer to the minority class. We can specify, number of random rows to be generated. This way we can make the set as balanced. 
Downsample majority class
During downsampling random sets of data are selected from majority class. This makes the set as balanced.
I preferred the second option. There are many algorithms to do this, like SMOTE, sklearn.resample pacakge.


Next we have the cleaned Training and test data set 
We now start applying the alogrithms
First algorithm We try the Logistic regression.
we have split the training data into train/test
We can evaluate the accuracy.

Next algorithm We try the Random Forest Classification.

Next We try the Light GBM tree based model.
we have split the training data into train/test
We can evaluate the accuracy.
