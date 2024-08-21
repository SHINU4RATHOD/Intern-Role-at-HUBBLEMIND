import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.feature_selection import RFE #(recursive feature elimination) help use to select important feature for model building
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sqlalchemy import create_engine
import joblib
import pickle

# loading the dataset 
df = pd.read_csv('C:/Users/SHINU RATHOD/Desktop/internship assignment/03_intern_HUBBLEMIND LABS PRIVATE LIMITED/Dataset/Stock Market Dataset.csv')
df.head()
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
df.to_sql('flight_ad_bid', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from flight_ad_bid;'
df = pd.read_sql_query(sql, engine) 
df.sample()
df.info()  # there are 1243 rows and 38 columns and i see issue with data types of some column like Date and other columns as well
 
df.drop('Date', axis = 1, inplace = True)

 # 2. converting non-numeric column to numeric columns
# Remove commas from all columns in the DataFrame since in our dataset numeric column contain commas 
df = df.replace({',': ''}, regex=True)
df = df.apply(pd.to_numeric)
df.info()     
df.describe()
df.isnull().sum()
df.duplicated().sum()     # there is no duplicate value  present in dataset




# ############################# univariate analysis
# # 1. univariate analysis
# df.columns
# df['Amazon_Price'].unique()
# df['Amazon_Price'].value_counts()

# # Histogram # Visualize the distribution and frequency of the target variable
# plt.hist(df['Amazon_Price'], bins=37, color='skyblue', edgecolor='red')
# plt.title('Histogram of Amazon_Price')
# plt.xlabel('Amazon_Price')
# plt.ylabel('Frequency')
# plt.show()

# # Boxplot
# df.boxplot(column=['Amazon_Price'])
# plt.title('Boxplot')
# plt.show()


# # P-P plot(probability-probability plot)
# import scipy.stats as stats
# # Sort the data
# data_sorted = np.sort(df['Amazon_Price'])

# # Get the theoretical quantiles for a normal distribution
# probabilities = np.linspace(0, 1, len(data_sorted))
# theoretical_quantiles = stats.norm.ppf(probabilities)

# # Plot the P-P plot
# plt.plot(theoretical_quantiles, data_sorted, marker='o', linestyle='none')
# plt.plot([theoretical_quantiles[0], theoretical_quantiles[-1]], 
#          [data_sorted[0], data_sorted[-1]], color='r', linestyle='--')


# # Q-Q Plot (Quantile-Quantile Plot):
# stats.probplot(df['Amazon_Price'], dist="norm", plot=plt)
# plt.show()


# sns.distplot(df['Amazon_Price'])
# sns.kdeplot(df['Amazon_Price'], shade=True)
# sns.ecdfplot(df['Amazon_Price'])


# # Shapiro-Wilk Test:
# from scipy.stats import shapiro
# stat, p = shapiro(df['Amazon_Price'])
# print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))

# # 3. Skewness and Kurtosis:
# df['Amazon_Price'].skew()
# df['Amazon_Price'].kurt()


# ######################### bivariate analysi
# corrmat = df.corr()

# # Compute the correlation matrix
# sns.heatmap(corrmat, annot = True, cmap = "YlGnBu")  #camp = 'coolwarm'

# # Plot the heatmap for visualization
# plt.figure(figsize=(12, 8))
# sns.heatmap(corrmat, annot=True, cmap='coolwarm')
# plt.show()

# # Heatmap enhanced
# # Upper triangle of an array.
# mask = np.triu(np.ones_like(corrmat, dtype = bool))     ## Generate a mask to show values on only the bottom triangle
# # mask = np.tril(np.ones_like(corrmat, dtype=bool))     ## Generate a mask to show values on only the bottom triangle
# plt.figure(figsize=(12, 8))  # Adjust the width and height as needed
# sns.heatmap(corrmat, annot = True, mask = mask, vmin = -1, vmax = 1)
# plt.title('Correlation Coefficient Of Predictors')
# plt.show()


#  # Correlation with the target variable
# corr_with_target = df.corrwith(df['Amazon_Price'])
# plt.figure(figsize=(10, 6))
# corr_with_target.plot(kind='bar')
# plt.title('Correlation with Target Variable')
# plt.xlabel('Features')
# plt.ylabel('Correlation')
# plt.show()

# ################### multivariate analysis
# # Pairplot for visualizing relationships between features
# sns.pairplot(df)
# plt.title('Pairplot of All Features')
# plt.show()


# # playing with AutoEDA Lib to check data quality
# # 1) SweetViz
# import sweetviz as sv
# s = sv.analyze(df)
# s.show_html()

# # 3) D-Tale
# import dtale 
# d = dtale.show(df)
# d.open_browser()


# selecting the important features or columns baseed  on  analyzing correlation
corr_df = df.corr()
x = df[['Copper_Price', 'Bitcoin_Price', 'Ethereum_Price', 'S&P_500_Price', 'Nasdaq_100_Price', 'Apple_Price', 'Tesla_Price', 'Microsoft_Price', 'Silver_Price', 'Google_Price', 'Netflix_Price', 'Meta_Price', 'Gold_Price']]
corrmat_x = x.corr()


x.isnull().sum().sum()   # 0 total missing or null values present in dataset
############################
# Define pipeline for missing data if any
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'median'))])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, x.columns)])
x_imputation = preprocessor.fit(x)
joblib.dump(preprocessor, 'meanimpute')

imputed_df = pd.DataFrame(x_imputation.transform(x), columns = x.columns)
imputed_df
imputed_df.isnull().sum()

###################### playing with outliers
# Defining a function to count outliers present in dataset
def count_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    return outliers
# Counting outliers before applying Winsorization tech
outliers_before = imputed_df.apply(count_outliers)
outliers_before      
outliers_before.sum()  # here 55 total num of outlier/extreame values are present in dataset after imputing missing val with mean val

# plotting boxplot for to check outliers
imputed_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (35,20)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()

############################## Define Winsorization pipeline
# Define the model with percentiles:# Default values # Right tail: 95th percentile # Left tail: 5th percentile
winsorizer_pipeline = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
X_winsorized = winsorizer_pipeline.fit(imputed_df)
joblib.dump(winsorizer_pipeline, 'winsor')  

# Transform Winsorized data back to DataFrame
X_winsorized_df = pd.DataFrame(X_winsorized.transform(imputed_df), columns=imputed_df.columns)


# Count outliers after Winsorization
outliers_after = X_winsorized_df.apply(count_outliers)
outliers_after

X_winsorized_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (35, 25)) 
plt.subplots_adjust(wspace = 0.75)  
plt.show()


############################ creating pipline for standard scaler
scale_pipeline = Pipeline([('scale', RobustScaler())])
X_scaled = scale_pipeline.fit(X_winsorized_df)
joblib.dump(scale_pipeline, 'scale')

X_scaled_df = pd.DataFrame(X_scaled.transform(X_winsorized_df), columns = X_winsorized_df.columns)
X_scaled_df

clean_data = X_scaled_df

# extracting independent and dependent var
x = clean_data.drop('Amazon_Price', axis = 1)
y = df['Amazon_Price']


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# import statsmodels.formula.api as smf
import statsmodels.api as sm
P = add_constant(x)
basemodel0 = sm.OLS(y, P).fit()
basemodel0.summary()       # p-values of coefficients found to be insignificant due to colinearity p values should below 0.05

vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
vif  # Identify the variale with highest colinearity using Variance Inflation factor (VIF)
# Variance Inflation Factor (VIF) Assumption: VIF > 10 = colinearity # VIF on clean Data
clean_data1 = x.drop(['Netflix_Price', 'Meta_Price'], axis = 1)

# Drop colinearity variable - variable 6
# clean_data1 = x.drop(['Ethereum_Price', 'Nasdaq_100_Price', 'Tesla_Price', 'Google_Price'], axis = 1)


 

P = add_constant(clean_data1)
basemode1 = sm.OLS(y, P).fit()
basemode1.summary()
vif1 = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
vif1
# clean_data2 = clean_data1.drop(['Microsoft_Price', 'Nasdaq_100_Price', 'S&P_500_Price'], axis = 1)
 

# =============================================================================
# # P = add_constant(clean_data2)
# # basemode2 = sm.OLS(y, P).fit()
# # basemode2.summary()
# # vif2 = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
# # vif2
# # clean_data3 = clean_data2.drop(['Apple_Price', 'Google_Price', 'Ethereum_Price'], axis = 1)
# 
# =============================================================================

# P = add_constant(clean_data2)
# basemode3 = sm.OLS(y, P).fit()
# basemode3.summary()
# vif3 = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
# vif3

# Tune the model by verifying for influential observations influence plot
fig, ax = plt.subplots(figsize=(28, 24))
sm.graphics.influence_plot(basemode1, ax = ax)
plt.show()


clean_data_new = clean_data1.drop(clean_data.index[[320, 364, 365, 366, 0, 556, 559, 555, 87]])
y_new = y.drop(y.index[[320, 364, 365, 366, 0, 556, 559, 555, 87]])

# clean_data1_new['y'] = y
# clean_data1_new.head()
# corrr = clean_data1_new.corr()

# Build model on dataset
P = add_constant(clean_data_new)
basemode1 = sm.OLS(y_new, P).fit()
basemode1.summary()



#python code for to av(added variable) plot 
# from statsmodels.graphics.regressionplots import plot_partregress
# # List of independent variables (excluding the constant)
# P.columns = P.columns.str.replace('&', '_and_').str.replace(' ', '_').str.replace('.', '').str.replace('-', '_')
# features = P.columns[1:]
# # Plot AV plots for all features in the dataset
# fig, axes = plt.subplots(len(features), 1, figsize=(10, len(features) * 2))

# # Iterate over each feature and create the AV plot
# for i, feature in enumerate(features):
#      sm.graphics.plot_ccpr(basemode1, exog_idx=i+1, ax=axes[i])
#      axes[i].set_title(f'Added Variable Plot for {feature}')

# # Adjust layout
# plt.tight_layout()
# plt.show()




# Splitting data into training and testing data set
from sklearn.metrics import r2_score
X_train, X_test, Y_train, Y_test = train_test_split(P, y_new, test_size = 0.2, random_state = 0) 
model = sm.OLS(Y_train, X_train).fit()
model.summary()

 # TRAINING DATA EVALUATION  predicting on the train dataset
ytrain_pred = model.predict(X_train)
r_squared_train = r2_score(Y_train, ytrain_pred)  # # Calculating R-squared for the training data
r_squared_train  #0.9235677444595755
# Calculating residuals for the training data
train_resid = Y_train - ytrain_pred
train_rmse = np.sqrt(np.mean(train_resid**2))    # # Calculating RMSE for the training data
train_rmse

# ---- TEST DATA EVALUATION ---- #
# Predicting on the test data
ytest_pred = model.predict(X_test)
r_squared_test = r2_score(Y_test, ytest_pred)
r_squared_test  #0.928097583134017

print('MAE:', metrics.mean_absolute_error(Y_test, ytest_pred))
print('MSE:', metrics.mean_squared_error(Y_test, ytest_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, ytest_pred)))

sns.distplot(Y_test-ytest_pred)
plt.scatter(Y_test,ytest_pred)



######################### hyperparameter tuning with with Cross Validation (cv)
from sklearn.linear_model import LinearRegression
# k-fold CV (using all variables)
lm = LinearRegression()
lm = lm.fit(X_train, Y_train)
pickle.dump(lm, 'bestmodel')
## Scores with KFold
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(lm, X_train, Y_train, scoring = 'r2', cv = folds)
scores.mean()
scores.std()

print("Coefficient of determination R^2 <-- on train set: {}".format(lm.score(X_train, Y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(lm.score(X_test, Y_test)))

y_pred = lm.predict(X_test)
r_squared_test = r2_score(Y_test, y_pred)
print('Test R2 values: ', r_squared_test)  #0.928097583134017

print('MAE:', metrics.mean_absolute_error(Y_test, y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

# saving the best model in pickle file formate for feature use 
pickle.dump(lm, open('bestmodel.pkl', 'wb'))

coefficients = lm.coef_
intercepts = lm.intercept_

# Task 3: Feature Importance Analysis
# ● Analyse the coefficients of the linear regression model to determine featureimportance.
# ● Visualise the feature importances using a bar plot
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': coefficients
})
# Sort the DataFrame by the absolute value of the coefficients
feature_importance['Absolute Importance'] = feature_importance['Importance'].abs()
feature_importance = feature_importance.sort_values(by='Absolute Importance', ascending=False)



# Plotting the feature importances
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.gca().invert_yaxis()  # To display the most important features at the top
plt.show()





################################## Model testing on New sample dataset
df = pd.read_csv('C:/Users/SHINU RATHOD/Desktop/internship assignment/03_intern_HUBBLEMIND LABS PRIVATE LIMITED/Dataset/Stock Market Dataset.csv')
x = df[['Copper_Price', 'Bitcoin_Price', 'Ethereum_Price', 'S&P_500_Price', 'Nasdaq_100_Price', 'Apple_Price', 'Tesla_Price', 'Microsoft_Price', 'Silver_Price', 'Google_Price', 'Netflix_Price', 'Meta_Price', 'Gold_Price']]


model1 = pickle.load(open('bestmodel.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
StandScal = joblib.load('scale')
# encoding = joblib.load('encoding_pipeline')

clean = pd.DataFrame(impute.transform(x), columns = x)
clean1 = pd.DataFrame(winsor.transform(clean),columns = clean.columns)
clean2 = pd.DataFrame(StandScal.transform(clean1),columns = clean1.columns)
# clean3 = pd.DataFrame(encoding.transform(data1), columns = encoding.get_feature_names_out(input_features = data1.columns))

# clean_data = pd.concat([clean2, clean3], axis = 1)
# clean_data.info()

# required_features = model1.get_feature_names_out()
# Reorder the columns in `clean_data` to match the order in `required_features`
# clean_data = clean_data[required_features]
# Add the missing 'const' column and its in first place of dataset order is important while fitting the model 'const' column was at first position 
clean_data.insert(0, 'const', 1)
prediction = pd.DataFrame(model1.predict(clean_data), columns=['pred_Est_Spend_(USD)'])
prediction



 

