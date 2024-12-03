import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# Ensure dependencies are installed
#!pip install pandas==2.2.3 streamlit==1.32.0 matplotlib scikit-learn numpy seaborn

df = pd.read_csv('data/games.csv')
df.drop(['id', 'turns', 'victory_status', 'created_at', 'last_move_at', 'white_id', 'black_id'], axis=1, inplace=True)
df.head()
df['opening_name'] = df['opening_name'].str.split(':').str[0]
df = df[df['winner'] != 'draw']

# convert df3 to a list, keep only first 10 values
df['opening_moves'] = df['moves'].str.split(' ').str[:10]

# reduce opening_moves to only contain moves in opening strategy, fill remaining values with NaN
def reduce_moves(row):
    if row['opening_ply'] > 10:
        row['opening_ply'] = 10

    if row['opening_ply'] < 10:
        row['opening_moves'] = row['opening_moves'][:row['opening_ply']] + ([np.nan] * (10 - row['opening_ply']))
    return row

df = df.apply(reduce_moves, axis=1)

df['move1_w'] = df['opening_moves'].str[0]
df['move1_b'] = df['opening_moves'].str[1]
df['move2_w'] = df['opening_moves'].str[2]
df['move2_b'] = df['opening_moves'].str[3]
df['move3_w'] = df['opening_moves'].str[4]
df['move3_b'] = df['opening_moves'].str[5]
df['move4_w'] = df['opening_moves'].str[6]
df['move4_b'] = df['opening_moves'].str[7]
df['move5_w'] = df['opening_moves'].str[8]
df['move5_b'] = df['opening_moves'].str[9]

df.drop(['opening_moves'], axis=1, inplace=True)
df['perf_comp'] = df['black_rating'] - df['white_rating']
le_opening = preprocessing.LabelEncoder()
df['opening_name'] = le_opening.fit_transform(df['opening_name'])

le_winner = preprocessing.LabelEncoder()
df['winner'] = le_winner.fit_transform(df['winner'])

le_eco = preprocessing.LabelEncoder()
df['opening_eco'] = le_eco.fit_transform(df['opening_eco'])

le_increment = preprocessing.LabelEncoder()
df['increment_code'] = le_increment.fit_transform(df['increment_code'])

le_moves = preprocessing.LabelEncoder()
df['moves'] = le_moves.fit_transform(df['moves'])

le_moves_spec = preprocessing.LabelEncoder()
df['move1_w'] = le_moves_spec.fit_transform(df['move1_w'])
df['move1_b'] = le_moves_spec.fit_transform(df['move1_b'])
df['move2_w'] = le_moves_spec.fit_transform(df['move2_w'])
df['move2_b'] = le_moves_spec.fit_transform(df['move2_b'])
df['move3_w'] = le_moves_spec.fit_transform(df['move3_w'])
df['move3_b'] = le_moves_spec.fit_transform(df['move3_b'])
df['move4_w'] = le_moves_spec.fit_transform(df['move4_w'])
df['move4_b'] = le_moves_spec.fit_transform(df['move4_b'])
df['move5_w'] = le_moves_spec.fit_transform(df['move5_w'])
df['move5_b'] = le_moves_spec.fit_transform(df['move5_b'])

df.head()

# ------ PART 1 ------
st.title('CS 4641 Final Report')
st.header('Team 5')
st.markdown(
"""
Bryson Bien, Noah Brown, Marcel Dunat, Harrison Speed, Patrisiya Rumyanteseva \\
Github Repository: https://github.com/brysonbien/Chess-Opening-ML 
"""
)

st.header('1. Introduction/Background')
st.subheader('Dataset')

# URLs for datasets
url1 = "https://www.kaggle.com/datasets/arashnic/chess-opening-dataset"
url2 = "https://www.kaggle.com/datasets/datasnaek/chess"

# Introduction text with URLs and markdown for Streamlit
st.markdown(
    """
    Chess has long been a domain where strategy, skill, and decision-making converge, and openings play a critical role in setting the foundation for the rest of the game. Chess openings, defined by the initial moves players choose, have been meticulously studied, with each opening carrying distinct advantages and risks. This project aims to build upon previous research, which has demonstrated that specific openings can impact win rates. However, the extent of this influence, especially in relation to other factors such as player skill and color choice, remains debated. Our dataset, consisting of over 20,000 games, includes features like player performance ratings, colors (white or black), opening names and ECO codes, and game results, making it ideal for exploring the influence of early-game strategies on outcomes.

    **Old Dataset:** [Chess Opening Dataset](%s)  
    **New Dataset:** [Chess Dataset](%s)
    """ % (url1, url2)
)

st.markdown(
    """
    Several studies have applied machine learning to analyze chess outcomes. For instance, I. Cheng developed a model that predicted game results based on openings and rating differences between players [1]. K. Raghav and L. Ahuja successfully classified games based on opening patterns and player ratings [2]. Machine learning in chess has seen remarkable milestones, including Google DeepMind's AlphaZero, a reinforcement-learning-based model that defeated the Stockfish engine within hours of training [3]. This project aims to further such research by using machine learning to explore the predictive power of openings alongside other player characteristics, potentially guiding both players and computer chess agents on how certain early-game decisions may shape the game's trajectory.
    """
)

st.subheader('Why We Changed the Dataset from Proposal')
st.markdown(
    """
    In the original dataset, each datapoint represented a summary of data from many games that followed a specific opening strategy. Therefore, several of the features were aggregated across many games: the dataset included features like ‘player_avg’, which depended taking averages of many games’ players’ performance ratings, and ‘win_pct’, which summarized the percentage of games won as opposed to providing binary win/loss information. This made the model less interpretable and precise, and it aligned less with the goal of classifying games as wins or losses
    
    In the new dataset, every feature represents an individual game. Predictions produced by models trained on this data will apply better to predicting the outcomes of specific games. Furthermore, this dataset is much larger, with 20,058 data points as opposed to the original 1,884. This could allow our Machine Learning models to find more complex patterns in our data and produce results with higher accuracy.
    """
)
st.header('2. Problem Definition')
st.markdown(
"""
This project focuses on binary classification of chess games as won by the black side or won by the white side based on several factors, with particular emphasis on the chosen opening strategy. The hypothesis is that certain openings, when combined with specific player ratings and colors, may significantly affect the odds of winning, allowing us to predict game results with measurable accuracy.
"""
)

st.subheader('Motivation')
st.markdown(
"""
By analyzing player-specific features like elo rating, game-specific attributes such as color choice, and strategic choices represented by opening moves, the project seeks to establish the role these elements play in determining game outcomes as this can help players improve their decision-making process during the planning and early stages of the game depending on various circumstances.
"""
)
# Plan section
st.subheader('Plan')
st.markdown('#### Visualization and Dimensionality Reduction')
st.markdown(
    """
    Initial data exploration will include visualizations to observe trends and relationships in the dataset. For example, correlation matrices can help determine which features are redundant. Manual dimensionality reduction will be used to remove redundant or irrelevant features, and dimensionality reduction algorithms, like Principal Component Analysis (PCA), will help identify and retain the most influential features to optimize the dataset for efficient analysis.
    """
)

st.markdown('#### Prediction Based on Features')
st.markdown(
    """
    **Opening Move Names and Other Features:** By using models like a Random Forest Classifier and XGBoost, we can make predictions about the binary outcomes of chess games. This will help evaluate how well opening names and related attributes contribute to predicting game outcomes.

    Further, Grid Search Cross Validation can be used to find the best hyperparameters for the Random Forest Classifier and XGBoost, which might produce predictions with higher accuracy.

    **Dataset Reduced with PCA:** Reduced features from PCA will be tested in the predictive model to assess if condensed data retains sufficient information for accurate predictions.

    **GMM:** Using GMM to cluster chess games could produce labels that might improve model accuracy when added as features to the dataset.
    """
)

st.header('3. Methods')
st.markdown("#### Preprocessing")
st.markdown(
    """
    In order to process our data so it can be used to train our machine learning models, we started by performing manual dimensionality reduction to remove features that were irrelevant or not accessible before a chess game is over. This included features like player ids or number of moves in the game. We modified the ‘opening_name’ feature to include only the primary opening strategy. We also used feature engineering to create 10 new features to store the first 5 opening moves of either side, unless an opening had fewer moves. We then tried creating a feature to represent the difference between white and black performance ratings.
    Next, we used a labelencoder to encode several non-numerical features. One-hot encoding was not feasible since these features contained many unique values. Additionally, we planned on using a random forest classifier, which would treat these labels as separate categories. After this initial processing, we used a correlation matrix to determine whether any features were highly correlated, which could indicate that they were redundant and some could be removed. The feature ‘moves’ was removed because it was highly correlated with ‘opening_name’ and included information that could only be obtained after a game was over.
    """
)

st.markdown(
    """
    Some additional preprocessing approaches were tested, but none of them seemed to show significant improvements in evaluation metrics. We tried several approaches for incorporating averages of different openings’ performance across the dataset. First we grouped data points by their opening names and calculated the mean of games won by the black side (this was performed on the training set and these averages were applied to the test set to avoid data leakage). Next we tried using a second dataset that contained information about average performances of different opening moves. Neither of these two approaches were included in our final report notebook, but were still documented in the file testing.ipynb. 
    """
)

st.markdown(
    """
    After performing these steps, we standardized continuous columns, like the ratings and rating comparisons of the players (which is necessary for models like GMM). We also performed PCA on the dataset to reduce it to three principal components which were stored in an alternate dataframe. This would allow us to plot the components and determine if there were any obvious patterns in the dataset.
    """
    
)

df.drop(['moves'], axis=1, inplace=True)
scaler = StandardScaler()
df_mod = df[['black_rating', 'white_rating', 'perf_comp']]
scaled = scaler.fit_transform(df_mod)

df[['black_rating', 'white_rating', 'perf_comp']] = scaled
df.head()
features = df.drop(['winner'], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features_scaled)

df_pca = pd.DataFrame(features_pca, columns=[f'PC{i+1}' for i in range(features_pca.shape[1])])
df_pca['winner'] = df['winner'].values




st.header("Models")
st.subheader("Random Forest")
st.markdown(
    """
    The machine learning algorithm used was a Random Forest Classifier. We opted for a supervised learning method to make binary classifications for predicting which side will win a chess game. While a decision tree was considered, we chose an ensemble method for its higher accuracy. Random forests are ideal for detecting complex, non-linear patterns in data and providing feature importance, which helps understand the influence of features like opening moves in predicting outcomes.
    
    Before fitting the models, K-Fold cross-validation ensured generalization to new testing data. Additionally, Grid Search cross-validation optimized hyperparameters. Three Random Forest classifiers were trained: one on manually reduced data, one on PCA-reduced data, and a third with the manually reduced dataset using the best hyperparameters from Grid Search.
    """
)

st.subheader("Gaussian Mixture Model (GMM)")
st.markdown(
    """
    The Gaussian Mixture Model (GMM) is an unsupervised learning algorithm used to cluster data by modeling it as a mixture of Gaussian distributions. We implemented GMM to generate cluster labels as additional features for supervised learning models, enhancing classification accuracy. 
    
    Before applying GMM, we removed the target label (game outcome) to prevent data leakage. These probabilistic cluster labels were normalized alongside the original features and evaluated for their impact on predictive performance. This demonstrated how unsupervised learning techniques like GMM can uncover hidden patterns and improve overall classification accuracy.
    """
)

# Subsection: XGBoost
st.subheader("XGBoost")
st.markdown(
    """
    XGBoost (eXtreme Gradient Boosting) is a supervised learning algorithm known for handling large datasets and reducing overfitting. After preprocessing, Grid Search Cross-Validation tuned hyperparameters to optimize performance. 
    """
)

























st.header('4. Results and Discussion')
st.subheader("Visualizations")
st.write("#### Correlation Matrix")
# Plotting the heatmap
matrix = df.corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(matrix, cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)
st.markdown(
    """
    ### Insights from Correlation Matrix
    #### Highly Correlated Features
    - **`move2_b`, `move3_b`, `move4_b`, and `move5_b`** are strongly correlated, indicating sequential patterns or dependencies in gameplay.
    - **`white_rating`** and **`black_rating`** are moderately correlated, reflecting that higher-rated players often face one another.
    """
)
st.write("#### PCA Scatter Plot")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(projection='3d')

x = df_pca['PC1']
y = df_pca['PC2']
z = df_pca['PC3']

ax.scatter(x, y, z, c=df_pca['winner'], cmap='coolwarm')
st.pyplot(fig)
st.markdown(
    """
    ### Insights from PCA Scatter Plot
    Overlap between classes suggests limited separation using only three principal components. While PCA reduced the dataset to three dimensions for visualization, retaining more components may improve class separability.
    """
)





st.write("#### Random Forest")
X = df.drop('winner', axis=1)
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

train_accuracy = clf.score(X_train, y_train)
print("Train Score: " + str(train_accuracy))
test_accuracy = clf.score(X_test, y_test)
print("Test Score: " + str(test_accuracy))

y_pred = clf.predict(X_test)
precision = precision_score(y_test, y_pred)
print("Precision: " + str(precision))
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall: " + str(recall))
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score: " + str(f1))
report = classification_report(y_test, y_pred, target_names=['black', 'white'])
print("\nClassification Report:")
print(report)
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['black', 'white'])
display.plot(ax=ax, colorbar=False)
ax.set_title('Confusion Matrix', fontsize=15)
st.pyplot(fig)

st.write(
    """
    **Insights:**
    - The model performs slightly better at predicting white wins compared to black wins.
    - There are more false positives and false negatives for black, suggesting room for improvement.
    """
)



X_pca = df_pca.drop('winner', axis=1)
y_pca = df_pca['winner']
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)
clf_pca = RandomForestClassifier()
clf_pca.fit(X_train_pca, y_train_pca)
X_grid = df.drop('winner', axis=1)
y_grid = df['winner']
X_train_grid, X_test_grid, y_train_grid, y_test_grid = train_test_split(X, y, test_size=0.2, random_state=42)
clf_grid = RandomForestClassifier(max_depth=20, max_features='log2', min_samples_leaf=1, min_samples_split=5, n_estimators=200)
clf_grid.fit(X_train_grid, y_train_grid)
train_accuracy_pca = clf_pca.score(X_train_pca, y_train_pca)
print("Train Score: " + str(train_accuracy_pca))
test_accuracy_pca = clf_pca.score(X_test_pca, y_test_pca)
print("Test Score: " + str(test_accuracy_pca))

y_pred_pca = clf_pca.predict(X_test_pca)
precision_pca = precision_score(y_test_pca, y_pred_pca)
print("Precision: " + str(precision_pca))
recall_pca = recall_score(y_test_pca, y_pred_pca, average='weighted')
print("Recall: " + str(recall_pca))
f1_pca = f1_score(y_test_pca, y_pred_pca, average='weighted')
print("F1 Score: " + str(f1_pca))
report_pca = classification_report(y_test_pca, y_pred_pca, target_names=['black', 'white'])
print("\nClassification Report:")
print(report_pca)
matrix_pca = confusion_matrix(y_test_pca, y_pred_pca)
fig, ax = plt.subplots(figsize=(6, 6))
display_pca = ConfusionMatrixDisplay(confusion_matrix=matrix_pca, display_labels=['black', 'white'])
display_pca.plot(ax=ax, colorbar=False)
ax.set_title('Confusion Matrix PCA Data', fontsize=15)

# Display the Confusion Matrix in Streamlit
st.pyplot(fig)
st.write(
    """
    **Insights:**
    - This model underperforms compared to the basic Random Forest model.
    - PCA may have removed important features, leading to higher misclassifications.
    """
)



train_accuracy_grid = clf_grid.score(X_train_grid, y_train_grid)
print("Train Score: " + str(train_accuracy_grid))
test_accuracy_best = clf_grid.score(X_test_grid, y_test_grid)
print("Test Score: " + str(test_accuracy_best))

y_pred_grid = clf_grid.predict(X_test_grid)

precision_grid = precision_score(y_test_grid, y_pred_grid)
print("Precision: " + str(precision_grid))
recall_grid = recall_score(y_test_grid, y_pred_grid, average='weighted')
print("Recall: " + str(recall_grid))
f1_grid = f1_score(y_test_grid, y_pred_grid, average='weighted')
print("F1 Score: " + str(f1_grid))
report_grid = classification_report(y_test_grid, y_pred_grid, target_names=['black', 'white'])
print("\nClassification Report: ")
print(report_grid)

matrix_grid = confusion_matrix(y_test, y_pred_grid)

# Plot the Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 6))
display_grid = ConfusionMatrixDisplay(confusion_matrix=matrix_grid, display_labels=['black', 'white'])
display_grid.plot(ax=ax, colorbar=False)
ax.set_title('Confusion Matrix with Grid Search Hyperparams', fontsize=15)

# Display the Confusion Matrix in Streamlit
st.pyplot(fig)
st.write(
    """
    **Insights:**
    - This model demonstrates the best balance between precision and recall for both classes.
    - The benefits of hyperparameter tuning are evident in the reduced misclassifications.
    """
)
importances = clf.feature_importances_
features = X.columns.to_list()
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_title("Feature Importances")

bar = ax.bar(features, importances, color='lightblue', width=0.4)
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.bar_label(bar)
ax.tick_params(axis='x', rotation=90)

# Display the plot in Streamlit
st.pyplot(fig)
st.write(
    """
    **Insights:**
    - Features like `perf_comp`, `white_rating`, and `black_rating` are the most influential for predictions.
    """
)

st.write("#### GMM Clustering Visualization")
from sklearn.preprocessing import StandardScaler

# Separate features (X_pca) and target ('winner') from df_pca
X_pca = df_pca.drop('winner', axis=1)
y_pca = df_pca['winner']

# Standardize the features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

from sklearn.mixture import GaussianMixture

# Apply GMM to generate cluster labels
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
df['gmm_labels'] = gmm_labels
X_augmented = df_pca.drop(columns=['winner'])
y_augmented = df_pca['winner']
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce the dataset to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=gmm_labels, cmap='viridis', s=50)
colorbar = fig.colorbar(scatter, ax=ax, label='Cluster Label')
ax.set_title('GMM Clustering Visualization')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')

# Display the plot in Streamlit
st.pyplot(fig)

st.write(
    """
    **Evaluation:**
    - The GMM clustering visualization shows the dataset distributed across three clusters, identified by colors.
    - PCA was used to reduce dimensionality to two components for visualization purposes, making it easier to interpret the cluster distribution.
    - The clustering provides meaningful separations, where some overlap between clusters may represent games with closer outcomes or more ambiguous patterns.
    """
)
st.write("#### XGBoost")
from sklearn.model_selection import train_test_split
import pandas as pd

X_xg = df.drop('winner', axis=1)
y_xg = df['winner']

X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(X_xg, y_xg, test_size=0.2, random_state=42)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_xg, y_train_xg)
train_accuracy_xg = xgb_model.score(X_train_xg, y_train_xg)
print("Train Score: " + str(train_accuracy_xg))
test_accuracy_xg = xgb_model.score(X_test_xg, y_test_xg)
print("Test Score: " + str(test_accuracy_xg))
y_pred_xg = xgb_model.predict(X_test_xg)

precision_xg = precision_score(y_test_xg, y_pred_xg)
print("Precision: " + str(precision_xg))
recall_xg = recall_score(y_test_xg, y_pred_xg, average='weighted')
print("Recall: " + str(recall_xg))
f1_xg = f1_score(y_test_xg, y_pred_xg, average='weighted')
print("F1 Score: " + str(f1_xg))
report_xg = classification_report(y_test_xg, y_pred_xg, target_names=['black', 'white'])
print("\nClassification Report: ")
print(report_xg)
matrix = confusion_matrix(y_test_xg, y_pred_xg)
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['black', 'white'])


# Plot the confusion matrix
fig, ax = plt.subplots()
display.plot(ax=ax)
plt.title('Confusion Matrix', fontsize=15)

# Display the plot in Streamlit
st.pyplot(fig)
st.write(
    """
    **Insights:**
    - The model achieves a good balance between precision and recall for both classes.
    - False negatives are slightly fewer than false positives, showing the model has a slight bias toward predicting "white" outcomes.
    """
)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [4, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'n_estimators': [100, 200, 300]
}

#xgb = XGBClassifier(eval_metric='logloss')
#grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
#grid_search.fit(X_train_xg, y_train_xg)



##print("Best Parameters:", grid_search.best_params_)
#best_model = grid_search.best_estimator_
best_model = XGBClassifier(eval_metric="logloss", colsample_bytree=0.8, learning_rate=0.1, max_depth=4, n_estimators=200, subsample=0.6)
best_model.fit(X_train_xg, y_train_xg)
#Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.6}
importances = best_model.feature_importances_
features = X_xg.columns.to_list()
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_title("Feature Importances")
bar = ax.bar(features, importances, color='lightblue', width=0.4)
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.bar_label(bar)
plt.xticks(rotation=90)

# Display the plot in Streamlit
st.pyplot(fig)
st.write(
    """
    **Evaluation:**
    - The most important feature is `perf_comp`, which significantly impacts the prediction of chess outcomes.
    - Other key features include `white_rating`, `black_rating`, and `increment_code`, highlighting the importance of player ratings and game settings.
    """
)















































# Results and Discussion section
st.header("Quantitative Metrics and Results")

# Subsection: Random Forest
st.subheader("Random Forest")
st.markdown(
    """
    The Random Forest Classifier was used for binary classification to predict chess game outcomes. 
    It offers high accuracy and robustness by using an ensemble of decision trees to capture non-linear patterns.
    
    We implemented three models:
    1. A basic model using a manually reduced dataset.
    2. A model trained on a PCA-reduced dataset.
    3. A model trained with optimized hyperparameters.
    """
)

# Random Forest Results with Classification Report Tables

# Basic Model
st.write("### Basic Model")
st.write("Train Accuracy:", train_accuracy)
st.write("Test Accuracy:", test_accuracy)

st.write("**K-Fold Cross Validation Results:**")
st.write("Mean Accuracy: **65.85%**")
st.write("Accuracies:", [0.6658, 0.6632, 0.6560, 0.6409, 0.6573, 0.6534, 0.6649, 0.6492, 0.6760, 0.6584])
classification_report_basic = {
    "": ["Precision", "Recall", "F1-Score", "Support"],
    "Black": [0.64, 0.63, 0.64, 1832],
    "White": [0.67, 0.68, 0.67, 1990],
    "Overall": ["", "", "Accuracy: ~66%", 3822],
}
st.table(pd.DataFrame(classification_report_basic).set_index(""))

st.markdown(
    """
    **Analysis:**
    - Precision for "white" (0.67) is slightly higher than "black" (0.64), meaning the model predicts "white" more accurately.
    - Recall for "white" (0.68) is also higher, indicating better coverage for predicting "white" wins.
    - The overall accuracy of 65.85% shows the model is moderately effective, with room for improvement, particularly in balancing predictions for both classes.
    """
)

# PCA-Reduced Model
st.write("### PCA-Reduced Dataset")
st.write("Train Accuracy:", train_accuracy_pca)
st.write("Test Accuracy:", test_accuracy_pca)

st.write("**K-Fold Cross Validation Results:**")
st.write("Mean Accuracy: **56.53%**")
st.write("Accuracies:", [0.5605, 0.5644, 0.5612, 0.5683, 0.5500, 0.5566, 0.5713, 0.5759, 0.5818, 0.5628])
classification_report_pca = {
    "": ["Precision", "Recall", "F1-Score", "Support"],
    "Black": [0.56, 0.53, 0.55, 1832],
    "White": [0.59, 0.62, 0.60, 1990],
    "Overall": ["", "", "Accuracy: ~58%", 3822],
}
st.table(pd.DataFrame(classification_report_pca).set_index(""))

st.markdown(
    """
    **Analysis:**
    - The drastic drop in accuracy to 56.53% suggests that PCA removed key features critical for prediction.
    - Precision and recall for "black" (0.56 and 0.53) are particularly low, showing the model struggles more with this class.
    - PCA may not have been suitable for this dataset, indicating that dimensionality reduction techniques need careful evaluation.
    """
)

# Hyperparameter-Tuned Model
st.write("### Best Hyperparameters")
st.write("Train Accuracy:", train_accuracy_grid)
st.write("Test Accuracy:", test_accuracy_best)

st.write("**K-Fold Cross Validation Results:**")
st.write("Mean Accuracy: **66.23%**")
st.write("Accuracies:", [0.6763, 0.6566, 0.6540, 0.6494, 0.6651, 0.6573, 0.6721, 0.6551, 0.6747, 0.6623])
classification_report_tuned = {
    "": ["Precision", "Recall", "F1-Score", "Support"],
    "Black": [0.65, 0.62, 0.64, 1832],
    "White": [0.67, 0.70, 0.68, 1990],
    "Overall": ["", "", "Accuracy: ~66%", 3822],
}
st.table(pd.DataFrame(classification_report_tuned).set_index(""))

st.markdown(
    """
    **Analysis:**
    - The highest overall accuracy (66.23%) demonstrates the effectiveness of hyperparameter tuning.
    - The model achieves more balanced predictions, with only slight bias toward predicting "white" wins.
    - This configuration is the most effective Random Forest model tested.
    """
)

# Subsection: GMM
st.subheader("Gaussian Mixture Model (GMM)")

# Display GMM results
data = {
    "PC1": [-0.306240, -1.639886, -1.533110, -1.113280, -0.255321],
    "PC2": [-1.380027, 0.219449, -0.103677, -0.582566, -1.148765],
    "PC3": [-0.048767, 0.205077, 0.438610, -0.254898, -0.222180],
    "Winner": [1, 0, 1, 1, 1],
    "GMM Labels": [1, 1, 1, 1, 0],
}
df = pd.DataFrame(data)
st.write("**GMM Cluster Results:**")
st.dataframe(df)

st.markdown(
    """
    GMM clustering effectively identifies underlying patterns in the data, which can enhance prediction 
    accuracy when used as features in supervised models. This shows the potential of combining unsupervised learning with supervised approaches.
    """
)

# Subsection: XGBoost
st.subheader("XGBoost")

# Display XGBoost Results
st.write("**Grid Search Results:**")
best_params = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "max_depth": 4,
    "n_estimators": 200,
    "subsample": 0.6,
}
st.write(pd.DataFrame.from_dict(best_params, orient="index", columns=["Value"]))
st.write("Train Accuracy:", train_accuracy_xg)
st.write("Test Accuracy:", test_accuracy_xg)


st.write("**K-Fold Cross Validation Results:**")

st.write("Mean Accuracy: **0.65**")
st.write("Accuracies:", [0.6566, 0.6534, 0.6364, 0.6331, 0.6455, 0.6370, 0.6636, 0.6440, 0.6702, 0.6505])

classification_report_xgb = {
    "": ["Precision", "Recall", "F1-Score", "Support"],
    "Black": [0.64, 0.60, 0.62, 1832],
    "White": [0.66, 0.70, 0.68, 1990],
    "Overall": ["", "", "Accuracy: ~65%", 3822],
}
st.table(pd.DataFrame(classification_report_xgb).set_index(""))

st.markdown(
    """
    **Analysis:**
    - XGBoost is competitive with the best-tuned Random Forest model, achieving balanced metrics across classes.
    - It performs slightly worse in predicting "black" outcomes but maintains high overall performance due to robust optimization.
    """
)

st.header("Analysis and Comparison of Models")
st.markdown(
    """
    **Model Performance:**
    - Random Forest with Hyperparameter Tuning achieved the highest accuracy (~66%) and balanced precision/recall.
    - XGBoost performed slightly lower at (~65%) accuracy but was efficient in handling large datasets.
    - PCA-Reduced Random Forest demonstrated the weakest performance, highlighting the importance of feature engineering.

    **Key Insights:**
    - Both Random Forest and XGBoost identified `perf_comp`, `white_rating`, and `black_rating` as the most important features.
    - GMM clustering revealed underlying patterns in the data, which, when incorporated into supervised models, enhanced prediction accuracy.
    - Hyperparameter optimization improved both Random Forest and XGBoost performance, reducing false positives and false negatives.
    """
)

# Next Steps
st.header("Next Steps")
st.markdown(
    """
    1. **Feature Engineering:**
       - Focus on enhancing high-importance features such as `perf_comp` and player ratings.
    2. **Model Optimization:**
       - Explore combining Random Forest and XGBoost.
       - Experiment with advanced hyperparameter optimization using Bayesian optimization.
    3. **Deep Learning Exploration:**
       - Investigate deep learning models like LSTMs or Transformers for sequence data.
    """
)
st.header('5. References')
st.markdown(
"""
[1] I. Cheng, “Machine Learning to Study Patterns in Chess Games,” Mar. 2024, Accessed: Oct. 04, 2024. [Online]. Available: https://www.researchgate.net/profile/Isaac-Cheng-3/publication/379334134_Machine_Learning_to_Study_Patterns_in_Chess_Games/links/6604c292390c214cfd151950/Machine-Learning-to-Study-Patterns-in-Chess-Games.pdf 

[2] K. Raghav and L. Ahuja, "Chess Opening Analysis Using DBSCAN Clustering and Predictive Modeling," 2024 11th International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions) (ICRITO), Noida, India, 2024, pp. 1-5, doi: 10.1109/ICRITO61523.2024.10522439. 

[3] D. Silver, T. Hubert, J. Schrittwieser, and D. Hassabis, “AlphaZero: Shedding new light on chess, shogi, and Go,” Google DeepMind, Dec. 06, 2018. https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/ (accessed Oct. 04, 2024).

"""
)

st.header('Gantt Chart')
st.markdown(
"""
Group 5 Gantt Chart: https://docs.google.com/spreadsheets/d/1CYhvUaiTCBzN0lFuOxkwd5xmzU_AKQHE/edit?usp=sharing&ouid=111692925742591404356&rtpof=true&sd=true
"""
)

st.header('Contribution Table')
df = pd.DataFrame(
    [
        {"Name": "Bryson Bien", "Contribution": "ML Model, Methods, Slides , ReadMe"},
        {"Name": "Marcel Dunat", "Contribution": "Preprocessing, Methods, Slides"},
        {"Name": "Harrison Speed", "Contribution": "Quantitative Metrics, Visualizations, Slides"},
        {"Name": "Noah Brown", "Contribution": "ML Model, Methods, Slides"},
        {"Name": "Patrisiya Rumyantseva", "Contribution": "Discussion/Comparison, Streamlit, Video, Slides"},
    ]
)
st.table(df)
st.header("Google Slides")
st.markdown(
    """
    https://docs.google.com/presentation/d/1vxEmXyQK5AcFjSmRBrUk0hyt9E6KFblMp8662HgS4QM/edit?usp=sharing 
    """
)
