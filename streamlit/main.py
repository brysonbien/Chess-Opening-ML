import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('data/games.csv')
df.drop(['id', 'turns', 'victory_status', 'created_at', 'last_move_at', 'white_id', 'black_id'], axis=1, inplace=True)
df.head()
df['opening_name'] = df['opening_name'].str.split(':').str[0]
df = df[df['winner'] != 'draw']
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

df.head()

# ------ PART 1 ------
st.title('CS 4641 Midterm Checkpoint Report')
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
    In the original dataset, each datapoint represented a summary of data from many games that followed a specific opening strategy. Therefore, several of the features were aggregated across many games: the dataset included features like ‘player_avg’, which depended on taking averages of many games’ players’ performance ratings, and ‘win_pct’, which summarized the percentage of games won as opposed to providing binary win/loss information. This made the model less interpretable and precise, and it aligned less with the goal of classifying games as wins or losses (or wins/losses/draws). 

    In the new dataset, every feature represents an individual game. Predictions produced by models trained on this data will apply better to predicting the outcomes of specific games. Furthermore, this dataset is much larger, with 20,058 data points as opposed to the original 1,884. This could allow our Machine Learning models to find more complex patterns in our data and produce results with higher accuracy.
    """
)
st.subheader('Problem Definition')
st.markdown(
"""
This project focuses on binary classification of chess games as won by the black side or won by the white side (or: multiclass classification of chess games as won by the black side, won by the white side, or draws) based on several factors, with particular emphasis on the chosen opening strategy. The hypothesis is that certain openings, when combined with specific player ratings and colors, may significantly affect the odds of winning, allowing us to predict game results with measurable accuracy.
"""
)

st.subheader('Motivation')
st.markdown(
"""
By analyzing player-specific features like elo rating, game-specific attributes such as color choice, and strategic choices represented by opening moves, the project seeks to establish the role these elements play in determining game outcomes as this can help players improve their decision-making process during the planning and early stages of the game depending on various circumstances.
"""
)
# Plan section
st.header('Plan')
st.subheader('Visualization and Dimensionality Reduction')
st.markdown(
    """
    Initial data exploration will include visualizations to observe trends and relationships in the dataset. For example, correlation matrices can help determine which features are redundant. Manual dimensionality reduction will be used to remove redundant or irrelevant features, and dimensionality reduction algorithms, like Principal Component Analysis (PCA), will help identify and retain the most influential features to optimize the dataset for efficient analysis.
    """
)

st.subheader('Prediction Based on Features')
st.markdown(
    """
    **Opening Move Names and Other Features:** By using models like a Random Forest Classifier, we can make predictions about the binary outcomes of chess games. This will help evaluate how well opening names and related attributes contribute to predicting game outcomes.

    Further, Grid Search Cross Validation can be used to find the best hyperparameters for the Random Forest Classifier, which might produce predictions with higher accuracy.

    **Dataset Reduced with PCA:** Reduced features from PCA will be tested in the predictive model to assess if condensed data retains sufficient information for accurate predictions.
    """
)

st.header('Methods')
st.markdown(
"""
Preprocessing Methods 

In order to process our data so it can be used to train our machine learning models, we started by performing manual dimensionality reduction to remove features that were irrelevant or not accessible before a chess game is over. This included features like player ids or number of moves in the game. We also modified the ‘opening_name’ feature to include only the primary opening strategy. Next, we used a labelencoder to encode several non-numerical features. One-hot encoding was not feasible since these features contained many unique values. Additionally, we planned on using a random forest classifier, which would treat these labels as separate categories. After this initial processing, we used a correlation matrix to determine whether any features were highly correlated, which could indicate that they were redundant and some could be removed. The feature ‘moves’ was removed because it was highly correlated with ‘opening_name’ and included information that could only be obtained after a game was over.

### Correlation Matrix
"""
)
matrix = df.corr()

plt.figure(figsize=(9,7))
plt.title("Correlation Matrix")
sns.heatmap(matrix, cmap="coolwarm", annot=False)
st.pyplot(plt)



st.markdown(
"""
After performing these steps, we performed PCA on the dataset to reduce it to three principal components which were stored in an alternate dataframe. This would allow us to plot the components and determine if there were any obvious patterns in the dataset. 
"""
)
df.drop(['moves'], axis=1, inplace=True)
features = df.drop(['winner'], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features_scaled)

df_pca = pd.DataFrame(features_pca, columns=[f'PC{i+1}' for i in range(features_pca.shape[1])])
df_pca['winner'] = df['winner'].values
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(projection='3d')

x = df_pca['PC1']
y = df_pca['PC2']
z = df_pca['PC3']

ax.scatter(x, y, z, c=df_pca['winner'], cmap='coolwarm')
st.pyplot(plt)

st.header("Machine Learning Models")
st.markdown(
    """
    The machine learning algorithm that was used was a random forest classifier. We decided to use a supervised learning method to make binary classifications for which side will win a chess game. While we considered using a decision tree, we decided to begin with an ensemble method due to its higher accuracy. Random forests were ideal for our problem because they can detect complex, non-linear patterns in the data. Furthermore, random forests can indicate feature importance, which is helpful for understanding which features had the most influence in the classifier’s predictions. This is helpful for addressing the extent to which features like opening moves can be used to predict outcomes.
    Before fitting our models, we used K-Fold cross validation to ensure that our model results would generalize well to new testing data. We also used Grid Search cross validation to determine the best hyperparameters for our random forest. In total, three random forest classifiers from sklearn.ensemble were trained and evaluated. One was fit on the manually reduced dataset, one used the PCA-reduced dataset, and a third one used the manually reduced data (because it had high accuracy), but also used the best hyperparameters found in grid search.

    """
)
st.subheader('Basic Model Fitting to Understand Performance')
X = df.drop(['winner'], axis=1)
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
kfolds = KFold(n_splits=10)
scores = cross_val_score(model, X_train, y_train, cv=kfolds)
print("Accuracies: " + str(scores))
print("Mean Accuracy: " + str(scores.mean()))
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

st.subheader('Fitting Model with PCA-Reduced Dataset') 
X_pca = df_pca.drop('winner', axis=1)
y_pca = df_pca['winner']
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)
st.subheader('Cross Validation: Perform K-Fold cross validation to verify the generalizability of the predictions.') 
model_pca = RandomForestClassifier()

kfolds_pca = KFold(n_splits=10)
scores_pca = cross_val_score(model_pca, X_train_pca, y_train_pca, cv=kfolds)
print("Accuracies: " + str(scores_pca))
print("Mean Accuracy: " + str(scores_pca.mean()))
clf_pca = RandomForestClassifier()
clf_pca.fit(X_train_pca, y_train_pca)


st.subheader("Fitting Model with Best Hyperparameters: Grid Search Cross Validation")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],    
    'min_samples_leaf': [1, 2]            
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
X_grid = df.drop('winner', axis=1)
y_grid = df['winner']
X_train_grid, X_test_grid, y_train_grid, y_test_grid = train_test_split(X, y, test_size=0.2, random_state=42)
st.subheader('Cross Validation: Perform K-Fold cross validation to verify the generalizability of the predictions.')
model_grid = RandomForestClassifier(max_depth=20, max_features='log2', min_samples_leaf=1, min_samples_split=5, n_estimators=200)

kfolds_grid = KFold(n_splits=10)
scores_grid = cross_val_score(model_grid, X_train_grid, y_train_grid, cv=kfolds_grid)
print("Accuracies: " + str(scores_grid))
print("Mean Accuracy: " + str(scores_grid.mean()))
clf_grid = RandomForestClassifier(max_depth=20, max_features='log2', min_samples_leaf=1, min_samples_split=5, n_estimators=200)
clf_grid.fit(X_train_grid, y_train_grid)


st.header('Evaluation Metrics')
st.subheader("Basic Random Forest Classifier")
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

st.subheader("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['black', 'white'])
display.plot()
plt.title('Confusion Matrix', fontsize=15)
st.pyplot(plt)

st.subheader("Random Forest (Fit on PCA-Reduced Data) Evaluation")
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
display_pca = ConfusionMatrixDisplay(confusion_matrix=matrix_pca, display_labels=['black', 'white'])
display_pca.plot()
plt.title('Confusion Matrix PCA Data', fontsize=15)
st.pyplot(plt)

st.subheader("Random Forest with Best Hyperparameters Evaluation")
train_accuracy_grid = clf_grid.score(X_train_grid, y_train_grid)
print("Train Score: " + str(train_accuracy_grid))
test_accuracy_best = clf_grid.score(X_test_grid, y_test_grid)
print("Test Score: " + str(test_accuracy_best))

y_pred_grid = clf_grid.predict(X_test_grid)
precision_grid = precision_score(y_test_grid, y_pred_grid)
print("Precision: " + str(precision_grid))
recalll_grid = recall_score(y_test_grid, y_pred_grid, average='weighted')
print("Recall: " + str(recall))
f1_grid = f1_score(y_test_grid, y_pred_grid, average='weighted')
print("F1 Score: " + str(f1_grid))
report_grid = classification_report(y_test_grid, y_pred_grid, target_names=['black', 'white'])
print("\nClassification Report: ")
print(report_grid)
matrix_grid = confusion_matrix(y_test_grid, y_pred_grid)
display_grid = ConfusionMatrixDisplay(confusion_matrix=matrix_grid, display_labels=['black', 'white'])
display_grid.plot()
plt.title('Confusion Matrix with Grid Search Hyperparams', fontsize=15)
st.pyplot(plt)

st.subheader('Evaluating Feature Importance')
importances = clf.feature_importances_
features = X.columns.to_list()

plt.figure(figsize=(9, 9))
plt.title("Feature Importances")

bar = plt.bar(features, importances, color ='lightblue', width = 0.4)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.bar_label(bar)
plt.xticks(rotation=90)
st.pyplot(plt)




st.header('Results and Discussion')
st.markdown(
"""
Quantitative Metrics 

1. **Accuracy:** Measure how accurately the models predict win/loss outcomes.
2. **Precision:** Evaluate how many of the predicted wins are actual wins.
3. **Recall:** Assess the model's ability to identify all winning games.

"""
)
st.subheader('Goals/Potential Results')
st.markdown(
"""

Project Goals

- Achieve high predictive accuracy while ensuring model interpretability.
- Evaluate the importance of different features (e.g., player rating, color, opening type) in predicting outcomes.

Expected Results

- Discover openings that have a higher probability of leading to a win.
- Identify patterns in player behavior that influence the outcome of the game.

"""
)

st.header('References')
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
        {"Name": "Bryson Bien", "Contribution": "Preprocessing, Introduction, Problem Definition, Streamlit"},
        {"Name": "Marcel Dunat", "Contribution": "Preprocessing, ML Model, Problem Definition, Methods"},
        {"Name": "Harrison Speed", "Contribution": "ML Model, Visualizations, Evaluation Metrics"},
        {"Name": "Noah Brown", "Contribution": "Preprocessing, ML Model, Visualizations"},
        {"Name": "Patrisiya Rumyantseva", "Contribution": "Discussion, Evaluation Metrics, Streamlit"},
    ]
)
st.table(df)