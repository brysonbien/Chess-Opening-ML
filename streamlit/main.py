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
    **Opening Move Names and Other Features:** By using models like a Random Forest Classifier, we can make predictions about the binary outcomes of chess games. This will help evaluate how well opening names and related attributes contribute to predicting game outcomes.

    Further, Grid Search Cross Validation can be used to find the best hyperparameters for the Random Forest Classifier, which might produce predictions with higher accuracy.

    **Dataset Reduced with PCA:** Reduced features from PCA will be tested in the predictive model to assess if condensed data retains sufficient information for accurate predictions.
    """
)

st.header('3. Methods')
st.markdown("#### Preprocessing Methods")
df = pd.read_csv('data/games.csv')
df.head()
st.markdown("Manual Dimensionality Reduction: to remove features that definitely won't be helpful for prediction or won't be accessible before a game is over.")
df.drop(['id', 'turns', 'victory_status', 'created_at', 'last_move_at', 'white_id', 'black_id'], axis=1, inplace=True)
df.head()
st.markdown("Modifying 'opening_name' so only the primary opening strategy is included.")
df['opening_name'] = df['opening_name'].str.split(':').str[0]
st.markdown("#### Winner Column Analysis")
st.write(f"**Unique Winner Values (Before Filtering)**: {df['winner'].unique()}")

# Count and display number of "draw" entries
draw_count = (df['winner'] == 'draw').sum()
st.write(f"**Number of Entries that are Draw**: {draw_count}")
df = df[df['winner'] != 'draw']
st.write(f"**Unique Winner Values (After Filtering)**: {df['winner'].unique()}")

st.markdown("#### Feature Encoding: Encode non-numerical features")
increment_code_count = df['increment_code'].nunique()
opening_eco_count = df['opening_eco'].nunique()
opening_name_count = df['opening_name'].nunique()
moves_count = df['moves'].nunique()

# Display counts of unique values
st.markdown("#### Unique Value Counts for Feature Encoding")
st.write(f"**increment_code**: {increment_code_count}")
st.write(f"**opening_eco**: {opening_eco_count}")
st.write(f"**opening_name**: {opening_name_count}")
st.write(f"**moves**: {moves_count}")

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
st.markdown(
"""
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

st.subheader("Dimensionality Reduction with PCA:")
features = df.drop(['winner'], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features_scaled)

df_pca = pd.DataFrame(features_pca, columns=[f'PC{i+1}' for i in range(features_pca.shape[1])])
df_pca['winner'] = df['winner'].values
st.markdown("#### PCA-Reduced Data Preview")
st.write(df_pca.head())  # Display the first few rows of the PCA-reduced dataframe

# Display the shapes
st.write(f"**Original Shape**: {df.shape}")
st.write(f"**PCA-Reduced Shape**: {df_pca.shape}")

explained_variance = pca.explained_variance_ratio_

# Display explained variance for each principal component
st.markdown("#### Explained Variance Ratio for Each Principal Component")
for i, variance in enumerate(explained_variance):
    st.write(f"**Principal Component {i+1}**: {variance:.2%}")
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
st.markdown("#### Cross-Validation Results")
X = df.drop(['winner'], axis=1)
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
kfolds = KFold(n_splits=10)
scores = cross_val_score(model, X_train, y_train, cv=kfolds)
print("Accuracies: " + str(scores))
print("Mean Accuracy: " + str(scores.mean()))
st.write(f"Accuracies: {scores}")
st.write(f"Mean Accuracy: {scores.mean()}")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

st.subheader('Fitting Model with PCA-Reduced Dataset') 
st.markdown("#### Cross-Validation Results")
X_pca = df_pca.drop('winner', axis=1)
y_pca = df_pca['winner']
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)
model_pca = RandomForestClassifier()

kfolds_pca = KFold(n_splits=10)
scores_pca = cross_val_score(model_pca, X_train_pca, y_train_pca, cv=kfolds)
print("Accuracies: " + str(scores_pca))
print("Mean Accuracy: " + str(scores_pca.mean()))
st.write(f"Accuracies: {scores_pca}")
st.write(f"Mean Accuracy: {scores_pca.mean()}")
clf_pca = RandomForestClassifier()
clf_pca.fit(X_train_pca, y_train_pca)


st.subheader("Fitting Model with Best Hyperparameters")
st.markdown("#### Grid Search Cross Validation")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],    
    'min_samples_leaf': [1, 2]            
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
st.write("**Best Hyperparameters:**", grid_search.best_params_)

X_grid = df.drop('winner', axis=1)
y_grid = df['winner']
X_train_grid, X_test_grid, y_train_grid, y_test_grid = train_test_split(X, y, test_size=0.2, random_state=42)
st.markdown('#### Cross Validation: Perform K-Fold cross validation to verify the generalizability of the predictions.')
model_grid = RandomForestClassifier(max_depth=20, max_features='log2', min_samples_leaf=1, min_samples_split=5, n_estimators=200)

kfolds_grid = KFold(n_splits=10)
scores_grid = cross_val_score(model_grid, X_train_grid, y_train_grid, cv=kfolds_grid)
print("Accuracies: " + str(scores_grid))
print("Mean Accuracy: " + str(scores_grid.mean()))
st.write(f"Accuracies: {scores_grid}")
st.write(f"Mean Accuracy: {scores_grid.mean()}")
clf_grid = RandomForestClassifier(max_depth=20, max_features='log2', min_samples_leaf=1, min_samples_split=5, n_estimators=200)
clf_grid.fit(X_train_grid, y_train_grid)


st.header('Evaluation Metrics')
st.subheader("Basic Random Forest Classifier")
train_accuracy = clf.score(X_train, y_train)
print("Train Score: " + str(train_accuracy))
test_accuracy = clf.score(X_test, y_test)
print("Test Score: " + str(test_accuracy))
st.write(f"Train Score: {train_accuracy}")
st.write(f"Test Score: {test_accuracy}")

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
st.write("#### Model Evaluation Metrics")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1 Score: {f1}")

st.write("Classification Report")
st.text(report)

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
st.write(f"Train Score: {train_accuracy_pca}")
st.write(f"Test Score: {test_accuracy_pca}")

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
st.write("#### Model Evaluation Metrics")
st.write(f"Precision: {precision_pca}")
st.write(f"Recall: {recall_pca}")
st.write(f"F1 Score: {f1_pca}")

st.write("Classification Report")
st.text(report)
st.subheader("Confusion Matrix PCA Data")

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
st.write(f"Train Score: {train_accuracy_grid}")
st.write(f"Test Score: {test_accuracy_best}")

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
st.write("#### Model Evaluation Metrics")
st.write(f"Precision: {precision_grid}")
st.write(f"Recall: {recalll_grid}")
st.write(f"F1 Score: {f1_grid}")

st.write("Classification Report")
st.text(report)

st.subheader("Confusion Matrix with Grid Search Hyperparams")
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




st.header('4. Results and Discussion')
st.subheader("Visualizations")
st.markdown(
"""
- Correlation Matrix: The correlation matrix shows the relationships among features. For instance, a high correlation between certain features (like moves and opening_name) helped us identify redundancy. By removing such features, we minimized noise and potential overfitting, allowing the model to focus on the most relevant information.
- PCA Visualization: After standardizing the dataset (excluding the "winner" column), Principal Component Analysis was applied to reduce the data to three principal components. These components capture different percentages of the variance, respectively, cumulatively explaining about 60% of the total variance. Even though a specific amount of variance is unexplained, this is a substantial portion. A 3D scatter plot was generated based on the first three principal components, with points color-coded by the winner class. The plot reveals how data points are spread within the principal component space, though there is noticeable overlap, suggesting limited separation between classes. Overall, PCA has provided a condensed representation of the data, but the overlap in the plot suggests that the first three components alone may not sufficiently capture the features required for clear class distinction.
- Confusion Matrices: The three confusion matrices reveal different performance characteristics of the models. The Standard Model, the PCA Model, and the Grid Search Hyperparameters Model. The PCA Model indicates that PCA alone does not significantly enhance class distinction, while the Grid Search Hyperparameters Model shows the most balanced improvement, suggesting a minor increase in accuracy over the other models. Overall, while grid search tuning provided the best balance, all models show significant overlap between classes, suggesting that additional feature engineering may be necessary to further improve performance.
- Feature Importance Bar Chart: The feature importance chart highlights which factors are most influential in the model’s predictions. The `white_rating` and `black_rating` features stand out with the highest importance scores, indicating that these player ratings are the strongest predictors in the model. In comparison, `opening_eco` and `increment_code` show moderate importance suggesting that they hold some predictive power but are less crucial than the player ratings. Additionally, `opening_name` and `opening_ply` contribute to the model with slightly lower importance values. Finally, the `rated` feature has the lowest importance, indicating it has minimal influence on the model’s outcomes.
"""
)
st.subheader("Quantitative Metrics")
st.markdown(
    """
- Cross-Validation Accuracy: For each model, the 10-fold cross-validation accuracy scores varied but were generally moderate, indicating that none of the configurations performed very well. The mean accuracy across folds for each model gave a summary of its generalization capability, with the Grid Search-tuned model showing a slight improvement over the Standard and PCA models. However, the accuracy for all models suggested that there was still significant room for improvement.


- Precision: Precision scores for each model showed that the Random Forest classifier was moderately effective in minimizing false positives, with the Grid Search-tuned model slightly outperforming the other models. The precision for each configuration suggested that while the models could make correct positive predictions, a significant number of false positives still occurred.


- Recall: Recall scores indicated how well each model captured true positives, with similar values across the Standard, PCA, and Grid Search models. The Grid Search-tuned model slightly improved recall, but the overall values suggested that all models missed a number of positive instances, resulting in a significant rate of false negatives. Therefore, despite tuning and dimensionality reduction, the models struggled to identify all instances of each class effectively.


- F1 Score: The F1 Score showed that none of the configurations achieved a high level of both precision and recall. Although the Grid Search model had a marginally higher F1 score, all models displayed only moderate performance. This suggested that the feature set and model structure might need further refinement.


- Classification Report: The classification reports, detailing precision, recall, and F1 score for each class, revealed that all models tended to treat the "black" and "white" classes with similar accuracy but struggled to achieve high scores in either. Furthermore, the reports emphasized the class overlap issue noted in the confusion matrices, indicating that the Random Forest models were not distinguishing between the classes as clearly as desired.  

"""
)
st.subheader("Anaysis of Models")
st.markdown(
    """
    The visualizations and metrics indicate that the Random Forest Classifier models perform moderately well but still exhibit class overlap. The confusion matrices for each model show a high rate of misclassifications, with both false positives and false negatives which suggests that the model struggles to distinguish effectively between the "black" and "white" classes.Even though the grid search-tuned model offers a slight improvement, its accuracy remains limited since class overlap is prevalent across all of the models.
    """
)
st.subheader("Next Steps")
st.markdown(
    """
    - Testing Additional Models/Hyperparameter Tuning: Experimenting with different algorithms may reveal new insights and potentially higher accuracy, especially if we implement hyperparameter tuning. Some of the results suggest that the models might be overfitting to the training data. By using Grid Search Cross Validation with more hyperparameters or exploring other cross validation methods for hyperparameter tuning, we might reduce this overfitting and improve model accuracy. Additionally, we plan on exploring additional ML models that might make better predictions from our data. Logistic regression might reveal linear patterns in our data, and could allow us to better understand which features have the most impact on predictions. Gradient boosting models like XGBoost might be less likely to overfit on our training data which could produce results with higher accuracy. 
    
    - Improving Data Preprocessing: Using additional preprocessing steps could enhance model depth and accuracy. We will explore feature engineering to see how adding new features might affect model performance. One area that hasn't been explored is creating new features for specific opening moves. While the group decided to discard the feature describing all the moves in the game for the current models, creating new features for a select number of moves might reveal additional patterns that could improve predictions. 
    
    - Increase Data: Adding more data points might help generalize predictions better and reduce the risk of overfitting on current data.
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
        {"Name": "Bryson Bien", "Contribution": "Preprocessing, Introduction, Problem Definition, Streamlit"},
        {"Name": "Marcel Dunat", "Contribution": "Preprocessing, ML Model, Problem Definition, Methods"},
        {"Name": "Harrison Speed", "Contribution": "ML Model, Visualizations, Evaluation Metrics"},
        {"Name": "Noah Brown", "Contribution": "Preprocessing, ML Model, Visualizations"},
        {"Name": "Patrisiya Rumyantseva", "Contribution": "Discussion, Evaluation Metrics, Streamlit"},
    ]
)
st.table(df)