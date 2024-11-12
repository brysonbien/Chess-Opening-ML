#import
import streamlit as st
import pandas as pd

# ------ PART 1 ------
st.title('CS 4641 Project Proposal')
st.header('Team 5')
st.markdown(
"""
Bryson Bien, Noah Brown, Marcel Dunat, Harrison Speed, Patrisiya Rumyanteseva \\
Github Repository: https://github.com/brysonbien/Chess-Opening-ML 
"""
)

st.header('1. Introduction/Background')
st.subheader('Dataset')
url = "https://www.kaggle.com/datasets/arashnic/chess-opening-dataset"
st.markdown(
"""
Chess openings have been studied for their impact on game outcomes, influencing player strategy and decision-making. Previous research has explored the correlation between certain openings and win rates, with debates on their overall effect compared to player skill and color choice. The dataset contains over 20,000 chess games, with features like player elo ratings, color (white/black), opening name and code, and game result to analyze how early strategies and player characteristics affect match outcomes. 

Dataset: %s 
""" % url 
)
st.subheader('Literature Review')
st.markdown(
"""
Several studies have employed machine learning to analyze the outcomes of chess games based on various game features. I. Cheng was able to create a model which accurately predicted game outcomes based on openings used and elo differences between the players [1]. K. Raghav and L. Ahuja were able to create a model which could accurately classify games based on the opening patterns and player ratings [2]. Studies like these can help further the field of computer-chess agents - previous successes include DeepZero created by Google DeepMind, which was trained entirely using Reinforcement learning and was able to beat StockFish chess engine within four hours [3].
"""
)

st.header('2. Problem Definition')
st.subheader('Problem')
st.markdown(
"""
The goal of this project is to determine whether opening moves during the early stages in a chess match significantly influence the likelihood of winning or losing. The project seeks to use a combination of player-specific features and game-specific features to predict the outcome of a chess game, leveraging the player's rating, the player's color, and the type of opening move to analyze their impact on whether a player wins or loses.
"""
)
st.subheader('Motivation')
st.markdown(
"""
Understanding how opening moves, player ratings, and color choice impact the outcome of chess games can can guide players on which openings are more advantageous and help them improve their decision-making process during the planning and early stages of the game depending on various circumstances.
"""
)

st.header('3. Methods')
st.markdown(
"""
Preprocessing Methods 

1. **Data Cleaning:** Data points that are missing features or duplicates of other points might cause problems and should be removed in a data cleaning process. (Pandas dropna and drop_duplicates)
2. **Dimensionality Reduction:** Features that are irrelevant to the project goals or redundant, like last_played_date or opening_name, can be removed to allow certain ML algorithms to run more efficiently. (Numpy or Pandas functions)
3. **Feature Engineering:** Features like opening_name or ECO may need to be encoded with numerical values to work with certain models. (Scikit Learn LabelEncoder)

Machine Learning Methods/Models

1. **KMeans/GMM:** Different openings can be clustered to determine which openings might be placed in similar categories and whether these groups influence game outcomes. (Scikit Learn Kmeans or GaussianMixture)
2. **Logistic Regression:** Logistic regression can be used to predict game outcomes based on different features. It is ideal for binary classification problems like determining whether an outcome will be a win or loss. (Scikit Learn LogisticRegression)
3. **Random Forest:** Can also be used to predict the binary outcomes of chess games. Considering the scale and complexity of certain features, like moves_list, this model is ideal. (Scikit Learn RandomForestClassifier) 

"""
)

st.header('4. Potential Results and Discussion')
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
        {"Name": "Bryson Bien", "Contribution": "Introduction, Problem, Motivation, Github, Slides"},
        {"Name": "Marcel Dunat", "Contribution": "Streamlit, Methods, ML algorithms"},
        {"Name": "Harrison Speed", "Contribution": "Literary Review, Sources, Citations, Initial ideas"},
        {"Name": "Noah Brown", "Contribution": "Project Initial Ideas, finding dataset, Gantt chart, Contribution Table"},
        {"Name": "Patrisiya Rumyantseva", "Contribution": "Creating proposal video, initial ideas"},
    ]
)
st.table(df)