import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from tensorflow.keras.models import load_model

# avoid tensorflow inner loggings

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


st.title("Precision farming -- Forecasting 📈")
st.subheader("Agriculture development 🌾 (CLIMATE BASED)")


# empty spaces
st.write("")
st.write("")
# data source
img = 'download.png'
col1 , col2 = st.columns([13,8])
with col1 : 
    st.markdown(" **Data Source** : National Oceanic and Atmospheric Administration")
    st.markdown("""The National Oceanic and Atmospheric Administration is a US scientific and regulatory agency charged with forecasting weather, monitoring oceanic and atmospheric conditions, charting the seas, conducting deep-sea exploration, and managing fishing and protection of marine mammals and endangered species in the US exclusive economic zone. The agency is part of the United States Department of Commerce and is headquartered in Silver Spring, Maryland.""")
with col2 :
    st.image(img)

st.write("")

st.header('Problem Statement:')
st.subheader("""
**States:** Iowa, Illinois, Indiana, Nebraska, Ohio, Kansas, Missouri, and parts of Minnesota and South Dakota.
""")
st.write("""
This region is a major producer of corn, soybeans, and other grains. Climate monitoring is critical for crop management, yield forecasting, and assessing the impact of weather on agricultural productivity.  
""")

st.write("")
st.write("")
# model implementation

model = load_model('my_lstm_model.keras')

window_size = 21
n_future = st.number_input("Enter the number of future days to predict:", min_value=1, max_value=1825, value=30, step=1)
sc = pd.read_csv('p_data.csv')
sc_ = sc.columns
scaled_data_ = np.array(sc)
def future_pred(model, scaled_data_, window_size, n_future):
    pred = []
    current = scaled_data_[-window_size:, :]
    current = current.reshape((1, window_size, scaled_data_.shape[1]))
    
    for i in range(n_future):
        c_pred = model.predict(current)[0]
        pred.append(c_pred)
        current = np.append(current[:, 1:, :], [[c_pred]], axis=1)
    
    return np.array(pred)


# make future predictions
if st.button('predict'):
    f_pred = future_pred(model,scaled_data_,window_size,n_future)
    date_  = pd.date_range(start = '2024-09-09',periods=n_future)
    f_df = pd.DataFrame(f_pred, index = date_,columns = sc_ )

    st.write(f'Predictions for the next {n_future} days')
    st.dataframe(f_df)
    out_fig = px.line(f_df, x=f_df.index, y='TOBS', markers=True, title=f'future predictions of temp from your selected date ** 2024-09-09:00:00:00 to {f_df.index[-1]} ')

    out_fig_1 = px.line(f_df, x=f_df.index, y='SNOW', markers=True, title=f'future predictions of snow from your selected date ** 2024-09-09:00:00:00 to {f_df.index[-1]} ')

    st.plotly_chart(out_fig)

    st.plotly_chart(out_fig_1)
    



# dataframe oveview
st.header('Data overview : ')
df = pd.read_csv('3772522.csv')
st.dataframe(df)

# df details 
# st.subheader('Descriptive statistical analysis')
# st.write(df.describe())

# timeline
st.subheader("Project Goals")
st.markdown("""
**The primary goals of this project are**:
1. **Temperature Prediction**: Accurately forecast future temperature trends using historical data.
2. **Impact Analysis**: Analyze the relationship between temperature fluctuations and agricultural yield.
3. **Risk Mitigation**: Identify potential risks to crops due to temperature extremes and provide early warnings.
4. **Decision Support**: Aid in making informed decisions for crop selection, planting schedules, and resource allocation.
5. **Sustainability**: Promote sustainable agricultural practices by understanding climate impacts.
""")


df['DATE'] = pd.to_datetime(df['DATE'])

# Group data by year and calculate the mean TMAX for each year
df_yearly_max = df.groupby(df['DATE'].dt.year)['TMAX'].max().reset_index()
df_yearly_min = df.groupby(df['DATE'].dt.year)['TMIN'].min().reset_index()
df_yearly_temp = df.groupby(df['DATE'].dt.year)['TOBS'].mean().reset_index()

# Rename columns for clarity
# df_yearly.columns = ['Year', 'Max Temperature']
# st.line_chart(df_yearly.set_index('Year'))

fig = px.line(df_yearly_max, x='DATE', y='TMAX', markers=True, title='MAX temp by year')
fig1 = px.line(df_yearly_min, x='DATE', y='TMIN', markers=True ,title='MIn temp by year')
fig2 = px.line(df_yearly_temp, x='DATE', y='TOBS', markers=True,title='Average temperature range in the data (Year)]')

# Add labels to each point
# fig.update_traces(text=df_yearly['TMAX'], textposition="top center", mode='markers+lines+text')

# Display in Streamlit
# col3 ,col4  = st.columns([10,10])
# with col3:
st.plotly_chart(fig2)
# with col4:    
    # st.plotly_chart(fig1)
# with col5:    
    # st.plotly_chart(fig2)
   

# data preprocessing
st.header('Data Preprocessing')
st.write("""
    In the data preprocessing stage, several critical steps were undertaken to prepare the dataset for modeling. These steps ensured that the data was clean, free from inconsistencies, and rich with features that enhance the predictive power of the LSTM model.""")

st.write(""" **1. Handling Missing Data:**
    Missing values in the dataset were addressed using the forward fill (ffill) method. This technique propagates the last valid observation forward to fill missing values. It is particularly effective for time series data, ensuring that temporal continuity is maintained without introducing significant biases. """)

st.write(""" **2. Feature Engineering:**
    To capture more complex patterns and trends in the data, additional rolling window features were engineered. These features include both short-term (3-day) and long-term (14-day) averages and percentage changes, providing a richer temporal context to the model. The newly added features are: """)

st.markdown("""
1. **rolling_3_TMAX:** 3-day rolling mean of maximum temperature.
2. **rolling_3_TMAX_pct:** 3-day rolling percentage change of maximum temperature.
3. **rolling_3_TMIN:** 3-day rolling mean of minimum temperature.
4. **rolling_3_TMIN_pct:** 3-day rolling percentage change of minimum temperature.
5. **rolling_3_TOBS:** 3-day rolling mean of observed temperature.
6. **rolling_3_TOBS_pct:** 3-day rolling percentage change of observed temperature.
7. **rolling_14_TMAX:** 14-day rolling mean of maximum temperature.
8. **rolling_14_TMAX_pct:** 14-day rolling percentage change of maximum temperature.
9. **rolling_14_TMIN:** 14-day rolling mean of minimum temperature.
10. **rolling_14_TMIN_pct:** 14-day rolling percentage change of minimum temperature.
11. **rolling_14_TOBS:** 14-day rolling mean of observed temperature.
12. **rolling_14_TOBS_pct:** 14-day rolling percentage change of observed temperature.
13. **month_avg_PRCP:** Monthly average of precipitation.
14. **day_avg_PRCP:** Daily average of precipitation.
15. **month_avg_TMAX:** Monthly average of maximum temperature.
16. **day_avg_TMAX:** Daily average of maximum temperature.
17. **month_avg_TMIN:** Monthly average of minimum temperature.
18. **day_avg_TMIN:** Daily average of minimum temperature.
        """)

st.write("""These features are designed to provide the model with both smoothed and dynamic representations of the climate variables, enabling it to learn from both recent and more extended historical data.""")

st.write("""**3. Handling Infinite Values:**
    During the feature engineering process, there was a possibility of generating infinite values, particularly when calculating percentage changes. To prevent these from impacting the model, all infinite values were replaced with NaN using df.replace([np.inf, -np.inf], np.nan, inplace=True). Subsequently, any remaining NaN values were handled appropriately to maintain data integrity.""")

st.write("""These preprocessing steps were crucial in ensuring that the dataset was well-prepared for training the LSTM model, enhancing its ability to make accurate predictions based on both short-term fluctuations and long-term trends in the climate data.""")


# model architecture
st.header("LSTM Model Architecture")
model.summary(print_fn=lambda x: st.text(x))
st.markdown("""
1. **Optimizer**: ADAM 
2. **Learning_rate**: 0.01
""")
st.write('After tuned with random search and grid search , the best params came up with')

# training desciption
st.header('Training Process:')
st.markdown("""
* Epoch 1/500

623/623 - 151s - 242ms/step - loss: 762.6677

* Epoch 500/500

623/623 - 182s - 212ms/step - loss: 0.1320
""")
st.markdown("""
1. **Batch_size** : 32 
2. **verbose** : 2
3. **Epochs** : 500
""")

# evluation metrics
st.header('Evaluation By Metrics:')
st.markdown("""
* **RMSE** : **1.9393139730443252**
* **MAE**  : **1.5011177484528952**
""")

st.header('Hyperparameter tuning')
st.subheader('RandomSearch')
st.code("""
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

def build_model(hp):
    lstm_units_1 = hp.Int('lstm_units_1', min_value=50, max_value=200, step=50)
    lstm_units_2 = hp.Int('lstm_units_2', min_value=50, max_value=200, step=50)
    lstm_units_3 = hp.Int('lstm_units_3', min_value=50, max_value=200, step=50)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    optimizer_type = hp.Choice('optimizer_type', values=['adam', 'rmsprop', 'sgd'])
    
    model = LSTMModel(lstm_units_1, lstm_units_2, lstm_units_3, dropout_rate, learning_rate, optimizer_type).build()
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='lstm_tuning',
    )
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, y_train,
             epochs=100,
             validation_data=(x_val, y_val),
             callbacks=[early_stopping])
best_model = tuner.get_best_models(num_models=1)[0]                     
""",language = "python")

st.subheader("""best Hyperparameters:""")
st.markdown("""
* **lstm_units_1:** 150
* **lstm_units_2:** 150
* **lstm_units_3:** 50
* **dropout_rate:** 0.0
* **learning_rate:** 0.002296059476632574
* **optimizer_type:** adam
* **Score:** 0.016188278794288635
""")

st.header("Conclusion Based on the Model and Accuracy")
st.write("""
The LSTM model developed for climate monitoring in the Midwest (Corn Belt) region, covering states like Iowa, Illinois, Indiana, Nebraska, Ohio, Kansas, Missouri, and parts of Minnesota and South Dakota, has shown promising accuracy. With an RMSE of 1.9393 and an MAE of 1.5011, the model demonstrates its capability to predict temperature patterns with a high degree of precision. These metrics indicate that the model can effectively capture the complex relationships between climatic variables, making it a valuable tool for monitoring and forecasting weather conditions in this agriculturally critical region.

Given the Midwest's importance in producing corn, soybeans, and other grains, accurate climate predictions are essential for optimizing crop management, improving yield forecasts, and mitigating the impacts of adverse weather conditions. The model's accuracy suggests that it can be relied upon to provide actionable insights for farmers, agronomists, and policymakers in the region, ultimately contributing to enhanced agricultural productivity and sustainability.
""")