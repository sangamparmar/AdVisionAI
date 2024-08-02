import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog

st.title("Advanced AI-Driven Media Investment Plan")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Ad Spend Data", type="csv")
if uploaded_file is not None:
    ad_spend_data = pd.read_csv(uploaded_file)
    st.write(ad_spend_data.head())
    
    # Feature Engineering
    ad_spend_data['click_through_rate'] = ad_spend_data['clicks'] / ad_spend_data['impressions']
    ad_spend_data['conversion_rate'] = ad_spend_data['conversions'] / ad_spend_data['clicks']
    ad_spend_data['cost_per_click'] = ad_spend_data['amount_spent'] / ad_spend_data['clicks']
    
    # Prepare data for model
    X = ad_spend_data[['impressions', 'clicks', 'click_through_rate', 'conversion_rate', 'cost_per_click']]
    y = ad_spend_data['conversions']
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Gradient Boosting Model with Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.01, 0.1]
    }
    model = GradientBoostingRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predict
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error: {mae:.2f}")
    
    total_budget = st.number_input("Enter Total Budget", value=100000)
    channels = ad_spend_data['channel'].unique()
    
    # Predict conversions for each channel
    channel_budgets = {}
    predicted_conversions = {}
    
    for channel in channels:
        channel_data = ad_spend_data[ad_spend_data['channel'] == channel]
        if not channel_data.empty:
            mean_features = pd.DataFrame(channel_data[['impressions', 'clicks', 'click_through_rate', 'conversion_rate', 'cost_per_click']].mean().values.reshape(1, -1), 
                                         columns=['impressions', 'clicks', 'click_through_rate', 'conversion_rate', 'cost_per_click'])
            predicted_conversion = best_model.predict(mean_features)[0]
            predicted_conversions[channel] = predicted_conversion
    
    # Optimization using Linear Programming
    c = [-predicted_conversions.get(channel, 0) for channel in channels]
    A_eq = np.ones((1, len(channels)))
    b_eq = [total_budget]
    bounds = [(0.1 * total_budget, 0.5 * total_budget) for _ in channels]  # Example bounds: 10% to 50%
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        for i, channel in enumerate(channels):
            channel_budgets[channel] = result.x[i]
        
        st.write("Reallocated Budgets:")
        st.json(channel_budgets)
        
        # Enhanced Visualization
        fig = go.Figure()

        # Bar chart for Reallocated Budgets
        fig.add_trace(go.Bar(
            x=list(channel_budgets.keys()),
            y=list(channel_budgets.values()),
            name='Reallocated Budget'
        ))

        # Add predicted conversions as a secondary axis
        fig.add_trace(go.Scatter(
            x=list(predicted_conversions.keys()),
            y=list(predicted_conversions.values()),
            mode='lines+markers',
            name='Predicted Conversions',
            yaxis='y2'
        ))

        fig.update_layout(
            title='Budget Allocation by Channel',
            xaxis_title='Channel',
            yaxis_title='Reallocated Budget',
            yaxis2=dict(
                title='Predicted Conversions',
                overlaying='y',
                side='right'
            ),
            template='plotly_dark'
        )

        st.plotly_chart(fig)
    else:
        st.write("Optimization failed. Please check constraints and try again.")
    
    # Scenario Analysis
    st.subheader("Scenario Analysis")
    scenario_budget = st.number_input("Enter Budget for Scenario Analysis", value=total_budget)
    scenario_predictions = {}
    
    for channel in channels:
        channel_data = ad_spend_data[ad_spend_data['channel'] == channel]
        if not channel_data.empty:
            mean_features = pd.DataFrame(channel_data[['impressions', 'clicks', 'click_through_rate', 'conversion_rate', 'cost_per_click']].mean().values.reshape(1, -1), 
                                         columns=['impressions', 'clicks', 'click_through_rate', 'conversion_rate', 'cost_per_click'])
            predicted_conversion = best_model.predict(mean_features)[0]
            scenario_predictions[channel] = predicted_conversion
    
    # Scenario Optimization
    c = [-scenario_predictions.get(channel, 0) for channel in channels]
    bounds = [(0.1 * scenario_budget, 0.5 * scenario_budget) for _ in channels]
    
    scenario_result = linprog(c, A_eq=A_eq, b_eq=[scenario_budget], bounds=bounds, method='highs')
    
    if scenario_result.success:
        scenario_budgets = {}
        for i, channel in enumerate(channels):
            scenario_budgets[channel] = scenario_result.x[i]
        
        st.write("Scenario Analysis - Reallocated Budgets:")
        st.json(scenario_budgets)
        
        # Enhanced Visualization for Scenario
        fig_scenario = go.Figure()

        # Bar chart for Scenario Reallocated Budgets
        fig_scenario.add_trace(go.Bar(
            x=list(scenario_budgets.keys()),
            y=list(scenario_budgets.values()),
            name='Scenario Reallocated Budget'
        ))

        # Add predicted conversions for scenario
        fig_scenario.add_trace(go.Scatter(
            x=list(scenario_predictions.keys()),
            y=list(scenario_predictions.values()),
            mode='lines+markers',
            name='Scenario Predicted Conversions',
            yaxis='y2'
        ))

        fig_scenario.update_layout(
            title='Scenario Budget Allocation by Channel',
            xaxis_title='Channel',
            yaxis_title='Scenario Reallocated Budget',
            yaxis2=dict(
                title='Scenario Predicted Conversions',
                overlaying='y',
                side='right'
            ),
            template='plotly_dark'
        )

        st.plotly_chart(fig_scenario)
    else:
        st.write("Scenario Optimization failed. Please check constraints and try again.")
