import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import calendar
import statsmodels.api as sm  # Ensure statsmodels is imported

# Set page config
st.set_page_config(layout="wide", page_title="NJ Transit Mechanical Cancellations Analysis")

@st.cache_data
def load_data():
    # Load both datasets
    mechanical_df = pd.read_csv('data/RAIL_CANCELLATIONS_DATA.csv')
    train_df = pd.read_csv('data/Combined/cleaned_train_data.csv')
    
    # Clean mechanical data
    mechanical_df['MONTH'] = mechanical_df['MONTH'].str.strip()
    mechanical_df = mechanical_df[mechanical_df['CATEGORY'].str.contains('Mechanical', na=False)]
    
    # Convert month names to numbers in mechanical data
    month_map = {
        'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
        'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
        'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12
    }
    mechanical_df['MONTH_NUM'] = mechanical_df['MONTH'].map(month_map)
    
    # Merge with train data
    train_df['MONTH_NUM'] = train_df['MONTH'].map(month_map)
    merged_df = pd.merge(
        mechanical_df,
        train_df[['YEAR', 'MONTH_NUM', 'MEAN_DISTANCE_BEFORE_FAILURE', 'ON_TIME_PERCENTAGE']],
        on=['YEAR', 'MONTH_NUM'],
        how='left'
    )
    
    # Create date column
    merged_df['DATE'] = pd.to_datetime(merged_df.apply(lambda x: f"{x['YEAR']}-{x['MONTH_NUM']:02d}-01", axis=1))
    
    return merged_df

def predict_mechanical_failures(data, target_month=None):
    # Prepare features
    X = data[['YEAR', 'MONTH_NUM', 'MEAN_DISTANCE_BEFORE_FAILURE', 'ON_TIME_PERCENTAGE']].values
    y = data['CANCEL_PERCENTAGE'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    if target_month:
        # Predict for specific month across years
        years = range(data['YEAR'].min(), data['YEAR'].max() + 2)
        
        # Use average values for other features
        avg_distance = data['MEAN_DISTANCE_BEFORE_FAILURE'].mean()
        avg_ontime = data['ON_TIME_PERCENTAGE'].mean()
        
        future_dates = [[year, target_month, avg_distance, avg_ontime] for year in years]
        predictions = model.predict(np.array(future_dates))
        return future_dates, predictions
    else:
        # Predict next 6 months
        current_year = data['YEAR'].max()
        current_month = data['MONTH_NUM'].max()
        
        avg_distance = data['MEAN_DISTANCE_BEFORE_FAILURE'].mean()
        avg_ontime = data['ON_TIME_PERCENTAGE'].mean()
        
        future_dates = []
        for i in range(1, 7):
            month = (current_month + i) % 12
            if month == 0:
                month = 12
            year = current_year + (current_month + i - 1) // 12
            future_dates.append([year, month, avg_distance, avg_ontime])
        
        predictions = model.predict(np.array(future_dates))
        return future_dates, predictions

def show_feature_importance(data):
    X = data[['YEAR', 'MONTH_NUM', 'MEAN_DISTANCE_BEFORE_FAILURE', 'ON_TIME_PERCENTAGE']]
    y = data['CANCEL_PERCENTAGE']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df, 
                 x='Feature', 
                 y='Importance',
                 title='Feature Importance for Prediction Model')
    return fig

def create_monthly_heatmap(df):
    # Pivot data for heatmap
    heatmap_data = df.pivot_table(
        values='CANCEL_PERCENTAGE',
        index='YEAR',
        columns='MONTH_NUM',
        aggfunc='mean'
    ).round(1)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[calendar.month_abbr[i] for i in range(1, 13)],
        y=heatmap_data.index,
        colorscale='RdYlBu_r',
        text=np.round(heatmap_data.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title='Cancellation Rate (%)')
    ))
    
    fig.update_layout(
        title='Monthly Mechanical Cancellation Rates Heatmap',
        xaxis_title='Month',
        yaxis_title='Year'
    )
    
    return fig

def main():
    st.title("🚂 NJ Transit Rail Mechanical Cancellations Analysis")
    
    # Display explanation image
    st.image('assets/output.png',
             caption='Distribution of Cancellation Categories',
             use_container_width=True)
    
    # Load data
    df = load_data()
    
    # Main metrics
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_mechanical_rate = df['CANCEL_PERCENTAGE'].mean()
        st.metric("Average Mechanical Cancellation Rate", f"{avg_mechanical_rate:.1f}%")
    
    with col2:
        recent_rate = df[df['YEAR'] == df['YEAR'].max()]['CANCEL_PERCENTAGE'].mean()
        st.metric("Recent Year Average", f"{recent_rate:.1f}%", 
                 f"{recent_rate - avg_mechanical_rate:.1f}%")
    
    with col3:
        max_month = df.loc[df['CANCEL_PERCENTAGE'].idxmax()]
        st.metric("Highest Cancellation Month", 
                 f"{max_month['MONTH']} {max_month['YEAR']}", 
                 f"{max_month['CANCEL_PERCENTAGE']:.1f}%")

    # Monthly Pattern Analysis
    st.header("Monthly Cancellation Patterns")
    
    # Heatmap
    heatmap_fig = create_monthly_heatmap(df)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Historical trend with trend line
    fig_trend = px.scatter(df, 
                          x='DATE', 
                          y='CANCEL_PERCENTAGE',
                          trendline="lowess",
                          title='Historical Mechanical Cancellation Trend',
                          labels={'CANCEL_PERCENTAGE': 'Cancellation Rate (%)',
                                 'DATE': 'Date'})
    
    fig_trend.update_traces(marker=dict(size=8))
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Monthly box plot
    fig_box = px.box(df, 
                     x='MONTH', 
                     y='CANCEL_PERCENTAGE',
                     title='Monthly Distribution of Mechanical Cancellations',
                     labels={'CANCEL_PERCENTAGE': 'Cancellation Rate (%)',
                            'MONTH': 'Month'})
    st.plotly_chart(fig_box, use_container_width=True)

    # Feature Importance
    st.header("Prediction Model Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(show_feature_importance(df), use_container_width=True)
    
    with col2:
        st.write("""
        ### Model Features:
        - Mean Distance Before Failure: Average distance traveled before a mechanical issue occurs
        - On-Time Percentage: Overall performance metric
        - Month: Seasonal patterns
        - Year: Long-term trends
        
        This enhanced model considers multiple factors to provide more accurate predictions.
        """)

    # Prediction Section
    st.header("Mechanical Failure Predictions")
    
    # Month selector for predictions
    months = list(calendar.month_name)[1:]
    selected_month = st.selectbox("Select month for prediction", months)
    selected_month_num = months.index(selected_month) + 1
    
    # Get predictions for selected month
    future_dates, predictions = predict_mechanical_failures(df, selected_month_num)
    
    # Create prediction chart
    fig_predict = go.Figure()
    
    # Historical data for selected month
    historical_data = df[df['MONTH_NUM'] == selected_month_num]
    fig_predict.add_trace(go.Scatter(
        x=historical_data['YEAR'],
        y=historical_data['CANCEL_PERCENTAGE'],
        name='Historical Data',
        mode='markers+lines',
        marker=dict(size=8)
    ))
    
    # Predictions
    fig_predict.add_trace(go.Scatter(
        x=[date[0] for date in future_dates],
        y=predictions,
        name='Prediction Trend',
        mode='lines',
        line=dict(dash='dash', color='red')
    ))
    
    fig_predict.update_layout(
        title=f'Mechanical Cancellation Predictions for {selected_month}',
        xaxis_title='Year',
        yaxis_title='Cancellation Rate (%)',
        hovermode='x unified'
    )
    st.plotly_chart(fig_predict, use_container_width=True)

    # Cost Impact Analysis
    st.header("Cost Impact Analysis")
    avg_cost_per_cancellation = 5000
    
    col1, col2 = st.columns(2)
    with col1:
        predicted_rate = predictions[-1]
        predicted_cost = predicted_rate * avg_cost_per_cancellation
        st.metric(
            f"Predicted Cost for {selected_month}",
            f"${predicted_cost:,.2f}",
            f"Based on {predicted_rate:.1f}% cancellation rate"
        )
    
    with col2:
        potential_savings = predicted_cost * 0.15
        st.metric(
            "Potential Monthly Savings",
            f"${potential_savings:,.2f}",
            "Through preventive maintenance"
        )

    # Insights
    st.header("Key Insights")
    st.write(f"""
    ### Monthly Analysis for {selected_month}:
    - Historical average cancellation rate: {historical_data['CANCEL_PERCENTAGE'].mean():.1f}%
    - Predicted cancellation rate: {predicted_rate:.1f}%
    - Risk level: {'High' if predicted_rate > avg_mechanical_rate else 'Moderate' if predicted_rate > avg_mechanical_rate/2 else 'Low'}
    
    ### Recommendations:
    - {'Increase maintenance frequency' if predicted_rate > avg_mechanical_rate else 'Maintain regular schedule'}
    - {'Schedule additional inspections' if predicted_rate > avg_mechanical_rate * 1.2 else 'Continue standard inspections'}
    - Focus on preventive maintenance during {'winter months' if selected_month_num in [12,1,2] else 'summer months' if selected_month_num in [6,7,8] else 'transition months'}
    """)

if __name__ == "__main__":
    main()