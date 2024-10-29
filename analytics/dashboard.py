import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analytics.database_handler import ChatbotDB
import sqlite3
from datetime import datetime, timedelta

def load_data():
    """Load and prepare data for the dashboard."""
    db = ChatbotDB()
    return db.get_analytics_data()

def create_intent_distribution(common_intents):
    """Create intent distribution chart."""
    if not common_intents:
        return None
        
    df = pd.DataFrame(common_intents, columns=['Intent', 'Count'])
    fig = px.bar(
        df,
        x='Intent',
        y='Count',
        title='Most Common Intents',
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False)
    return fig

def create_rating_distribution(rating_distribution):
    """Create rating distribution chart."""
    if not rating_distribution:
        return None
        
    df = pd.DataFrame(rating_distribution, columns=['Rating', 'Count'])
    fig = px.pie(
        df,
        values='Count',
        names='Rating',
        title='User Satisfaction Distribution',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    return fig

def create_interactions_timeline(interactions_over_time):
    """Create interactions timeline chart."""
    if not interactions_over_time:
        return None
        
    df = pd.DataFrame(interactions_over_time, columns=['Date', 'Count'])
    fig = px.line(
        df,
        x='Date',
        y='Count',
        title='Interactions Over Time',
        markers=True
    )
    fig.update_traces(line_color='#19A7CE')
    return fig

def main():
    st.set_page_config(
        page_title="Chatbot Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Chatbot Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    data = load_data()
    
    if not data:
        st.error("No data available. Please make sure the database is properly configured.")
        return
    
    # Key Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Interactions",
            value=f"{data['total_interactions']:,}"
        )
    
    with col2:
        st.metric(
            label="Average Confidence Score",
            value=f"{data['avg_confidence']:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Average User Rating",
            value=f"{data['avg_rating']:.1f}/5.0"
        )
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Intent Distribution
        intent_fig = create_intent_distribution(data['common_intents'])
        if intent_fig:
            st.plotly_chart(intent_fig, use_container_width=True)
        else:
            st.info("No intent data available")
    
    with col2:
        # Rating Distribution
        rating_fig = create_rating_distribution(data['rating_distribution'])
        if rating_fig:
            st.plotly_chart(rating_fig, use_container_width=True)
        else:
            st.info("No rating data available")
    
    # Timeline Chart
    timeline_fig = create_interactions_timeline(data['interactions_over_time'])
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.info("No timeline data available")

if __name__ == "__main__":
    main()
