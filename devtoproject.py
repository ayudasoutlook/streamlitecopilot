import os
import tempfile
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime


def save_and_download_plot(fig, plot_name):
    try:
        # Create temp directory if it doesn't exist
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plot_name}_{timestamp}.png"
        filepath = os.path.join(temp_dir, filename)
        
        # Save figure
        fig.write_image(filepath)
        
        # Create download button
        with open(filepath, "rb") as file:
            btn = st.download_button(
                label=f"Download {plot_name}",
                data=file,
                file_name=filename,
                mime="image/png"
            )
        
        # Cleanup
        os.remove(filepath)
        os.rmdir(temp_dir)
        
    except Exception as e:
        st.error(f"Error saving plot: {str(e)}")

def perform_advanced_analysis(df):
    st.write("### Advanced Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Correlation Analysis", "Linear Regression", "K-means Clustering"]
    )
    
    if analysis_type == "Correlation Analysis":
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                annotation_text=corr_matrix.round(2).values,
                showscale=True
            )
            st.plotly_chart(fig)
            save_and_download_plot(fig, "correlation_heatmap")
        else:
            st.warning("No numeric columns available for correlation analysis")
    
    elif analysis_type == "Linear Regression":
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        x_col = st.selectbox("Select Independent Variable (X)", numeric_cols)
        y_col = st.selectbox("Select Dependent Variable (Y)", numeric_cols)
        
        X = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        fig = px.scatter(df, x=x_col, y=y_col, title=f'Linear Regression: {y_col} vs {x_col}')
        fig.add_scatter(x=df[x_col], y=model.predict(X), 
                       name='Regression Line',
                       line=dict(color='red'))
        
        st.plotly_chart(fig)
        save_and_download_plot(fig, "regression_plot")
        st.write(f"R² Score: {model.score(X, y):.4f}")
        st.write(f"Coefficient: {model.coef_[0]:.4f}")
        st.write(f"Intercept: {model.intercept_:.4f}")
    
    elif analysis_type == "K-means Clustering":
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        col1 = st.selectbox("Select First Feature", numeric_cols)
        col2 = st.selectbox("Select Second Feature", numeric_cols)
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        X = df[[col1, col2]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        fig = px.scatter(df, x=col1, y=col2, color='Cluster',
                        title=f'K-means Clustering ({n_clusters} clusters)')
        st.plotly_chart(fig)
        save_and_download_plot(fig, "clustering_plot")
        


def create_filters(df):
    st.sidebar.write("### Data Filters")
    filtered_df = df.copy()
    
    # Filter by categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        unique_values = df[col].unique()
        if len(unique_values) < 50:  # Only show if manageable number of categories
            selected_values = st.sidebar.multiselect(
                f'Select {col}',
                unique_values,
                default=unique_values
            )
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    # Filter by numeric columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        range_vals = st.sidebar.slider(
            f'Range for {col}',
            min_val, max_val,
            (min_val, max_val)
        )
        filtered_df = filtered_df[
            (filtered_df[col] >= range_vals[0]) & 
            (filtered_df[col] <= range_vals[1])
        ]
    
    # Filter by date columns if any
    for col in df.select_dtypes(include=['datetime64']).columns:
        min_date = df[col].min()
        max_date = df[col].max()
        date_range = st.sidebar.date_input(
            f'Range for {col}',
            [min_date, max_date]
        )
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df[col].dt.date >= date_range[0]) & 
                (filtered_df[col].dt.date <= date_range[1])
            ]
    
    return filtered_df

def create_visualizations(df):
    st.write("### Data Visualization")
    
    # Select columns for visualization
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Chart type selector
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Plot"]
    )
    
    if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
        x_col = st.selectbox("Select X-axis", df.columns)
        y_col = st.selectbox("Select Y-axis", numeric_cols)
        
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
        else:  # Scatter Plot
            fig = px.scatter(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
            
    elif chart_type == "Pie Chart":
        value_col = st.selectbox("Select Value Column", numeric_cols)
        name_col = st.selectbox("Select Name Column", categorical_cols)
        fig = px.pie(df, values=value_col, names=name_col, title=f'{value_col} Distribution')
    
    st.plotly_chart(fig)
    save_and_download_plot(fig, chart_type.replace(" ", "_").lower())


def display_statistics(df):
    st.write("### Dataset Statistics")
    
    # Basic dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Total Null Values", df.isnull().sum().sum())
    
    # Null values by column
    st.write("#### Null Values by Column")
    null_df = pd.DataFrame({
        'Column': df.columns,
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(null_df)
    
    # Descriptive statistics for numeric columns
    st.write("#### Numeric Columns Statistics")
    numeric_stats = df.describe().round(2)
    st.dataframe(numeric_stats)

def load_data():
    st.title('Upload data')
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Convert date columns if any
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            st.success(f'Successfully loaded: {uploaded_file.name}')
            
            # Apply filters
            filtered_df = create_filters(df)
            
            # Show filtered data info
            st.write(f"### Filtered Data ({len(filtered_df)} rows)")
            st.write(filtered_df)
            
            # Add download button
            csv = filtered_df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filtered_data_{timestamp}.csv"
            
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
            # Display statistics and visualizations with filtered data
            #""" display_statistics(filtered_df)
            #create_visualizations(filtered_df)
            #perform_advanced_analysis(filtered_df) """
            
            return filtered_df
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
    
    return None

def main():
    st.set_page_config(layout="wide")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Upload Data", "Data Statistics", "Visualization", "Advanced Analysis"]
    )
    
    # Main title
    st.title("Data Analysis Dashboard")
    
    # Handle navigation
    if page == "Upload Data":
        df = load_data()
        if df is not None:
            st.session_state.data = df
    
    elif page == "Data Statistics":
        if st.session_state.data is not None:
            display_statistics(st.session_state.data)
        else:
            st.warning("Please upload data first")
    
    elif page == "Visualization":
        if st.session_state.data is not None:
            create_visualizations(st.session_state.data)
        else:
            st.warning("Please upload data first")
    
    elif page == "Advanced Analysis":
        if st.session_state.data is not None:
            perform_advanced_analysis(st.session_state.data)
        else:
            st.warning("Please upload data first")

if __name__ == "__main__":
    main()