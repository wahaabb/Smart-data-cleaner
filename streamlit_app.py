import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="Smart Data Cleaner",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------
# Custom CSS
# -----------------------------------

st.markdown("""
<style>

.main {
    padding: 1rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 10px;
    padding: 0.7rem;
    font-weight: 600;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Session State
# -----------------------------------

if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None

# -----------------------------------
# Header
# -----------------------------------

st.markdown("""
<div class="card">
    <h1>✨ Smart Data Cleaner</h1>
    <p>Upload, analyze, clean and download your dataset instantly.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# Upload File
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# -----------------------------------
# Main App
# -----------------------------------

if uploaded_file is not None:

    # Load Data
    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # -----------------------------------
    # Dataset Metrics
    # -----------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())

    st.markdown("---")

    # -----------------------------------
    # Tabs
    # -----------------------------------

    tab1, tab2, tab3 = st.tabs([
        "📋 Data Preview",
        "⚠️ Data Issues",
        "📊 Statistics"
    ])

    # -----------------------------------
    # Tab 1 - Preview
    # -----------------------------------

    with tab1:

        st.subheader("Dataset Preview")

        st.dataframe(df.head(100), use_container_width=True)

        if st.checkbox("Show Full Dataset"):
            st.dataframe(df, use_container_width=True)

    # -----------------------------------
    # Tab 2 - Issues
    # -----------------------------------

    with tab2:

        # Missing Values
        st.subheader("Missing Values")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if not missing.empty:

            missing_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Count": missing.values,
                "Percentage": (
                    missing.values / len(df) * 100
                ).round(2)
            })

            st.dataframe(missing_df, use_container_width=True)

            fig_missing = px.bar(
                missing_df,
                x="Column",
                y="Missing Count",
                color="Missing Count",
                title="Missing Values by Column"
            )

            st.plotly_chart(fig_missing, use_container_width=True)

        else:
            st.success("✅ No Missing Values Found")

        st.markdown("---")

        # Duplicate Rows
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows")
        else:
            st.success("✅ No Duplicate Rows Found")

        st.markdown("---")

        # Outliers
        st.subheader("Outlier Detection")

        numeric_cols = df.select_dtypes(include=np.number).columns

        outlier_info = {}

        for col in numeric_cols:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            outliers = df[
                (df[col] < Q1 - 1.5 * IQR) |
                (df[col] > Q3 + 1.5 * IQR)
            ]

            outlier_info[col] = len(outliers)

        outlier_df = pd.DataFrame({
            "Column": outlier_info.keys(),
            "Outlier Count": outlier_info.values()
        })

        st.dataframe(outlier_df, use_container_width=True)

        fig_outlier = px.bar(
            outlier_df,
            x="Column",
            y="Outlier Count",
            color="Outlier Count",
            title="Outliers by Column"
        )

        st.plotly_chart(fig_outlier, use_container_width=True)

    # -----------------------------------
    # Tab 3 - Statistics
    # -----------------------------------

    with tab3:

        st.subheader("Statistical Summary")

        st.dataframe(df.describe(include='all'),
                     use_container_width=True)

        st.markdown("---")

        # Correlation Matrix
        if len(numeric_cols) > 1:

            corr = df[numeric_cols].corr()

            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix"
            )

            st.plotly_chart(fig_corr, use_container_width=True)

    # -----------------------------------
    # Cleaning Section
    # -----------------------------------

    st.markdown("---")

    st.header("🧹 Data Cleaning")

    cleaning_method = st.selectbox(
        "Choose Cleaning Method",
        [
            "Auto Fill (Median/Mode)",
            "Mean Fill",
            "Drop Missing Rows",
            "Custom Fill"
        ]
    )

    custom_value = None

    if cleaning_method == "Custom Fill":
        custom_value = st.text_input(
            "Enter custom fill value"
        )

    remove_duplicates = st.checkbox(
        "Remove Duplicate Rows",
        value=True
    )

    reset_index = st.checkbox(
        "Reset Index",
        value=True
    )

    # -----------------------------------
    # Apply Cleaning
    # -----------------------------------

    if st.button("✨ Apply Cleaning"):

        df_cleaned = df.copy()

        try:

            # -----------------------------------
            # Missing Value Handling
            # -----------------------------------

            if cleaning_method == "Auto Fill (Median/Mode)":

                for col in df_cleaned.columns:

                    if df_cleaned[col].dtype in ['int64', 'float64']:

                        df_cleaned[col].fillna(
                            df_cleaned[col].median(),
                            inplace=True
                        )

                    else:

                        mode_val = df_cleaned[col].mode()

                        if not mode_val.empty:
                            df_cleaned[col].fillna(
                                mode_val[0],
                                inplace=True
                            )
                        else:
                            df_cleaned[col].fillna(
                                "Unknown",
                                inplace=True
                            )

            elif cleaning_method == "Mean Fill":

                numeric_cols_clean = df_cleaned.select_dtypes(
                    include=np.number
                ).columns

                for col in numeric_cols_clean:

                    df_cleaned[col].fillna(
                        df_cleaned[col].mean(),
                        inplace=True
                    )

            elif cleaning_method == "Drop Missing Rows":

                df_cleaned.dropna(inplace=True)

            elif cleaning_method == "Custom Fill":

                df_cleaned.fillna(custom_value, inplace=True)

            # -----------------------------------
            # Remove Duplicates
            # -----------------------------------

            if remove_duplicates:
                df_cleaned.drop_duplicates(inplace=True)

            # -----------------------------------
            # Reset Index
            # -----------------------------------

            if reset_index:
                df_cleaned.reset_index(drop=True, inplace=True)

            # Save to Session
            st.session_state.df_cleaned = df_cleaned

            # -----------------------------------
            # Success Message
            # -----------------------------------

            st.success("✅ Data cleaned successfully!")

            # -----------------------------------
            # Cleaned Preview
            # -----------------------------------

            st.subheader("✅ Cleaned Dataset Preview")

            st.dataframe(
                df_cleaned.head(100),
                use_container_width=True
            )

            # -----------------------------------
            # Download CSV
            # -----------------------------------

            csv = df_cleaned.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(f"❌ Error: {str(e)}")

# -----------------------------------
# Empty State
# -----------------------------------

else:

    st.markdown("""
    <div style="text-align:center; padding:4rem;">
        <h2>📂 Upload a CSV File</h2>
        <p>Analyze and clean your dataset automatically.</p>
    </div>
    """, unsafe_allow_html=True)