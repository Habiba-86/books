from pymongo import MongoClient
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu  # type: ignore
from wordcloud import WordCloud
###############################################################
st.set_page_config(page_title="Book ", layout="wide")
##############################################################################
client = MongoClient("mongodb://localhost:27017/")
db = client["Database"]
collection = db["Books"]
data = list(collection.find())
df = pd.DataFrame(data)

# ======================================Sidebar navigation============================
with st.sidebar:
    st.markdown("####   Jump to Section")
    
    page = option_menu(
        menu_title=None,  # لا نحتاج عنوان داخل القائمة
        options=["Home Page", "View Data", "Data Analysis", "Visualizations"],
        icons=["house", "file-earmark-text", "bar-chart-line", "graph-up-arrow"],
        menu_icon="cast",  # الأيقونة اللي جنب العنوان (مش ضرورية هنا)
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#a6c1ee", "font-weight": "bold"},
        }
    )

    st.session_state.page = page

# تحديد الصفحة الحالية إذا لم تكن موجودة
if "page" not in st.session_state:
    st.session_state.page = "Home Page"
 
#====================home page====================================================================================================
if st.session_state.page == "Home Page":
    st.markdown("""
    <style>
        .main-header {
            color: #2E86AB;
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #f0f2f6;
            margin-bottom: 30px;
        }
        
    </style>
    """, unsafe_allow_html=True)

    # Main Title
    st.markdown('<h1 class="main-header">Book Data Analysis  </h1>', unsafe_allow_html=True)
    
    # ========== Welcome Section ==========
    with st.container():
        st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h3 style="color: #2E86AB;">Welcome to the Data Analysis 👋</h3>
        <p>Explore and analyze book data with interactive tools:</p>
        <ul>
            <li>View raw data directly from MongoDB</li>
            <li>Perform advanced data analysis</li>
            <li>Generate beautiful visualizations</li>
        </ul>
        <p><strong>📁 Project Files (PDF, Python Code, Report):</strong> 
        <a href="https://drive.google.com/drive/folders/1EIJHExDLks6hUlXnslfDsVec1jdnENjD?usp=drive_link" target="_blank">View on Google Drive</a></p>
        <p><strong>🌐 Data Source (Web Scraped):</strong> 
        <a href="https://books.toscrape.com/index.html" target="_blank">Books to Scrape Website</a></p>
    </div>
    """, unsafe_allow_html=True)

#=======================================View data==========================================================================================
elif  st.session_state.page == "View Data":
    st.subheader("Data View")
    st.write("### Column Types")
    st.write(pd.DataFrame(df.dtypes, columns=['Data Type']))
    rows_to_show = st.slider("Select number of rows to display", 5, 100, 10)
    st.write(f"Showing  {rows_to_show} rows:")
    st.dataframe(df.sample(rows_to_show))
    if st.checkbox("Show summary statistics"):
        st.write("### Descriptive Statistics")
        st.write(df.describe())
    st.markdown("---")
#==========================Data Analysis =============================================================================
elif  st.session_state.page==  "Data Analysis":
    st.subheader("Data Analysis 📊")

    analysis_option = st.selectbox("Select Analysis Type",
                                    [" Analysis Type","Descriptive statistics",
                                        "Trend identification",
                                        "Patterns Recognition and Relationship Analysis",
                                        "Clustering Analysis"])
####################1.Descriptive statistics:
    if analysis_option=="Descriptive statistics":
        st.markdown("### Descriptive Statistics Analysis")

        feature_type = st.radio("Select feature type:",
                            ["Numerical features", 
                            "Categorical features"],
                                horizontal=True)
        if feature_type=="Numerical features":
            num_col = st.selectbox("Select numerical column:",
                            ['numerical column',"Price", "Rating"])
            if num_col=="Price":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"${df['Price'].mean():.2f}")
                    st.metric("Min", f"${df['Price'].min():.2f}")
                with col2:
                    st.metric("Median", f"${df['Price'].median():.2f}")
                    st.metric("Max", f"${df['Price'].max():.2f}")
                with col3:
                    st.metric("Std Dev", f"${df['Price'].std():.2f}")
                    st.metric("Count", len(df))
            elif num_col == "Rating":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{df['Rating'].mean():.2f}")
                    st.metric("Min", f"{df['Rating'].min():.2f}")
                with col2:
                    st.metric("Median", f"{df['Rating'].median():.2f}")
                    st.metric("Max", f"{df['Rating'].max():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df['Rating'].std():.2f}")
                    st.metric("Count", len(df))        
        if feature_type=="Categorical features":
                category_counts = df['Category'].value_counts()
                most_common = category_counts.idxmax()
                most_common_count = category_counts.max()
                total_count = len(df)
                percentage = (most_common_count / total_count) * 100
                st.metric("Most Common Category", most_common)
                st.metric("Occurrence Count", most_common_count)
                st.metric("Percentage Share", f"{percentage:.2f}%")
###############2.Trend identification
    elif analysis_option == "Trend identification":
        st.markdown("### Trend Identification")
    
        tab1, tab2 = st.tabs(["Price Trends","Rating Trends"])
    
        with tab1:
        
            avg_price_category = df.groupby('Category')['Price'].mean().sort_values(ascending=False)
        
            st.write("**Top 5 Categories by Average Price:**")
            for category, avg_price in avg_price_category.head(5).items():
               st.markdown(f"- **{category}**: ${avg_price:.2f}")
        
            price_rating_corr = df['Price'].corr(df['Rating'])
        
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Correlation between price and rating:", f"{price_rating_corr:.2f}")

                if price_rating_corr > 0:
                    st.success("Higher-priced books tend to receive better ratings")
                else  :
                    st.warning("Lower-priced books tend to receive better ratings")
            with tab2:
                avg_rating_by_category = df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
                
                st.write("**Certain categories tend to have higher average ratings**")
                for i, (category, rating) in enumerate(avg_rating_by_category.head(5).items(), 1):
                        st.metric(
                label=f"{i}. {category}",
                value=f"{rating:.2f}",
                help=f"Average rating: {rating:.2f}"
            )
####################3.Patterns Recognition and Relationship Analysis
    elif analysis_option == "Patterns Recognition and Relationship Analysis":
        st.markdown("### Patterns and Relationships Analysis")
    
        correlation_matrix = df[['Price', 'Rating']].corr()
        price_rating_corr = correlation_matrix.loc['Price', 'Rating']
    
    
        st.markdown("#### Correlation Matrix")
        st.dataframe(correlation_matrix.style.highlight_max(axis=0, color='#d4f1d4'))
    
        st.metric("Price-Rating Correlation", 
            f"{price_rating_corr}",
            help="Range: -1 (perfect negative) to +1 (perfect positive)")
###############################4.Clustering Analysis
    elif analysis_option == "Clustering Analysis":
            st.markdown("### Clustering Analysis")
            n_clusters = st.slider("Number of Clusters", 
                            min_value=2, 
                            max_value=5, 
                            value=3,
                            help="Select how many groups to divide the books into")
    
            features = df[['Price', 'Rating']]
    
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(features)
    
            st.markdown("###  Cluster Centers")
            cluster_centers = pd.DataFrame(kmeans.cluster_centers_, 
                                columns=['Price', 'Rating'])
    
            st.dataframe(cluster_centers.style.highlight_max(axis=0, color='#d4f1d4'))
    
            st.markdown("###  Cluster Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))  
    
            scatter = ax.scatter(features['Price'], 
                        features['Rating'], 
                        c=df['Cluster'], 
                        cmap='viridis', 
                        alpha=0.6)
    
            ax.scatter(cluster_centers['Price'], 
                cluster_centers['Rating'], 
                c='red', 
                marker='X', 
                s=200, 
                label='Cluster Centers')
    
            ax.set_xlabel("Price")
            ax.set_ylabel("Rating")
            ax.set_title(f"Book Clusters (k={n_clusters})")
            ax.legend()
    
            plt.colorbar(scatter, ax=ax, label='Cluster')
    
            st.pyplot(fig)
#=============================Visualizations=====================================================================
elif st.session_state.page == "Visualizations":
    st.subheader("Visualizations 📉")
############1&3done
    st.markdown("#### Show the distribution of Book Prices")
    with st.expander(" Optional Filters"):
        col1, col2 = st.columns(2)
        with col1:
            category_filter = st.multiselect(
            "Filter by Category (optional):",
            options=df['Category'].unique()
            )

        with col2:
            rating_filter = st.multiselect(
            "Filter by Rating (optional):",
            options=sorted(df['Rating'].unique())
        )

    filtered_df = df.copy()

    if category_filter:
        filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]

    if rating_filter:
        filtered_df = filtered_df[filtered_df['Rating'].isin(rating_filter)]

    
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['Price'], bins=30, kde=True, color='teal')
        ax.set_title('Price Distribution')
        ax.set_xlabel('Price (£)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        st.warning(" No data available for the selected filters.")
    st.divider()
##################2done
    st.markdown("#### Rank Categories  ")
    selected_ratings = st.multiselect(
    " Select specific ratings to display (optional):",
    options=sorted(df['Rating'].unique()),
    default=sorted(df['Rating'].unique())  
)

    filtered_df = df[df['Rating'].isin(selected_ratings)] if selected_ratings else df

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('###  Rating Distribution' )
        fig, ax = plt.subplots(figsize=(6,4))
        rating_counts = filtered_df['Rating'].value_counts()
        st.write(rating_counts)
        sns.countplot(x='Rating', data=filtered_df, palette='viridis')
        ax.set_title('Distribution of Book Ratings')
        ax.set_xlabel('Rating (1-5)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    with col2:
        st.markdown('###  pie chart of Rating distribution')
        fig, ax = plt.subplots(figsize=(6,4))
        plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90, colors=['purple','pink','gray','teal','turquoise'])
        ax.set_title('Distribution of Book Ratings')
        st.pyplot(fig)
    st.divider()
################4&5
    st.markdown("###    Rank Categories")

    option = st.selectbox(
    "Choose the type of insight to display:",
    ["-- Select an option --", "Top Categories by Average Price", "Top Categories by Number of Books"]
)

    if option == "-- Select an option --":
        st.warning(" Please select a valid option to view the chart.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        if option == "Top Categories by Average Price":
            st.markdown("####  Top 10 Categories by Average Price")
            top_categories = df.groupby('Category')['Price'].mean().sort_values(ascending=False).head(10)
            st.write(top_categories)
            top_categories.plot(kind='barh', color='purple', ax=ax)
            ax.set_title('Top 10 Categories by Average Price')
            ax.set_xlabel('Average Price (£)')
            ax.set_ylabel('Category')

        elif option == "Top Categories by Number of Books":
            st.markdown("#### Top 20 Categories by Number of Books")
            top_categories_count = df['Category'].value_counts().head(20)
            sns.barplot(y=top_categories_count.index, x=top_categories_count.values, palette='rocket', ax=ax)
            ax.set_title('Top 20 Categories by Number of Books')
            ax.set_xlabel('Number of Books')
            ax.set_ylabel('Category')

        st.pyplot(fig)
        st.divider()


##############6done
    st.markdown("####  Price vs Rating by Category")

    selected_categories = st.multiselect(
    " Choose categories to visualize the relationship between Price and Rating:",
    options=df['Category'].unique(),
    default=df['Category'].value_counts().head(5).index.tolist()
    )

# تصفية الداتا على حسب اختيار المستخدم
    if selected_categories:
        sample_df = df[df['Category'].isin(selected_categories)]
    
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
        x='Rating', y='Price', hue='Category',
        data=sample_df, palette='deep', s=100, ax=ax
        )
        ax.set_title('Price vs Rating for Selected Categories')
        ax.set_xlabel('Rating (1-5)')
        ax.set_ylabel('Price (£)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Select one or more categories to explore the chart!")
    st.divider()
#####################7
    from wordcloud import WordCloud

    st.markdown("###  Word Cloud of Book Titles")

    categories = df['Category'].unique().tolist()
    selected_category = st.selectbox( " Choose a Category to Generate Word Cloud", ["All"] + categories)

    if selected_category == "All":
        titles = df['Title']
    else:
        titles = df[df['Category'] == selected_category]['Title']

    if not titles.empty:
        wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100
        ).generate(' '.join(titles))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Word Cloud of Titles in Category: {selected_category}" if selected_category != "All" else "Word Cloud of All Book Titles", fontsize=16)
        st.pyplot(fig)
    else:
        st.warning(" No book titles available for the selected category.")
    