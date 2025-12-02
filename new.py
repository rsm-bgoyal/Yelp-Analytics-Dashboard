import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs("/tmp/matplotlib", exist_ok=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from pathlib import Path

# Configure Streamlit
st.set_page_config(
    page_title="Yelp Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Force cache clear on first run
if 'cache_cleared' not in st.session_state:
    st.cache_data.clear()
    st.session_state.cache_cleared = True
    
# Custom CSS with beautiful light styling
st.markdown(
    """
    <style>
    * {
        margin: 0;
        padding: 0;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #e8ecf5 100%);
    }

    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        border: none !important;
    }

    .stMetric label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    .stMetric > div:last-child {
        color: white !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }

    .insight-box {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fc 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        border-top: 2px solid #764ba2;
    }

    .filter-container {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4fc 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.15);
    }

    h1 {
        color: #333 !important;
        font-weight: 800 !important;
        margin-bottom: 20px !important;
    }

    h2 {
        color: #555 !important;
        font-weight: 700 !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
        border-bottom: 3px solid #c5d3ff;
        padding-bottom: 10px;
    }

    h3 {
        color: #666 !important;
        font-weight: 600 !important;
        margin-top: 15px !important;
    }

    .stSelectbox, .stSlider, .stMultiSelect {
        background-color: white;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3e9f5 0%, #f0f4fc 100%);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #333 !important;
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: #333 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        color: #333 !important;
        background-color: rgba(102, 126, 234, 0.08);
        padding: 10px 15px;
        border-radius: 8px;
        margin: 5px 0;
        transition: all 0.3s ease;
        border-left: 4px solid rgba(102, 126, 234, 0.3);
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: rgba(102, 126, 234, 0.15);
        transform: translateX(5px);
        border-left: 4px solid #667eea;
    }

    .stDataFrame {
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }

    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        background: white;
        padding: 15px;
    }

    .stInfo, .stWarning, .stSuccess {
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }

    .divider {
        border: 2px solid #c5d3ff;
        margin: 20px 0;
    }

    .insight-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%);
        padding: 15px 20px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
        font-size: 14px;
        line-height: 1.6;
        color: #444;
    }

    .insight-card strong {
        color: #667eea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add header banner
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #e8ecf5 0%, #d9e4f0 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        text-align: center;
        border: 2px solid rgba(102, 126, 234, 0.2);
    ">
        <h1 style="color: #333; margin: 0; font-size: 36px;">🍽️ Yelp Analytics Dashboard</h1>
        <p style="color: #666; font-size: 16px; margin: 10px 0 0 0;">Discover Insights from Reviews & Ratings</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============ DATA FOLDER PATH ============
DATA_FOLDER = Path("data")


# ============ CHECK IF PREPROCESSED DATA EXISTS ============
if not DATA_FOLDER.exists():
    st.error("❌ Data folder not found!")
    st.warning(
        "⚠️ Please run `python preprocess_data.py` first to generate the aggregated data files."
    )
    st.stop()


# ============ LOAD AGGREGATED DATA ============
@st.cache_data
def load_aggregated_data(filename):
    """Load a specific aggregated CSV file"""
    filepath = DATA_FOLDER / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    else:
        st.warning(f"File not found: {filename}")
        return pd.DataFrame()


@st.cache_data
def load_kpis():
    """Load KPI data"""
    return load_aggregated_data("agg_kpis.csv")


@st.cache_data
def load_filter_options():
    """Load filter options with duplicate removal"""
    options = load_aggregated_data("filter_options.csv")
    if len(options) > 0:
        # Parse the cities list
        cities_raw = options["cities"].iloc[0]
        
        # Handle both string and list types
        if isinstance(cities_raw, str):
            try:
                cities = eval(cities_raw)
            except:
                cities = []
        else:
            cities = cities_raw if cities_raw else []
        
        # DEBUG: Print to see what we're getting
        print(f"DEBUG: Raw cities = {cities}")
        
        # Remove duplicates using a dictionary (case-insensitive)
        cities_dict = {}
        for city in cities:
            if city:  # Skip empty strings
                city_clean = str(city).strip()
                city_key = city_clean.lower()
                # Only add if not already present
                if city_key not in cities_dict:
                    cities_dict[city_key] = city_clean
        
        # Get unique cities and sort
        unique_cities = sorted(list(cities_dict.values()))
        
        # DEBUG: Print to see what we're returning
        print(f"DEBUG: Unique cities = {unique_cities}")
        print(f"DEBUG: Removed {len(cities) - len(unique_cities)} duplicates")
        
        # Parse price tiers
        price_tiers_raw = options["price_tiers"].iloc[0]
        if isinstance(price_tiers_raw, str):
            try:
                price_tiers = eval(price_tiers_raw)
            except:
                price_tiers = []
        else:
            price_tiers = price_tiers_raw if price_tiers_raw else []
        
        return {
            "cities": unique_cities,
            "price_tiers": sorted(price_tiers) if price_tiers else [],
            "min_date": options["min_date"].iloc[0],
            "max_date": options["max_date"].iloc[0],
        }
    
    return {
        "cities": [],
        "price_tiers": [],
        "min_date": "2020-01-01",
        "max_date": "2024-12-31",
    }


# ============ FIX THE DROPDOWN DIRECTION ============
# In your create_simple_filter function, change the city selectbox section to:

def create_simple_filter(key_prefix, show_city=True, show_price=True):
    """Create a simple filter section using pre-loaded options"""
    st.markdown("#### 🔍 Filter Options")

    filters = {}
    cols = st.columns(4)

    if show_city:
        with cols[0]:
            cities = ["All"] + filter_opts.get("cities", [])
            
            # Add custom CSS to force dropdown direction
            st.markdown("""
                <style>
                [data-baseweb="select"] {
                    margin-bottom: 300px !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            filters["city"] = st.selectbox(
                "🏙️ City", 
                cities, 
                key=f"{key_prefix}_city"
            )

    if show_price:
        with cols[1]:
            price_tiers = filter_opts.get("price_tiers", [])
            filters["price_tiers"] = st.multiselect(
                "💰 Price Tier",
                price_tiers,
                default=price_tiers,
                key=f"{key_prefix}_price",
            )

    with cols[2]:
        filters["min_rating"], filters["max_rating"] = st.slider(
            "⭐ Rating Range", 1.0, 5.0, (1.0, 5.0), 0.5, key=f"{key_prefix}_rating"
        )

    with cols[3]:
        filters["sentiment"] = st.multiselect(
            "😊 Sentiment",
            ["Positive", "Negative"],
            default=["Positive", "Negative"],
            key=f"{key_prefix}_sent",
        )

    st.markdown("---")
    return filters

@st.cache_data
def load_main_data():
    """Load main data for filtering (lighter version)"""
    return load_aggregated_data("main_data.csv")


# Load filter options
filter_opts = load_filter_options()

# Sidebar navigation
st.sidebar.title("🍽️ Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Time & Trends",
        "Exploratory Analysis",
        "Cuisine Analysis",
        "Map Explorer",
        "Sentiment Analysis",
        "Value & Outliers",
        "Advanced Insights",
        "Comparisons",
        "Data Table",
    ],
)


# ============ HELPER FUNCTIONS ============
def display_insight(insight_text, icon="💡"):
    """Display an insight box with consistent styling"""
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>{icon} Insight:</strong> {insight_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_simple_filter(key_prefix, show_city=True, show_price=True):
    """Create a simple filter section using pre-loaded options"""
    st.markdown("#### 🔍 Filter Options")

    filters = {}
    cols = st.columns(4)

    if show_city:
        with cols[0]:
            cities = ["All"] + filter_opts.get("cities", [])
            filters["city"] = st.selectbox("🏙️ City", cities, key=f"{key_prefix}_city")

    if show_price:
        with cols[1]:
            price_tiers = filter_opts.get("price_tiers", [])
            filters["price_tiers"] = st.multiselect(
                "💰 Price Tier",
                price_tiers,
                default=price_tiers,
                key=f"{key_prefix}_price",
            )

    with cols[2]:
        filters["min_rating"], filters["max_rating"] = st.slider(
            "⭐ Rating Range", 1.0, 5.0, (1.0, 5.0), 0.5, key=f"{key_prefix}_rating"
        )

    with cols[3]:
        filters["sentiment"] = st.multiselect(
            "😊 Sentiment",
            ["Positive", "Negative"],
            default=["Positive", "Negative"],
            key=f"{key_prefix}_sent",
        )

    st.markdown("---")
    return filters


def apply_filters_to_df(
    df,
    filters,
    city_col="city",
    rating_col="avg_rating",
    price_col="price_tier",
    sentiment_col=None,
):
    """Apply filters to a dataframe"""
    filtered = df.copy()

    if "city" in filters and filters["city"] != "All" and city_col in filtered.columns:
        filtered = filtered[filtered[city_col] == filters["city"]]

    if "price_tiers" in filters and price_col in filtered.columns:
        if filters["price_tiers"]:
            filtered = filtered[filtered[price_col].isin(filters["price_tiers"])]

    if (
        "min_rating" in filters
        and "max_rating" in filters
        and rating_col in filtered.columns
    ):
        filtered = filtered[
            (filtered[rating_col] >= filters["min_rating"])
            & (filtered[rating_col] <= filters["max_rating"])
        ]

    if sentiment_col and "sentiment" in filters and sentiment_col in filtered.columns:
        if filters["sentiment"]:
            filtered = filtered[filtered[sentiment_col].isin(filters["sentiment"])]

    return filtered


def get_insights_from_kpis(kpis):
    """Generate insights from KPI data"""
    insights = []
    if len(kpis) == 0:
        return ["No data available."]

    k = kpis.iloc[0]

    corr = k.get("correlation", 0)
    if corr > 0.7:
        insights.append(
            f"✅ Strong consistency: Sentiment and ratings are highly aligned (r={corr:.3f})"
        )
    elif corr < 0.4:
        insights.append(
            f"⚠️ Weak consistency: Reviews may not accurately reflect star ratings (r={corr:.3f})"
        )
    else:
        insights.append(
            f"📊 Moderate consistency: Sentiment and ratings show reasonable alignment (r={corr:.3f})"
        )

    insights.append(
        f"🏆 Total of {int(k.get('total_restaurants', 0)):,} restaurants with {int(k.get('total_reviews', 0)):,} reviews"
    )
    insights.append(
        f"⭐ Average rating: {k.get('avg_rating', 0):.2f}★ (Median: {k.get('median_rating', 0):.2f}★)"
    )
    insights.append(
        f"😊 {k.get('positive_pct', 0):.1f}% of reviews have positive sentiment"
    )
    insights.append(f"🏙️ Data covers {int(k.get('unique_cities', 0))} unique cities")

    return insights
def filter_out_non_cuisines(df, cuisine_col='cuisine'):
    """
    Remove non-food businesses that shouldn't appear as cuisines
    """
    if len(df) == 0 or cuisine_col not in df.columns:
        return df
    
    # List of terms to exclude (case-insensitive)
    exclude_terms = [
        'nutritionist',
        'nutritionists',
        'weight loss',
        'weight loss center',
        'weight loss centers',
        'health & medical',
        'health markets',
        'grocery',
        'farmers market',
        'specialty food',
        'imported food',
        'ethnic grocery',
        'butcher',
        'seafood markets',
        'cannabis',
        'active life',
        'arts & entertainment',
        'professional services',
        'event planning',
        'church',
        'churches',
        'temple',
        'school',
        'college',
        'sports club',
        'gym',
        'fitness',
        'yoga',
        'pilates',
        'spa',
        'salon',
        'massage',
        'beauty',
        'hair salon',
        'barber',
        'hotel',
        'motel',
    ]
    
    # Filter out rows where cuisine contains any of the exclude terms
    mask = df[cuisine_col].str.lower().apply(
        lambda x: not any(term in str(x).lower() for term in exclude_terms)
    )
    
    return df[mask]


# ============ END OF PART 1 ============
# ============ PAGE: OVERVIEW ============
if page == "Overview":
    st.title("📊 Overview")

    # Add filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        filters = create_simple_filter("overview", show_city=True, show_price=True)

    # Load pre-aggregated data
    kpis = load_kpis()
    ts_data = load_aggregated_data("agg_monthly_trends.csv")
    rating_dist = load_aggregated_data("agg_rating_distribution.csv")
    sentiment_dist = load_aggregated_data("agg_sentiment_distribution.csv")
    rating_cat = load_aggregated_data("agg_rating_category.csv")

    # Apply filters to time series data
    if len(ts_data) > 0 and "city" in filters:
        main_data = load_main_data()
        if len(main_data) > 0:
            filtered_main = apply_filters_to_df(
                main_data,
                filters,
                city_col="city",
                rating_col="stars",
                price_col="price_tier",
                sentiment_col="sentiment_text",
            )

            # Recalculate KPIs from filtered data
            if len(filtered_main) > 0:
                total_rest = filtered_main["business_id"].nunique()
                total_rev = len(filtered_main)
                avg_rat = filtered_main["stars"].mean()
                pos_pct = (
                    filtered_main["sentiment_label"].sum() / len(filtered_main) * 100
                )

                # Update KPIs display
                kpis = pd.DataFrame(
                    [
                        {
                            "total_restaurants": total_rest,
                            "total_reviews": total_rev,
                            "avg_rating": avg_rat,
                            "positive_pct": pos_pct,
                            "median_rating": filtered_main["stars"].median(),
                            "correlation": filtered_main[["stars", "sentiment_label"]]
                            .corr()
                            .iloc[0, 1],
                            "unique_cities": filtered_main["city"].nunique(),
                        }
                    ]
                )

    # KPI Cards
    if len(kpis) > 0:
        k = kpis.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 100%);
                    padding: 25px; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 16px rgba(167, 139, 250, 0.2); color: white;">
                    <div style="font-size: 32px; margin-bottom: 5px;">🏪</div>
                    <div style="font-size: 28px; font-weight: bold;">{int(k["total_restaurants"]):,}</div>
                    <div style="font-size: 12px; opacity: 0.9;">Restaurants</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #fda4af 0%, #f9a8d4 100%);
                    padding: 25px; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 16px rgba(253, 164, 175, 0.2); color: white;">
                    <div style="font-size: 32px; margin-bottom: 5px;">⭐</div>
                    <div style="font-size: 28px; font-weight: bold;">{int(k["total_reviews"]):,}</div>
                    <div style="font-size: 12px; opacity: 0.9;">Total Reviews</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #93c5fd 0%, #7dd3fc 100%);
                    padding: 25px; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 16px rgba(147, 197, 253, 0.2); color: white;">
                    <div style="font-size: 32px; margin-bottom: 5px;">📈</div>
                    <div style="font-size: 28px; font-weight: bold;">{k["avg_rating"]:.2f}</div>
                    <div style="font-size: 12px; opacity: 0.9;">Avg Rating</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #fde68a 0%, #fbbf24 100%);
                    padding: 25px; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 16px rgba(253, 230, 138, 0.2); color: white;">
                    <div style="font-size: 32px; margin-bottom: 5px;">😊</div>
                    <div style="font-size: 28px; font-weight: bold;">{k["positive_pct"]:.1f}%</div>
                    <div style="font-size: 12px; opacity: 0.9;">Positive</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col5:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #c4b5fd 0%, #e9d5ff 100%);
                    padding: 25px; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 16px rgba(196, 181, 253, 0.2); color: white;">
                    <div style="font-size: 32px; margin-bottom: 5px;">🔗</div>
                    <div style="font-size: 28px; font-weight: bold;">{k["correlation"]:.3f}</div>
                    <div style="font-size: 12px; opacity: 0.9;">Correlation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Key Insights
    st.markdown("<h2>💡 Key Insights</h2>", unsafe_allow_html=True)
    for i, insight in enumerate(get_insights_from_kpis(kpis)):
        colors = ["#a78bfa", "#fda4af", "#93c5fd", "#fbbf24", "#c4b5fd"]
        color = colors[i % len(colors)]
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                padding: 20px; border-radius: 12px; border-left: 5px solid {color};
                margin: 10px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <p style="margin: 0; color: #333; font-size: 15px; font-weight: 500;">{insight}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>📊 Rating & Review Trends</h3>", unsafe_allow_html=True)
        if len(ts_data) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=ts_data["year_month"],
                    y=ts_data["avg_stars"],
                    name="Avg Rating",
                    mode="lines+markers",
                    line=dict(color="#a78bfa", width=3),
                    marker=dict(size=6, color="#a78bfa")
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Bar(
                    x=ts_data["year_month"],
                    y=ts_data["review_count"],
                    name="Review Count",
                    marker=dict(color="rgba(147, 197, 253, 0.3)"),
                ),
                secondary_y=True,
            )
            fig.update_layout(
                title="Rating & Review Trends Over Time",
                height=400,
                hovermode="x unified",
            )
            fig.update_yaxes(title_text="Avg Rating", secondary_y=False)
            fig.update_yaxes(title_text="Review Count", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            trend = (
                "upward"
                if ts_data["avg_stars"].iloc[-1] > ts_data["avg_stars"].iloc[0]
                else "downward"
            )
            peak_idx = ts_data["review_count"].idxmax()
            display_insight(
                f"Rating trend is <strong>{trend}</strong>, from {ts_data['avg_stars'].iloc[0]:.2f}★ to {ts_data['avg_stars'].iloc[-1]:.2f}★. "
                f"Peak activity: <strong>{ts_data.loc[peak_idx, 'year_month']}</strong> with {ts_data['review_count'].max():,} reviews.",
                "📈",
            )

    with col2:
        st.markdown("<h3>⭐ Rating Distribution</h3>", unsafe_allow_html=True)
        if len(rating_dist) > 0:
            # Create gradient from pink to purple to blue
            fig = px.bar(
                rating_dist,
                x="stars",
                y="count",
                title="Rating Distribution",
                labels={"stars": "Star Rating", "count": "Count"},
                color="stars",
                color_continuous_scale=[[0, '#fda4af'], [0.5, '#a78bfa'], [1, '#93c5fd']]
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_traces(marker=dict(line=dict(width=0)))
            st.plotly_chart(fig, use_container_width=True)

            mode_rating = rating_dist.loc[rating_dist["count"].idxmax(), "stars"]
            display_insight(
                f"Most common rating: <strong>{mode_rating}★</strong>. "
                f"Distribution shows customer rating patterns - left-skewed distributions indicate satisfaction bias.",
                "⭐",
            )

    # Charts Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<h3>😊 Sentiment Distribution</h3>", unsafe_allow_html=True)
        if len(sentiment_dist) > 0:
            fig = px.pie(
                sentiment_dist,
                values="count",
                names="sentiment",
                title="Overall Sentiment",
                color_discrete_map={"Positive": "#fbbf24", "Negative": "#fda4af"},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            pos_row = sentiment_dist[sentiment_dist["sentiment"] == "Positive"]
            pos_pct = (
                (pos_row["count"].values[0] / sentiment_dist["count"].sum() * 100)
                if len(pos_row) > 0
                else 0
            )
            display_insight(
                f"<strong>{pos_pct:.1f}%</strong> positive sentiment. "
                f"{'High positivity indicates overall satisfaction.' if pos_pct > 70 else 'Mixed sentiment suggests variable experiences.'}",
                "😊",
            )

    with col4:
        st.markdown("<h3>🎯 Rating Categories</h3>", unsafe_allow_html=True)
        if len(rating_cat) > 0:
            # Sort categories in logical order
            category_order = ["Poor", "Average", "Good", "Excellent"]
            rating_cat['rating_category'] = pd.Categorical(
                rating_cat['rating_category'], 
                categories=category_order, 
                ordered=True
            )
            rating_cat_sorted = rating_cat.sort_values('rating_category')
            
            fig = px.bar(
                rating_cat_sorted,
                x="rating_category",
                y="count",
                title="Rating Category Distribution",
                color="rating_category",
                color_discrete_map={
                    "Poor": "#fda4af",
                    "Average": "#f9a8d4",
                    "Good": "#a78bfa",
                    "Excellent": "#93c5fd",
                },
            )
            fig.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

            total = rating_cat["count"].sum()
            excellent = rating_cat[rating_cat["rating_category"] == "Excellent"][
                "count"
            ].values
            excellent_pct = (excellent[0] / total * 100) if len(excellent) > 0 else 0
            display_insight(
                f"<strong>{excellent_pct:.1f}%</strong> of reviews are 'Excellent' (5★). "
                f"Focus on converting 'Good' to 'Excellent' for growth.",
                "🎯",
            )
# ============ PAGE: TIME & TRENDS (FIXED VERSION) ============
elif page == "Time & Trends":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #e8ecf5 0%, #d9e4f0 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1); text-align: center;
            border: 2px solid rgba(102, 126, 234, 0.2);">
            <h1 style="color: #333; margin: 0;">📈 Time & Trends Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        filters = create_simple_filter("time", show_city=True, show_price=True)

    # Load main data and apply filters
    main_data = load_main_data()
    
    if len(main_data) > 0:
        # Apply filters to main data
        filtered_data = apply_filters_to_df(
            main_data,
            filters,
            city_col="city",
            rating_col="stars",
            price_col="price_tier",
            sentiment_col="sentiment_text",
        )
        
        if len(filtered_data) > 0:
            # Ensure date column is datetime
            if 'date' not in filtered_data.columns and 'date' in filtered_data.columns:
                filtered_data['date'] = pd.to_datetime(filtered_data['date'])
            
            # Recalculate weekly data from filtered data
            if 'week' in filtered_data.columns:
                weekly_data = (
                    filtered_data.groupby("week")
                    .agg({
                        "stars": ["mean", "std", "count"]
                    })
                    .reset_index()
                )
                weekly_data.columns = ["week", "avg_stars", "std_stars", "review_count"]
            else:
                weekly_data = pd.DataFrame()
            
            # Recalculate daily sentiment from filtered data
            if 'date' in filtered_data.columns:
                filtered_data['date'] = pd.to_datetime(filtered_data['date'])
                daily_sentiment = (
                    filtered_data.groupby(filtered_data['date'].dt.date)
                    .agg({
                        "sentiment_label": "mean"
                    })
                    .reset_index()
                )
                daily_sentiment.columns = ["date", "avg_sentiment"]
            else:
                daily_sentiment = pd.DataFrame()
            
            # Recalculate seasonal data from filtered data
            if 'month_name' in filtered_data.columns:
                seasonal = (
                    filtered_data.groupby("month_name")
                    .agg({
                        "stars": ["mean", "count"]
                    })
                    .reset_index()
                )
                seasonal.columns = ["month_name", "avg_stars", "review_count"]
                
                # Sort by month order
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                seasonal['month_name'] = pd.Categorical(seasonal['month_name'], 
                                                        categories=month_order, 
                                                        ordered=True)
                seasonal = seasonal.sort_values('month_name')
            else:
                seasonal = pd.DataFrame()
            
            # Recalculate day of week stats from filtered data
            if 'day_of_week' in filtered_data.columns:
                day_stats = (
                    filtered_data.groupby("day_of_week")
                    .agg({
                        "stars": ["mean", "count"]
                    })
                    .reset_index()
                )
                day_stats.columns = ["day_of_week", "avg_stars", "review_count"]
                
                # Sort by day order
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_stats['day_of_week'] = pd.Categorical(day_stats['day_of_week'], 
                                                          categories=day_order, 
                                                          ordered=True)
                day_stats = day_stats.sort_values('day_of_week')
            else:
                day_stats = pd.DataFrame()
            
            # Recalculate hourly data from filtered data
            if 'hour' in filtered_data.columns:
                hourly = (
                    filtered_data.groupby("hour")
                    .agg({
                        "stars": ["mean", "count"]
                    })
                    .reset_index()
                )
                hourly.columns = ["hour", "avg_stars", "review_count"]
            else:
                hourly = pd.DataFrame()
        else:
            # No data after filtering
            weekly_data = pd.DataFrame()
            daily_sentiment = pd.DataFrame()
            seasonal = pd.DataFrame()
            day_stats = pd.DataFrame()
            hourly = pd.DataFrame()
    else:
        # Fallback to pre-aggregated data if main_data not available
        weekly_data = load_aggregated_data("agg_weekly_rating.csv")
        daily_sentiment = load_aggregated_data("agg_daily_sentiment.csv")
        seasonal = load_aggregated_data("agg_seasonal.csv")
        day_stats = load_aggregated_data("agg_day_of_week.csv")
        hourly = load_aggregated_data("agg_hourly.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>📊 Weekly Rating Trend</h3>", unsafe_allow_html=True)
        if len(weekly_data) > 0:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=weekly_data["week"],
                    y=weekly_data["avg_stars"],
                    mode="lines+markers",
                    name="Avg Rating",
                    line=dict(color="#a78bfa", width=2),
                    marker=dict(size=6, color="#a78bfa")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=weekly_data["week"],
                    y=weekly_data["avg_stars"] + weekly_data["std_stars"],
                    fill=None,
                    mode="lines",
                    line_color="rgba(0,0,0,0)",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=weekly_data["week"],
                    y=weekly_data["avg_stars"] - weekly_data["std_stars"],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(0,0,0,0)",
                    fillcolor="rgba(167, 139, 250, 0.2)",
                    name="Std Dev Band",
                )
            )
            fig.update_layout(title="Weekly Average Rating", height=450, hovermode="x")
            st.plotly_chart(fig, use_container_width=True)

            avg_std = weekly_data["std_stars"].mean()
            volatility = (
                "high" if avg_std > 1.2 else "moderate" if avg_std > 0.8 else "low"
            )
            display_insight(
                f"Average volatility: <strong>{avg_std:.2f}</strong> ({volatility}). "
                f"{'Wide bands = inconsistent experiences.' if volatility == 'high' else 'Narrow bands = reliable quality.'}",
                "📊",
            )
        else:
            st.info("No data available for the selected filters.")

    with col2:
        st.markdown("<h3>📈 Weekly Review Volume</h3>", unsafe_allow_html=True)
        if len(weekly_data) > 0:
            fig = px.bar(
                weekly_data,
                x="week",
                y="review_count",
                title="Weekly Review Volume",
                color_discrete_sequence=["#93c5fd"],
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            avg_vol = weekly_data["review_count"].mean()
            display_insight(
                f"Average: <strong>{avg_vol:.0f}</strong> reviews/week. "
                f"Volume spikes may indicate promotions or viral moments.",
                "📈",
            )
        else:
            st.info("No data available for the selected filters.")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<h3>💭 Sentiment Trend</h3>", unsafe_allow_html=True)
        if len(daily_sentiment) > 0:
            daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
            daily_sentiment["rolling_avg"] = (
                daily_sentiment["avg_sentiment"].rolling(7, min_periods=1).mean()
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=daily_sentiment["date"],
                    y=daily_sentiment["avg_sentiment"],
                    mode="markers",
                    name="Daily",
                    opacity=0.4,
                    marker=dict(color="#93c5fd", size=4),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=daily_sentiment["date"],
                    y=daily_sentiment["rolling_avg"],
                    mode="lines",
                    name="7-day MA",
                    line=dict(color="#fbbf24", width=3),
                )
            )
            fig.update_layout(
                title="Daily Sentiment with 7-day MA", height=400, hovermode="x"
            )
            st.plotly_chart(fig, use_container_width=True)

            recent = (
                daily_sentiment["rolling_avg"].iloc[-7:].mean()
                if len(daily_sentiment) >= 7
                else daily_sentiment["rolling_avg"].mean()
            )
            display_insight(
                f"Recent sentiment: <strong>{recent * 100:.1f}%</strong> positive. "
                f"Moving average smooths daily noise to reveal true trends.",
                "💭",
            )
        else:
            st.info("No data available for the selected filters.")

    with col4:
        st.markdown("<h3>🌍 Seasonal Patterns</h3>", unsafe_allow_html=True)
        if len(seasonal) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=seasonal["month_name"],
                    y=seasonal["avg_stars"],
                    mode="lines+markers",
                    name="Avg Rating",
                    line=dict(color="#a78bfa", width=3),
                    marker=dict(size=8, color="#a78bfa")
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Bar(
                    x=seasonal["month_name"],
                    y=seasonal["review_count"],
                    name="Review Count",
                    marker=dict(color="rgba(147, 197, 253, 0.3)"),
                ),
                secondary_y=True,
            )
            fig.update_layout(title="Seasonal Pattern by Month", height=400)
            fig.update_yaxes(title_text="Avg Rating", secondary_y=False)
            fig.update_yaxes(title_text="Review Count", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            best_month = seasonal.loc[seasonal["avg_stars"].idxmax(), "month_name"]
            display_insight(
                f"Best month: <strong>{best_month}</strong> ({seasonal['avg_stars'].max():.2f}★). "
                f"Plan promotions during historically weaker months.",
                "🌍",
            )
        else:
            st.info("No data available for the selected filters.")

    # Day of Week Analysis
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("<h3>📅 Rating by Day of Week</h3>", unsafe_allow_html=True)
        if len(day_stats) > 0:
            # Create gradient color mapping from pink to purple to blue
            fig = px.bar(
                day_stats,
                x="day_of_week",
                y="avg_stars",
                title="Average Rating by Day",
                color="avg_stars",
                color_continuous_scale=[[0, '#fda4af'], [0.5, '#c4b5fd'], [1, '#93c5fd']],
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            best_day = day_stats.loc[day_stats["avg_stars"].idxmax(), "day_of_week"]
            display_insight(
                f"Best day: <strong>{best_day}</strong> ({day_stats['avg_stars'].max():.2f}★).",
                "📅",
            )
        else:
            st.info("No data available for the selected filters.")

    with col6:
        st.markdown("<h3>🗓️ Volume by Day of Week</h3>", unsafe_allow_html=True)
        if len(day_stats) > 0:
            # Create gradient effect based on volume
            fig = px.bar(
                day_stats,
                x="day_of_week",
                y="review_count",
                title="Review Count by Day",
                color='review_count',
                color_continuous_scale=[[0, '#fda4af'], [0.33, '#e9d5ff'], [0.66, '#a78bfa'], [1, '#93c5fd']],
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            busiest = day_stats.loc[day_stats["review_count"].idxmax(), "day_of_week"]
            display_insight(
                f"Busiest day: <strong>{busiest}</strong> ({day_stats['review_count'].max():,} reviews).",
                "🗓️",
            )
        else:
            st.info("No data available for the selected filters.")
# ============ END OF TIME & TRENDS PAGE ============
# ============ PAGE: EXPLORATORY ANALYSIS (FIXED VERSION) ============
elif page == "Exploratory Analysis":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f0e5f5 0%, #e8d9f0 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(245, 87, 108, 0.08); text-align: center;
            border: 2px solid rgba(245, 87, 108, 0.15);">
            <h1 style="color: #333; margin: 0;">📊 Exploratory Data Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        filters = create_simple_filter("eda", show_city=True, show_price=True)

    # Load main data and apply filters
    main_data = load_main_data()
    
    if len(main_data) > 0:
        # Apply filters to main data
        filtered_data = apply_filters_to_df(
            main_data,
            filters,
            city_col="city",
            rating_col="stars",
            price_col="price_tier",
            sentiment_col="sentiment_text",
        )
        
        if len(filtered_data) > 0:
            # Recalculate KPIs from filtered data
            total_rest = filtered_data["business_id"].nunique()
            total_rev = len(filtered_data)
            avg_rat = filtered_data["stars"].mean()
            median_rat = filtered_data["stars"].median()
            pos_pct = (filtered_data["sentiment_label"].sum() / len(filtered_data) * 100)
            corr = filtered_data[["stars", "sentiment_label"]].corr().iloc[0, 1]
            unique_cities = filtered_data["city"].nunique()
            
            kpis = pd.DataFrame([{
                "total_restaurants": total_rest,
                "total_reviews": total_rev,
                "avg_rating": avg_rat,
                "median_rating": median_rat,
                "positive_pct": pos_pct,
                "correlation": corr,
                "unique_cities": unique_cities,
            }])
            
            # Recalculate rating distribution
            rating_dist = (
                filtered_data.groupby("stars")
                .size()
                .reset_index(name="count")
            )
            
            # Recalculate price stats
            if filtered_data["price_tier"].notna().any():
                price_stats = (
                    filtered_data.groupby("price_tier")
                    .agg({
                        "stars": ["mean", "std", "count"]
                    })
                    .reset_index()
                )
                price_stats.columns = ["price_tier", "avg_stars", "std_stars", "review_count"]
            else:
                price_stats = pd.DataFrame()
            
            # Recalculate price by city
            if filtered_data["price_tier"].notna().any():
                price_city = (
                    filtered_data.groupby(["city", "price_tier"])
                    .size()
                    .reset_index(name="count")
                )
                # Keep only top cities by total count
                top_cities = price_city.groupby("city")["count"].sum().nlargest(10).index
                price_city = price_city[price_city["city"].isin(top_cities)]
            else:
                price_city = pd.DataFrame()
            
            # Recalculate restaurant summary
            restaurant_summary = (
                filtered_data.groupby(["business_id", "name", "city"])
                .agg({
                    "stars": "mean",
                    "sentiment_label": "mean",
                    "price_tier": "median",
                    "review_id": "count",
                    "business_stars": "first",
                })
                .reset_index()
            )
            restaurant_summary.columns = [
                "business_id", "name", "city", "avg_rating", 
                "positive_sentiment_pct", "median_price", 
                "reviews_in_dataset", "yelp_rating"
            ]
            restaurant_summary["positive_sentiment_pct"] *= 100
            restaurant_summary = restaurant_summary.sort_values("avg_rating", ascending=False)
            
        else:
            # No data after filtering
            kpis = pd.DataFrame()
            rating_dist = pd.DataFrame()
            price_stats = pd.DataFrame()
            price_city = pd.DataFrame()
            restaurant_summary = pd.DataFrame()
    else:
        # Fallback to pre-aggregated data
        kpis = load_kpis()
        rating_dist = load_aggregated_data("agg_rating_distribution.csv")
        price_stats = load_aggregated_data("agg_price_stats.csv")
        price_city = load_aggregated_data("agg_price_by_city.csv")
        restaurant_summary = load_aggregated_data("agg_restaurant_summary.csv")
        
        # Apply filters to restaurant summary if available
        if len(restaurant_summary) > 0:
            restaurant_summary = apply_filters_to_df(
                restaurant_summary,
                filters,
                city_col="city",
                rating_col="avg_rating",
                price_col="median_price",
            )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>📊 Rating Distribution</h3>", unsafe_allow_html=True)
        if len(rating_dist) > 0 and len(kpis) > 0:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=rating_dist["stars"],
                    y=rating_dist["count"],
                    marker=dict(color="#a78bfa"),
                )
            )
            fig.add_vline(
                x=kpis.iloc[0]["avg_rating"],
                line_dash="dash",
                line_color="#fda4af",
                annotation_text=f"Mean: {kpis.iloc[0]['avg_rating']:.2f}",
            )
            fig.update_layout(
                title="Rating Distribution with Mean Line",
                height=450,
                xaxis_title="Star Rating",
                yaxis_title="Frequency",
            )
            st.plotly_chart(fig, use_container_width=True)

            display_insight(
                f"Mean: <strong>{kpis.iloc[0]['avg_rating']:.2f}★</strong>, "
                f"Median: <strong>{kpis.iloc[0]['median_rating']:.2f}★</strong>. "
                f"Ratings above the pink line exceed average performance.",
                "📊",
            )
        else:
            st.info("No data available for the selected filters.")

    with col2:
        st.markdown("<h3>💰 Rating by Price Tier</h3>", unsafe_allow_html=True)
        if len(price_stats) > 0:
            fig = px.bar(
                price_stats,
                x="price_tier",
                y="avg_stars",
                error_y="std_stars",
                title="Average Rating by Price Tier",
                labels={"price_tier": "Price Tier ($)", "avg_stars": "Avg Rating"},
                color="price_tier",
                color_continuous_scale=[[0, '#fda4af'], [0.5, '#a78bfa'], [1, '#93c5fd']],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            best_tier = price_stats.loc[price_stats["avg_stars"].idxmax(), "price_tier"]
            display_insight(
                f"Best-rated tier: <strong>${int(best_tier)}</strong> ({price_stats['avg_stars'].max():.2f}★). "
                f"Error bars show rating consistency within each tier.",
                "💰",
            )
        else:
            st.info("No data available for the selected filters.")

    # Price Statistics
    if len(price_stats) > 0:
        col3, col4, col5 = st.columns(3)
        with col3:
            best = price_stats.loc[price_stats["avg_stars"].idxmax()]
            st.metric(
                "Highest Rated Price Tier",
                f"${int(best['price_tier'])} ({best['avg_stars']:.2f}★)",
            )
        with col4:
            most_reviews = price_stats.loc[price_stats["review_count"].idxmax()]
            st.metric("Most Reviews Price Tier", f"${int(most_reviews['price_tier'])}")
        with col5:
            spread = price_stats["avg_stars"].max() - price_stats["avg_stars"].min()
            st.metric("Rating Spread Across Tiers", f"{spread:.2f}★")

    st.divider()

    # Price Mix by City
    st.markdown("<h3>🏙️ Price Tier Mix by City</h3>", unsafe_allow_html=True)
    if len(price_city) > 0:
        fig = px.bar(
            price_city,
            x="city",
            y="count",
            color="price_tier",
            title="Price Tier Distribution by Top Cities",
            barmode="stack",
            color_discrete_sequence=["#fda4af", "#f9a8d4", "#c4b5fd", "#a78bfa", "#93c5fd"],
        )
        st.plotly_chart(fig, use_container_width=True)

        display_insight(
            f"Cities with diverse price tiers offer options for all budgets. "
            f"Homogeneous markets may have gaps in the dining landscape.",
            "🏙️",
        )
    else:
        st.info("No data available for the selected filters.")

    # Top Performing Restaurants
    st.markdown("<h3>🏆 Top Performing Restaurants</h3>", unsafe_allow_html=True)
    if len(restaurant_summary) > 0:
        display_cols = [
            "name",
            "city",
            "avg_rating",
            "positive_sentiment_pct",
            "median_price",
            "reviews_in_dataset",
            "yelp_rating",
        ]
        st.dataframe(
            restaurant_summary[display_cols].head(30).round(2),
            use_container_width=True,
        )

        top = restaurant_summary.iloc[0]
        display_insight(
            f"Top performer: <strong>{top['name']}</strong> in {top['city']} "
            f"({top['avg_rating']:.2f}★, {top['positive_sentiment_pct']:.1f}% positive).",
            "🏆",
        )
    else:
        st.info("No restaurants match the selected filters.")
# ============ CUISINE ANALYSIS PAGE ============
elif page == "Cuisine Analysis":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #e1f0ff 0%, #d9ebf7 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(79, 172, 254, 0.08); text-align: center;
            border: 2px solid rgba(79, 172, 254, 0.15);">
            <h1 style="color: #333; margin: 0;">🍜 Cuisine Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        filters = create_simple_filter("cuisine", show_city=True, show_price=True)

    # Load main data and apply filters
    main_data = load_main_data()
    
    if len(main_data) > 0:
        # Apply filters to main data
        filtered_data = apply_filters_to_df(
            main_data,
            filters,
            city_col="city",
            rating_col="stars",
            price_col="price_tier",
            sentiment_col="sentiment_text",
        )
        
        if len(filtered_data) > 0 and 'categories' in filtered_data.columns:
            # Explode categories to individual cuisines
            filtered_data['cuisine_list'] = filtered_data['categories'].str.split(',')
            df_cuisine_exploded = filtered_data.explode('cuisine_list')
            df_cuisine_exploded['cuisine_list'] = df_cuisine_exploded['cuisine_list'].str.strip()
            
            # Filter out generic terms AND non-food businesses
            exclude_terms = ['Restaurants', 'Food', 'Event Planning & Services']
            df_cuisine_exploded = df_cuisine_exploded[
                ~df_cuisine_exploded['cuisine_list'].isin(exclude_terms)
            ]
            
            # Additional filtering for non-cuisines (case-insensitive substring match)
            non_cuisine_terms = [
                'nutritionist', 'weight loss', 'health & medical',
                'grocery', 'farmers market', 'specialty food',
                'butcher', 'seafood markets', 'gym', 'fitness',
                'yoga', 'spa', 'salon', 'church', 'school',
            ]
            df_cuisine_exploded = df_cuisine_exploded[
                ~df_cuisine_exploded['cuisine_list'].str.lower().apply(
                    lambda x: any(term in str(x).lower() for term in non_cuisine_terms)
                )
            ]
            
            # Recalculate cuisine rating
            cuisine_rating = (
                df_cuisine_exploded.groupby("cuisine_list")
                .agg({
                    "stars": ["mean", "std", "count"]
                })
                .reset_index()
            )
            cuisine_rating.columns = ["cuisine", "avg_rating", "std_rating", "count"]
            cuisine_rating = cuisine_rating[cuisine_rating["count"] >= 10]  # Min 10 reviews
            cuisine_rating = cuisine_rating.sort_values("avg_rating", ascending=False)
            
            # Recalculate cuisine count
            cuisine_count = (
                df_cuisine_exploded.groupby("cuisine_list")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            cuisine_count.columns = ["cuisine", "count"]
            
            # Recalculate heatmap data (cuisine × price tier)
            if df_cuisine_exploded["price_tier"].notna().any():
                heatmap_data = (
                    df_cuisine_exploded.groupby(["cuisine_list", "price_tier"])
                    .agg({"stars": "mean"})
                    .reset_index()
                )
                heatmap_data.columns = ["cuisine", "price_tier", "avg_rating"]
                # Keep only top cuisines
                top_cuisines = cuisine_count.head(15)["cuisine"].tolist()
                heatmap_data = heatmap_data[heatmap_data["cuisine"].isin(top_cuisines)]
            else:
                heatmap_data = pd.DataFrame()
            
            # Recalculate cuisine sentiment
            cuisine_sentiment = (
                df_cuisine_exploded.groupby("cuisine_list")
                .agg({
                    "sentiment_label": lambda x: (x.sum() / len(x)) * 100
                })
                .reset_index()
            )
            cuisine_sentiment.columns = ["cuisine", "positive_pct"]
            cuisine_sentiment = cuisine_sentiment.sort_values("positive_pct", ascending=False)
            
            # Recalculate cuisine trends over time
            if 'year_month' in df_cuisine_exploded.columns:
                # Get top 5 cuisines by volume
                top_5_cuisines = cuisine_count.head(5)["cuisine"].tolist()
                cuisine_trends_data = df_cuisine_exploded[
                    df_cuisine_exploded["cuisine_list"].isin(top_5_cuisines)
                ]
                
                cuisine_trends = (
                    cuisine_trends_data.groupby(["year_month", "cuisine_list"])
                    .agg({"stars": "mean"})
                    .reset_index()
                )
                cuisine_trends.columns = ["year_month", "cuisine", "avg_rating"]
            else:
                cuisine_trends = pd.DataFrame()
            
        else:
            # No data after filtering
            cuisine_rating = pd.DataFrame()
            cuisine_count = pd.DataFrame()
            heatmap_data = pd.DataFrame()
            cuisine_sentiment = pd.DataFrame()
            cuisine_trends = pd.DataFrame()
    else:
        # Fallback to pre-aggregated data
        cuisine_rating = load_aggregated_data("agg_cuisine_rating.csv")
        cuisine_count = load_aggregated_data("agg_cuisine_count.csv")
        heatmap_data = load_aggregated_data("agg_cuisine_price_heatmap.csv")
        cuisine_sentiment = load_aggregated_data("agg_cuisine_sentiment.csv")
        cuisine_trends = load_aggregated_data("agg_cuisine_trends.csv")
        
        # Apply filtering to pre-aggregated data too
        cuisine_rating = filter_out_non_cuisines(cuisine_rating, 'cuisine')
        cuisine_count = filter_out_non_cuisines(cuisine_count, 'cuisine')
        heatmap_data = filter_out_non_cuisines(heatmap_data, 'cuisine')
        cuisine_sentiment = filter_out_non_cuisines(cuisine_sentiment, 'cuisine')
        cuisine_trends = filter_out_non_cuisines(cuisine_trends, 'cuisine')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>🏆 Top Cuisines by Rating</h3>", unsafe_allow_html=True)
        if len(cuisine_rating) > 0:
            top_15 = cuisine_rating.head(15)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=top_15["cuisine"],
                    y=top_15["avg_rating"],
                    error_y=dict(type="data", array=top_15["std_rating"]),
                    marker=dict(
                        color=top_15["avg_rating"],
                        colorscale=[[0, '#fda4af'], [0.5, '#a78bfa'], [1, '#93c5fd']],
                        showscale=False,
                    ),
                )
            )
            fig.update_layout(
                title="Top Cuisines by Average Rating",
                height=450,
                xaxis_title="Cuisine",
                yaxis_title="Avg Rating",
            )
            st.plotly_chart(fig, use_container_width=True)

            top_c = top_15.iloc[0]
            display_insight(
                f"Top-rated: <strong>{top_c['cuisine']}</strong> ({top_c['avg_rating']:.2f}★). "
                f"Small error bars = consistent quality; large bars = hit-or-miss.",
                "🏆",
            )
        else:
            st.info("No data available for the selected filters.")

    with col2:
        st.markdown("<h3>📊 Top Cuisines by Volume</h3>", unsafe_allow_html=True)
        if len(cuisine_count) > 0:
            top_15 = cuisine_count.head(15)
            fig = px.bar(
                top_15,
                x="cuisine",
                y="count",
                title="Top Cuisines by Review Count",
                color="count",
                color_continuous_scale=[[0, '#fda4af'], [0.33, '#f9a8d4'], [0.66, '#a78bfa'], [1, '#93c5fd']],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            top_3 = cuisine_count.head(3)
            top_3_pct = top_3["count"].sum() / cuisine_count["count"].sum() * 100
            display_insight(
                f"Top 3: <strong>{', '.join(top_3['cuisine'].tolist())}</strong> = "
                f"<strong>{top_3_pct:.1f}%</strong> of reviews.",
                "📊",
            )
        else:
            st.info("No data available for the selected filters.")

    # Heatmap
    st.markdown("<h3>🔥 Cuisine × Price Tier Heatmap</h3>", unsafe_allow_html=True)
    if len(heatmap_data) > 0:
        heatmap_pivot = heatmap_data.pivot_table(
            index="cuisine", columns="price_tier", values="avg_rating"
        )
        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Price Tier ($)", y="Cuisine", color="Avg Rating"),
            color_continuous_scale=[[0, '#fda4af'], [0.5, '#fde68a'], [1, '#93c5fd']],
            aspect="auto",
            title="Average Rating Heatmap",
            zmin=2,
            zmax=5,
        )
        st.plotly_chart(fig, use_container_width=True)

        best = heatmap_data.loc[heatmap_data["avg_rating"].idxmax()]
        display_insight(
            f"Best combo: <strong>{best['cuisine']} at ${int(best['price_tier'])}</strong> ({best['avg_rating']:.2f}★). "
            f"Blue = value winners; Pink = expectation gaps.",
            "🔥",
        )
    else:
        st.info("No data available for the selected filters.")

    # Sentiment by Cuisine
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<h3>😊 Sentiment by Cuisine</h3>", unsafe_allow_html=True)
        if len(cuisine_sentiment) > 0:
            top_15 = cuisine_sentiment.head(15)
            fig = px.bar(
                top_15,
                x="cuisine",
                y="positive_pct",
                title="Sentiment by Cuisine (% Positive)",
                color="positive_pct",
                color_continuous_scale=[[0, '#fda4af'], [0.5, '#fbbf24'], [1, '#93c5fd']],
                text="positive_pct",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            display_insight(
                f"Most positive: <strong>{top_15.iloc[0]['cuisine']}</strong> ({top_15.iloc[0]['positive_pct']:.1f}%).",
                "😊",
            )
        else:
            st.info("No data available for the selected filters.")

    with col4:
        st.markdown("<h3>📈 Cuisine Trends Over Time</h3>", unsafe_allow_html=True)
        if len(cuisine_trends) > 0:
            # Get unique cuisines and assign pastel colors
            cuisines = cuisine_trends['cuisine'].unique()
            pastel_colors = ['#a78bfa', '#fda4af', '#93c5fd', '#fbbf24', '#f9a8d4']
            color_map = {cuisine: pastel_colors[i % len(pastel_colors)] for i, cuisine in enumerate(cuisines)}
            
            fig = px.line(
                cuisine_trends,
                x="year_month",
                y="avg_rating",
                color="cuisine",
                title="Top 5 Cuisines Rating Trends",
                markers=True,
                color_discrete_map=color_map,
            )
            st.plotly_chart(fig, use_container_width=True)

            display_insight(
                f"Track how cuisine ratings evolve. Converging lines = market maturation; "
                f"diverging = widening quality gaps.",
                "📈",
            )
        else:
            st.info("No data available for the selected filters.")

# ============ PAGE: MAP EXPLORER ============
elif page == "Map Explorer":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #e0f7f0 0%, #d9f0e8 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(67, 233, 123, 0.08); text-align: center;
            border: 2px solid rgba(67, 233, 123, 0.15);">
            <h1 style="color: #333; margin: 0;">🗺️ Map Explorer</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load pre-aggregated data
    map_data = load_aggregated_data("agg_map_data.csv")
    city_stats = load_aggregated_data("agg_city_stats.csv")

    # Filters
    st.markdown("#### 🔍 Map Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color_by = st.selectbox(
            "🎨 Color by:", ["Rating", "Sentiment", "Price"], key="map_color"
        )
    with col2:
        cities = (
            ["All"] + sorted(map_data["city"].dropna().unique().tolist())
            if len(map_data) > 0
            else ["All"]
        )
        filter_city = st.selectbox("🏙️ City:", cities, key="map_city")
    with col3:
        min_reviews = st.slider("📊 Min Reviews:", 0, 100, 0, key="map_reviews")
    with col4:
        min_rating, max_rating = st.slider(
            "⭐ Rating Range:", 1.0, 5.0, (1.0, 5.0), 0.5, key="map_rating"
        )

    st.markdown("---")

    # Filter map data
    if len(map_data) > 0:
        filtered_map = map_data.copy()
        if filter_city != "All":
            filtered_map = filtered_map[filtered_map["city"] == filter_city]
        filtered_map = filtered_map[filtered_map["review_count"] >= min_reviews]
        filtered_map = filtered_map[
            (filtered_map["business_stars"] >= min_rating)
            & (filtered_map["business_stars"] <= max_rating)
        ]

        st.markdown(f"**📊 Showing {len(filtered_map):,} restaurants**")

        # Create map
        if len(filtered_map) > 0:
            m = folium.Map(
                location=[
                    filtered_map["latitude"].mean(),
                    filtered_map["longitude"].mean(),
                ],
                zoom_start=11,
            )

            for _, row in filtered_map.iterrows():
                if color_by == "Rating":
                    color = (
                        "#93c5fd"  # blue for high
                        if row["business_stars"] >= 4
                        else "#fbbf24"  # yellow for medium
                        if row["business_stars"] >= 3
                        else "#fda4af"  # pink for low
                    )
                elif color_by == "Sentiment":
                    color = (
                        "#93c5fd"  # blue for high
                        if row["avg_sentiment"] >= 0.7
                        else "#fbbf24"  # yellow for medium
                        if row["avg_sentiment"] >= 0.4
                        else "#fda4af"  # pink for low
                    )
                else:
                    colors_map = {1: "#fda4af", 2: "#f9a8d4", 3: "#a78bfa", 4: "#93c5fd"}
                    color = (
                        colors_map.get(int(row["price_tier"]), "gray")
                        if pd.notna(row["price_tier"])
                        else "gray"
                    )

                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=min(max(5, row["review_count"] / 20), 15),
                    popup=f"<b>{row['name']}</b><br>{row['address']}<br>Rating: {row['business_stars']:.1f} | Reviews: {row['review_count']}<br>Sentiment: {row['avg_sentiment']:.1%}",
                    color=color,
                    fill=True,
                    fillOpacity=0.7,
                    weight=2,
                ).add_to(m)

            st_folium(m, width=None, height=500, use_container_width=True)

            display_insight(
                f"Showing <strong>{len(filtered_map):,}</strong> restaurants. "
                f"Size = review volume; Color = {color_by.lower()}. "
                f"Blue = high performance, Pink = needs attention.",
                "🗺️",
            )
        else:
            st.warning("No restaurants match the current filters.")

    st.divider()

    # City-Level Analysis
    st.markdown("<h3>🏙️ City-Level Analysis</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if len(city_stats) > 0:
            fig = px.bar(
                city_stats.head(15),
                x="city",
                y="positive_sentiment_pct",
                color="avg_rating",
                color_continuous_scale=[[0, '#fda4af'], [0.5, '#fbbf24'], [1, '#93c5fd']],
                title="City Sentiment & Rating",
                text="positive_sentiment_pct",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            best = city_stats.iloc[0]
            display_insight(
                f"Top city: <strong>{best['city']}</strong> ({best['positive_sentiment_pct']:.1f}% positive).",
                "🏙️",
            )

    with col2:
        if len(city_stats) > 0:
            fig = px.scatter(
                city_stats,
                x="median_price",
                y="avg_rating",
                size="review_count",
                color="positive_sentiment_pct",
                hover_name="city",
                color_continuous_scale=[[0, '#fda4af'], [0.5, '#fbbf24'], [1, '#93c5fd']],
                title="City: Price vs Rating",
                size_max=50,
            )
            st.plotly_chart(fig, use_container_width=True)

            display_insight(
                f"Bubble size = review volume; color = sentiment. "
                f"Top-right = premium high-quality markets.",
                "📍",
            )


# ============ END OF PART 3 ============
# ============ PAGE: SENTIMENT ANALYSIS ============
elif page == "Sentiment Analysis":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #e8ecf5 0%, #d9e4f0 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1); text-align: center;
            border: 2px solid rgba(102, 126, 234, 0.2);">
            <h1 style="color: #333; margin: 0;">💬 Sentiment Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        filters = create_simple_filter("sentiment", show_city=True, show_price=True)

    # Load pre-aggregated data
    kpis = load_kpis()
    rating_sentiment = load_aggregated_data("agg_rating_sentiment.csv")
    length_sentiment = load_aggregated_data("agg_length_sentiment.csv")
    sentiment_price = load_aggregated_data("agg_sentiment_price.csv")
    cuisine_sentiment = load_aggregated_data("agg_cuisine_sentiment.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>✅ Sentiment-Rating Correlation</h3>", unsafe_allow_html=True)
        if len(kpis) > 0:
            corr = kpis.iloc[0]["correlation"]
            consistency = (
                "excellent"
                if corr > 0.7
                else "good"
                if corr > 0.5
                else "moderate"
                if corr > 0.3
                else "weak"
            )

            # Create a gauge-like visualization
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=corr,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Sentiment-Rating Correlation"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "#a78bfa"},
                        "steps": [
                            {"range": [0, 0.3], "color": "#fda4af"},
                            {"range": [0.3, 0.5], "color": "#f9a8d4"},
                            {"range": [0.5, 0.7], "color": "#c4b5fd"},
                            {"range": [0.7, 1], "color": "#93c5fd"},
                        ],
                    },
                )
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            display_insight(
                f"Correlation: <strong>{corr:.3f}</strong> ({consistency} consistency). "
                f"{'Strong alignment = authentic reviews.' if corr > 0.5 else 'Weak alignment may indicate mixed opinions.'}",
                "✅",
            )

    with col2:
        st.markdown("<h3>💰 Sentiment by Price Tier</h3>", unsafe_allow_html=True)
        if len(sentiment_price) > 0:
            # Apply price filter
            if "price_tiers" in filters and filters["price_tiers"]:
                sentiment_price = sentiment_price[
                    sentiment_price["price_tier"].isin(filters["price_tiers"])
                ]

            fig = px.bar(
                sentiment_price,
                x="price_tier",
                y="avg_sentiment",
                error_y="std_sentiment",
                title="Sentiment by Price Tier",
                labels={
                    "price_tier": "Price Tier ($)",
                    "avg_sentiment": "Avg Sentiment",
                },
                color="price_tier",
                color_continuous_scale=[[0, '#fda4af'], [0.5, '#a78bfa'], [1, '#93c5fd']],
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            display_insight(
                f"Higher prices often show polarized sentiment—customers have stronger expectations "
                f"when paying more.",
                "💰",
            )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<h3>📊 Sentiment by Rating Category</h3>", unsafe_allow_html=True)
        if len(rating_sentiment) > 0:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=rating_sentiment["rating_category"],
                    y=rating_sentiment["avg_sentiment"] * 100,
                    error_y=dict(
                        type="data", array=rating_sentiment["std_sentiment"] * 100
                    ),
                    marker=dict(color=["#fda4af", "#f9a8d4", "#a78bfa", "#93c5fd"]),
                )
            )
            fig.update_layout(
                title="Sentiment by Rating Category",
                height=400,
                xaxis_title="Rating Category",
                yaxis_title="% Positive Sentiment",
            )
            st.plotly_chart(fig, use_container_width=True)

            display_insight(
                f"'Excellent' ratings should show high positive sentiment; 'Poor' should show low. "
                f"Mismatches suggest complex customer opinions.",
                "📊",
            )

    with col4:
        st.markdown("<h3>📝 Review Length by Sentiment</h3>", unsafe_allow_html=True)
        if len(length_sentiment) > 0:
            # Apply sentiment filter
            if "sentiment" in filters and filters["sentiment"]:
                length_sentiment = length_sentiment[
                    length_sentiment["sentiment"].isin(filters["sentiment"])
                ]

            fig = px.bar(
                length_sentiment,
                x="sentiment",
                y="mean_length",
                error_y="std_length",
                title="Average Review Length by Sentiment",
                labels={"sentiment": "Sentiment", "mean_length": "Avg Length (chars)"},
                color="sentiment",
                color_discrete_map={"Positive": "#93c5fd", "Negative": "#fda4af"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            pos_len = length_sentiment[length_sentiment["sentiment"] == "Positive"][
                "mean_length"
            ].values
            neg_len = length_sentiment[length_sentiment["sentiment"] == "Negative"][
                "mean_length"
            ].values
            if len(pos_len) > 0 and len(neg_len) > 0:
                longer = "Negative" if neg_len[0] > pos_len[0] else "Positive"
                display_insight(
                    f"<strong>{longer}</strong> reviews are longer on average. "
                    f"{'Dissatisfied customers write detailed complaints.' if longer == 'Negative' else 'Happy customers share enthusiastic experiences.'}",
                    "📝",
                )

    st.divider()

    # Sentiment by Cuisine
    st.markdown("<h3>🍜 Sentiment by Cuisine</h3>", unsafe_allow_html=True)
    if len(cuisine_sentiment) > 0:
        fig = px.bar(
            cuisine_sentiment.head(15),
            x="cuisine",
            y="positive_pct",
            title="Sentiment by Cuisine Type",
            text="positive_pct",
            color="positive_pct",
            color_continuous_scale=[[0, '#fda4af'], [0.5, '#fbbf24'], [1, '#93c5fd']],
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        display_insight(
            f"Top sentiment: <strong>{cuisine_sentiment.iloc[0]['cuisine']}</strong> "
            f"({cuisine_sentiment.iloc[0]['positive_pct']:.1f}% positive). "
            f"Low sentiment cuisines may have service or quality gaps.",
            "🍜",
        )


# ============ PAGE: VALUE & OUTLIERS ============
elif page == "Value & Outliers":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #fff5e8 0%, #fef0e0 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(250, 112, 154, 0.08); text-align: center;
            border: 2px solid rgba(250, 112, 154, 0.15);">
            <h1 style="color: #333; margin: 0;">💎 Value for Money & Outliers</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load pre-aggregated data
    restaurant_value = load_aggregated_data("agg_restaurant_value.csv")

    # Filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        filters = create_simple_filter("value", show_city=True, show_price=True)

    # Apply filters to restaurant value data
    if len(restaurant_value) > 0:
        restaurant_value = apply_filters_to_df(
            restaurant_value,
            filters,
            city_col="city",
            rating_col="dataset_avg_rating",
            price_col="median_price",
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h3>💎 Hidden Gems (High Rating, Low Price)</h3>", unsafe_allow_html=True
        )
        if len(restaurant_value) > 0:
            gems = restaurant_value[
                (restaurant_value["dataset_avg_rating"] >= 4.3)
                & (restaurant_value["median_price"] <= 2)
            ].sort_values("dataset_avg_rating", ascending=False)

            if len(gems) > 0:
                gems_display = gems[
                    [
                        "name",
                        "city",
                        "dataset_avg_rating",
                        "median_price",
                        "dataset_reviews",
                        "avg_sentiment",
                    ]
                ].copy()
                gems_display.columns = [
                    "Name",
                    "City",
                    "Avg Rating",
                    "Price",
                    "Reviews",
                    "Sentiment",
                ]
                gems_display["Avg Rating"] = gems_display["Avg Rating"].round(2)
                gems_display["Price"] = gems_display["Price"].apply(
                    lambda x: f"${x:.0f}"
                )
                gems_display["Sentiment"] = (gems_display["Sentiment"] * 100).round(
                    1
                ).astype(str) + "%"
                st.dataframe(gems_display.head(20), use_container_width=True)

                display_insight(
                    f"Found <strong>{len(gems)}</strong> hidden gems! Top: <strong>{gems.iloc[0]['name']}</strong> "
                    f"({gems.iloc[0]['dataset_avg_rating']:.2f}★). Best value propositions.",
                    "💎",
                )
            else:
                st.info("No hidden gems found. Try adjusting filters.")

    with col2:
        st.markdown(
            "<h3>⚠️ Overpriced (Low Rating, High Price)</h3>", unsafe_allow_html=True
        )
        if len(restaurant_value) > 0:
            overpriced = restaurant_value[
                (restaurant_value["dataset_avg_rating"] <= 3.5)
                & (restaurant_value["median_price"] >= 3)
            ].sort_values("dataset_avg_rating")

            if len(overpriced) > 0:
                overpriced_display = overpriced[
                    [
                        "name",
                        "city",
                        "dataset_avg_rating",
                        "median_price",
                        "dataset_reviews",
                        "avg_sentiment",
                    ]
                ].copy()
                overpriced_display.columns = [
                    "Name",
                    "City",
                    "Avg Rating",
                    "Price",
                    "Reviews",
                    "Sentiment",
                ]
                overpriced_display["Avg Rating"] = overpriced_display[
                    "Avg Rating"
                ].round(2)
                overpriced_display["Price"] = overpriced_display["Price"].apply(
                    lambda x: f"${x:.0f}"
                )
                overpriced_display["Sentiment"] = (
                    overpriced_display["Sentiment"] * 100
                ).round(1).astype(str) + "%"
                st.dataframe(overpriced_display.head(20), use_container_width=True)

                display_insight(
                    f"Found <strong>{len(overpriced)}</strong> potentially overpriced restaurants. "
                    f"Premium prices but underdelivering on quality.",
                    "⚠️",
                )
            else:
                st.success("No overpriced restaurants found—good market health!")

    st.divider()

    # Value Scatter Plot
    st.markdown("<h3>💰 Value for Money Scatter Plot</h3>", unsafe_allow_html=True)
    if len(restaurant_value) > 0:
        fig = px.scatter(
            restaurant_value,
            x="median_price",
            y="dataset_avg_rating",
            size="dataset_reviews",
            color="avg_sentiment",
            hover_data=["name", "city"],
            title="Rating vs Price (bubble size = review count)",
            color_continuous_scale=[[0, '#fda4af'], [0.5, '#fbbf24'], [1, '#93c5fd']],
            size_max=50,
            labels={"median_price": "Price Tier", "dataset_avg_rating": "Avg Rating"},
        )
        fig.add_hline(
            y=restaurant_value["dataset_avg_rating"].mean(),
            line_dash="dash",
            line_color="#a78bfa",
            annotation_text="Avg Rating",
        )
        fig.add_vline(
            x=restaurant_value["median_price"].median(),
            line_dash="dash",
            line_color="#a78bfa",
            annotation_text="Median Price",
        )
        st.plotly_chart(fig, use_container_width=True)

        display_insight(
            f"<strong>Top-left quadrant</strong> = best value (high rating, low price). "
            f"<strong>Bottom-right</strong> = worst value. Dashed lines mark averages.",
            "💰",
        )

    # Value Score Analysis
    st.markdown("<h3>⭐ Value Score Analysis</h3>", unsafe_allow_html=True)
    if len(restaurant_value) > 0:
        col1, col2 = st.columns(2)
        with col1:
            best_idx = restaurant_value["value_score"].idxmax()
            st.metric(
                "Best Value Restaurant",
                restaurant_value.loc[best_idx, "name"],
                f"Score: {restaurant_value['value_score'].max():.2f}",
            )
        with col2:
            worst_idx = restaurant_value["value_score"].idxmin()
            st.metric(
                "Worst Value Restaurant",
                restaurant_value.loc[worst_idx, "name"],
                f"Score: {restaurant_value['value_score'].min():.2f}",
            )

        fig = px.histogram(
            restaurant_value,
            x="value_score",
            nbins=30,
            title="Value Score Distribution",
            color_discrete_sequence=["#a78bfa"],
        )
        st.plotly_chart(fig, use_container_width=True)

        display_insight(
            f"Value Score = Rating ÷ Price. Higher = better value. "
            f"Restaurants above average offer competitive value.",
            "⭐",
        )
# ============ END OF PART 4 ============
# ============ PAGE: ADVANCED INSIGHTS ============
elif page == "Advanced Insights":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f0e9f5 0%, #e8dff0 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.08); text-align: center;
            border: 2px solid rgba(102, 126, 234, 0.15);">
            <h1 style="color: #333; margin: 0;">🎯 Advanced Insights & Patterns</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        filters = create_simple_filter("advanced", show_city=True, show_price=True)

    # Load pre-aggregated data
    rest_review_stats = load_aggregated_data("agg_rest_review_stats.csv")
    price_consistency = load_aggregated_data("agg_price_consistency.csv")
    rest_volatility = load_aggregated_data("agg_rest_volatility.csv")
    hourly = load_aggregated_data("agg_hourly.csv")
    cuisine_ranking = load_aggregated_data("agg_cuisine_ranking.csv")

    # 1. Review Volume vs Rating
    st.markdown(
        "<h3>1️⃣ Review Volume vs Rating Relationship</h3>", unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)

    with col1:
        if len(rest_review_stats) > 0:
            fig = px.scatter(
                rest_review_stats,
                x="review_count",
                y="avg_rating",
                title="Restaurant Review Count vs Average Rating",
                labels={
                    "review_count": "Number of Reviews",
                    "avg_rating": "Avg Rating",
                },
                color_discrete_sequence=["#a78bfa"],
                trendline="ols",
                trendline_color_override="#fda4af",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if len(rest_review_stats) > 0:
            corr_rating = rest_review_stats["review_count"].corr(
                rest_review_stats["avg_rating"]
            )
            corr_sentiment = rest_review_stats["review_count"].corr(
                rest_review_stats["avg_sentiment"]
            )

            st.info(f"""
            **📊 Correlations:**
            - Review Count ↔ Rating: **{corr_rating:.3f}**
            - Review Count ↔ Sentiment: **{corr_sentiment:.3f}**

            More reviews tend to {"increase" if corr_rating > 0 else "decrease"} ratings.
            """)

            display_insight(
                f"{'Popular restaurants tend to have better ratings—success breeds success.' if corr_rating > 0 else 'High volume restaurants face more scrutiny.'}",
                "📈",
            )

    st.divider()

    # 2. Sentiment Consistency by Price Tier
    st.markdown(
        "<h3>2️⃣ Sentiment Consistency by Price Tier</h3>", unsafe_allow_html=True
    )
    if len(price_consistency) > 0:
        # Apply price filter
        if "price_tiers" in filters and filters["price_tiers"]:
            price_consistency = price_consistency[
                price_consistency["price_tier"].isin(filters["price_tiers"])
            ]

        fig = px.bar(
            price_consistency,
            x="price_tier",
            y="correlation",
            title="Sentiment-Rating Correlation by Price Tier",
            text="correlation",
            color="correlation",
            color_continuous_scale=[[0, '#fda4af'], [0.5, '#c4b5fd'], [1, '#93c5fd']],
            labels={"price_tier": "Price Tier ($)", "correlation": "Correlation"},
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        highest = price_consistency.loc[price_consistency["correlation"].idxmax()]
        display_insight(
            f"Highest consistency: <strong>${int(highest['price_tier'])}</strong> (r={highest['correlation']:.3f}). "
            f"Higher correlation = review text matches star ratings more closely.",
            "🔗",
        )

    st.divider()

    # 3. Rating Volatility Analysis
    st.markdown("<h3>3️⃣ Rating Volatility Analysis</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if len(rest_volatility) > 0:
            fig = px.scatter(
                rest_volatility,
                x="avg_rating",
                y="rating_std",
                size="review_count",
                title="Rating Volatility vs Average Rating",
                labels={"avg_rating": "Average Rating", "rating_std": "Rating Std Dev"},
                color_discrete_sequence=["#a78bfa"],
                size_max=50,
            )
            st.plotly_chart(fig, use_container_width=True)

            display_insight(
                f"High volatility (top) = inconsistent experiences. "
                f"Low volatility + high rating (bottom-right) = reliably excellent.",
                "📊",
            )

    with col2:
        st.markdown("<h4>🔴 Most Volatile Restaurants</h4>", unsafe_allow_html=True)
        if len(rest_volatility) > 0:
            high_vol = rest_volatility.nlargest(10, "rating_std")[
                ["business_id", "avg_rating", "rating_std", "review_count"]
            ]
            high_vol["rating_std"] = high_vol["rating_std"].round(2)
            high_vol["avg_rating"] = high_vol["avg_rating"].round(2)
            st.dataframe(high_vol.reset_index(drop=True), use_container_width=True)

    st.divider()

    # 4. Hourly Patterns
    st.markdown("<h3>4️⃣ Temporal Patterns - Hour of Day</h3>", unsafe_allow_html=True)
    if len(hourly) > 0:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=hourly["hour"],
                y=hourly["avg_stars"],
                name="Avg Rating",
                line=dict(color="#a78bfa", width=3),
                mode="lines+markers",
                marker=dict(size=8)
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=hourly["hour"],
                y=hourly["review_count"],
                name="Review Count",
                marker=dict(color="rgba(147, 197, 253, 0.3)"),
            ),
            secondary_y=True,
        )
        fig.update_layout(title="Rating & Volume by Hour of Day", height=400)
        fig.update_yaxes(title_text="Avg Rating", secondary_y=False)
        fig.update_yaxes(title_text="Review Count", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        best_hour = hourly.loc[hourly["avg_stars"].idxmax(), "hour"]
        busiest_hour = hourly.loc[hourly["review_count"].idxmax(), "hour"]
        display_insight(
            f"Best-rated hour: <strong>{int(best_hour)}:00</strong>. "
            f"Busiest: <strong>{int(busiest_hour)}:00</strong>. "
            f"Peak hours may have service strain.",
            "⏰",
        )

    st.divider()

    # 5. Cuisine Performance Ranking
    st.markdown(
        "<h3>5️⃣ Cuisine Performance Ranking (Composite Score)</h3>",
        unsafe_allow_html=True,
    )
    if len(cuisine_ranking) > 0:
        fig = px.bar(
            cuisine_ranking.head(15),
            x="cuisine",
            y="rank_score",
            text="rank_score",
            color="avg_rating",
            color_continuous_scale=[[0, '#fda4af'], [0.5, '#a78bfa'], [1, '#93c5fd']],
            title="Top Cuisines - Composite Score (Rating 40% + Sentiment 30% + Popularity 30%)",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        top = cuisine_ranking.iloc[0]
        display_insight(
            f"Top performer: <strong>{top['cuisine']}</strong> (score: {top['rank_score']:.3f}). "
            f"Composite score balances quality, perception, and demand.",
            "🏆",
        )

# ============ PAGE: COMPARISONS (ENHANCED) ============
elif page == "Comparisons":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #ffe8ef 0%, #ffd9e8 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(255, 154, 158, 0.08); text-align: center;
            border: 2px solid rgba(255, 154, 158, 0.15);">
            <h1 style="color: #333; margin: 0;">🔍 Restaurant Comparisons</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add filters
    with st.expander("🔍 **Filter Restaurant Selection**", expanded=False):
        filters = create_simple_filter("compare", show_city=True, show_price=True)

    # Load data
    restaurant_list = load_aggregated_data("agg_restaurant_list.csv")
    restaurant_detailed = load_aggregated_data("agg_restaurant_detailed.csv")

    # Apply filters to restaurant list
    if len(restaurant_list) > 0:
        restaurant_list = apply_filters_to_df(
            restaurant_list,
            filters,
            city_col="city",
            rating_col="avg_rating",
            price_col="median_price",
        )

    if len(restaurant_list) > 0 and len(restaurant_detailed) > 0:
        st.markdown("#### Select Two Restaurants to Compare")
        col1, col2 = st.columns(2)

        restaurant_names = restaurant_list["name"].unique().tolist()

        with col1:
            rest1_name = st.selectbox("🏪 Restaurant 1:", restaurant_names, key="rest1")
        with col2:
            default_idx = min(1, len(restaurant_names) - 1)
            rest2_name = st.selectbox(
                "🏪 Restaurant 2:", restaurant_names, index=default_idx, key="rest2"
            )

        st.markdown("---")

        # Get restaurant data
        rest1_data = (
            restaurant_detailed[restaurant_detailed["name"] == rest1_name].iloc[0]
            if len(restaurant_detailed[restaurant_detailed["name"] == rest1_name]) > 0
            else None
        )
        rest2_data = (
            restaurant_detailed[restaurant_detailed["name"] == rest2_name].iloc[0]
            if len(restaurant_detailed[restaurant_detailed["name"] == rest2_name]) > 0
            else None
        )

        if rest1_data is not None and rest2_data is not None:
            # Winner insight
            winner = (
                rest1_name
                if rest1_data["avg_rating"] > rest2_data["avg_rating"]
                else rest2_name
            )
            display_insight(
                f"<strong>{winner}</strong> leads in ratings. "
                f"Rating difference: <strong>{abs(rest1_data['avg_rating'] - rest2_data['avg_rating']):.2f}★</strong>. "
                f"Sentiment gap: <strong>{abs(rest1_data['avg_sentiment'] - rest2_data['avg_sentiment']) * 100:.1f}pp</strong>.",
                "🔍",
            )

            st.divider()

            # CHART 1: Side-by-Side Metrics Comparison
            st.markdown("#### 📊 Key Metrics Comparison")
            comparison_df = pd.DataFrame(
                {
                    "Metric": [
                        "Avg Rating",
                        "Sentiment %",
                        "Review Count",
                        "Price Tier",
                    ],
                    rest1_name: [
                        rest1_data["avg_rating"],
                        rest1_data["avg_sentiment"] * 100,
                        rest1_data["review_count"],
                        rest1_data["median_price"],
                    ],
                    rest2_name: [
                        rest2_data["avg_rating"],
                        rest2_data["avg_sentiment"] * 100,
                        rest2_data["review_count"],
                        rest2_data["median_price"],
                    ],
                }
            )

            fig = px.bar(
                comparison_df,
                x="Metric",
                y=[rest1_name, rest2_name],
                barmode="group",
                title="Restaurant Comparison",
                color_discrete_map={rest1_name: "#a78bfa", rest2_name: "#fda4af"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # CHART 2: Performance Radar Chart
            st.markdown("#### 🎯 Performance Radar")
            
            # Calculate normalized metrics for radar chart
            max_reviews = max(rest1_data['review_count'], rest2_data['review_count'])
            
            radar_data = pd.DataFrame({
                'Metric': ['Rating', 'Sentiment', 'Popularity', 'Value'],
                rest1_name: [
                    rest1_data['avg_rating'] / 5 * 100,  # Normalize to 100
                    rest1_data['avg_sentiment'] * 100,
                    min(rest1_data['review_count'] / max_reviews * 100, 100) if max_reviews > 0 else 0,
                    (6 - rest1_data['median_price']) / 5 * 100  # Inverse (lower price = higher value)
                ],
                rest2_name: [
                    rest2_data['avg_rating'] / 5 * 100,
                    rest2_data['avg_sentiment'] * 100,
                    min(rest2_data['review_count'] / max_reviews * 100, 100) if max_reviews > 0 else 0,
                    (6 - rest2_data['median_price']) / 5 * 100
                ]
            })
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_data[rest1_name],
                theta=radar_data['Metric'],
                fill='toself',
                name=rest1_name,
                line_color='#a78bfa',
                fillcolor='rgba(167, 139, 250, 0.2)'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_data[rest2_name],
                theta=radar_data['Metric'],
                fill='toself',
                name=rest2_name,
                line_color='#fda4af',
                fillcolor='rgba(253, 164, 175, 0.2)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        showticklabels=True,
                        ticks='outside'
                    )
                ),
                showlegend=True,
                height=450
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)

            st.divider()

            # Detailed Metrics with Delta
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏪 {rest1_name}")
                st.metric("Average Rating", f"{rest1_data['avg_rating']:.2f}★")
                st.metric("Positive Sentiment", f"{rest1_data['avg_sentiment'] * 100:.1f}%")
                st.metric("Total Reviews", f"{rest1_data['review_count']:,}")
                st.metric("Price Tier", "💰" * int(rest1_data['median_price']))
                
            with col2:
                st.markdown(f"### 🏪 {rest2_name}")
                rating_diff = rest2_data['avg_rating'] - rest1_data['avg_rating']
                sentiment_diff = (rest2_data['avg_sentiment'] - rest1_data['avg_sentiment']) * 100
                review_diff = rest2_data['review_count'] - rest1_data['review_count']
                
                st.metric(
                    "Average Rating", 
                    f"{rest2_data['avg_rating']:.2f}★",
                    delta=f"{rating_diff:+.2f}"
                )
                st.metric(
                    "Positive Sentiment", 
                    f"{rest2_data['avg_sentiment'] * 100:.1f}%",
                    delta=f"{sentiment_diff:+.1f}pp"
                )
                st.metric(
                    "Total Reviews", 
                    f"{rest2_data['review_count']:,}",
                    delta=f"{review_diff:+,}"
                )
                st.metric("Price Tier", "💰" * int(rest2_data['median_price']))

            st.divider()

            # CHART 3: Popularity vs Quality Matrix
            st.markdown("#### 📊 Market Positioning")
            
            # Load all restaurants for context
            all_restaurants = restaurant_detailed.copy()
            
            # Create scatter plot with all restaurants in gray, highlight selected two
            fig_matrix = go.Figure()
            
            # Add all restaurants as background
            fig_matrix.add_trace(go.Scatter(
                x=all_restaurants['review_count'],
                y=all_restaurants['avg_rating'],
                mode='markers',
                name='Other Restaurants',
                marker=dict(size=8, color='lightgray', opacity=0.3),
                text=all_restaurants['name'],
                hovertemplate='<b>%{text}</b><br>Reviews: %{x}<br>Rating: %{y:.2f}★<extra></extra>'
            ))
            
            # Highlight Restaurant 1
            fig_matrix.add_trace(go.Scatter(
                x=[rest1_data['review_count']],
                y=[rest1_data['avg_rating']],
                mode='markers+text',
                name=rest1_name,
                marker=dict(size=15, color='#a78bfa', line=dict(width=2, color='white')),
                text=[rest1_name],
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>Reviews: %{x}<br>Rating: %{y:.2f}★<extra></extra>'
            ))
            
            # Highlight Restaurant 2
            fig_matrix.add_trace(go.Scatter(
                x=[rest2_data['review_count']],
                y=[rest2_data['avg_rating']],
                mode='markers+text',
                name=rest2_name,
                marker=dict(size=15, color='#fda4af', line=dict(width=2, color='white')),
                text=[rest2_name],
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>Reviews: %{x}<br>Rating: %{y:.2f}★<extra></extra>'
            ))
            
            fig_matrix.update_layout(
                title='Review Count vs Rating (Competitive Context)',
                xaxis_title='Number of Reviews (Popularity)',
                yaxis_title='Average Rating (Quality)',
                height=500,
                showlegend=True,
                yaxis=dict(range=[0, 5])
            )
            
            st.plotly_chart(fig_matrix, use_container_width=True)

            st.divider()

            # CHART 4: Sentiment Comparison
            st.markdown("#### 💭 Sentiment Breakdown")
            
            sentiment_data = pd.DataFrame({
                'Restaurant': [rest1_name, rest2_name],
                'Positive Sentiment': [
                    rest1_data['avg_sentiment'] * 100,
                    rest2_data['avg_sentiment'] * 100
                ],
                'Negative Sentiment': [
                    (1 - rest1_data['avg_sentiment']) * 100,
                    (1 - rest2_data['avg_sentiment']) * 100
                ]
            })
            
            fig_sentiment = go.Figure()
            
            fig_sentiment.add_trace(go.Bar(
                name='Positive Sentiment',
                x=sentiment_data['Restaurant'],
                y=sentiment_data['Positive Sentiment'],
                marker_color='#93c5fd',
                text=sentiment_data['Positive Sentiment'].apply(lambda x: f'{x:.1f}%'),
                textposition='inside'
            ))
            
            fig_sentiment.add_trace(go.Bar(
                name='Negative Sentiment',
                x=sentiment_data['Restaurant'],
                y=sentiment_data['Negative Sentiment'],
                marker_color='#fda4af',
                text=sentiment_data['Negative Sentiment'].apply(lambda x: f'{x:.1f}%'),
                textposition='inside'
            ))
            
            fig_sentiment.update_layout(
                barmode='stack',
                title='Customer Sentiment Distribution',
                yaxis_title='Percentage (%)',
                height=400,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_sentiment, use_container_width=True)

    else:
        st.warning("No restaurants match the current filters. Please adjust filters.")
        
# ============ PAGE: DATA TABLE ============
elif page == "Data Table":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #e8ecf5 0%, #d9e4f0 100%);
            padding: 30px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1); text-align: center;
            border: 2px solid rgba(102, 126, 234, 0.2);">
            <h1 style="color: #333; margin: 0;">📋 Full Data Table & Export</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load main data
    main_data = load_main_data()

    if len(main_data) > 0:
        st.markdown("#### 🔍 Filter Data")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            min_rating = st.slider("⭐ Min Rating:", 1.0, 5.0, 1.0, key="dt_rating")

        with col2:
            sentiment_filter = st.multiselect(
                "😊 Sentiment:",
                ["Positive", "Negative"],
                default=["Positive", "Negative"],
                key="dt_sentiment",
            )

        with col3:
            price_tiers = sorted(main_data["price_tier"].dropna().unique().tolist())
            price_filter = st.multiselect(
                "💰 Price Tier:", price_tiers, default=price_tiers, key="dt_price"
            )

        with col4:
            cities = ["All"] + sorted(main_data["city"].dropna().unique().tolist())
            city_filter = st.selectbox("🏙️ City:", cities, key="dt_city")

        st.markdown("---")

        # Apply filters
        filtered = main_data[
            (main_data["stars"] >= min_rating)
            & (main_data["sentiment_text"].isin(sentiment_filter))
            & (main_data["price_tier"].isin(price_filter))
        ].copy()

        if city_filter != "All":
            filtered = filtered[filtered["city"] == city_filter]

        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", f"{len(filtered):,}")
        with col2:
            st.metric("Unique Restaurants", f"{filtered['business_id'].nunique():,}")
        with col3:
            avg = filtered["stars"].mean() if len(filtered) > 0 else 0
            st.metric("Avg Rating", f"{avg:.2f}★")
        with col4:
            pos = (
                (filtered["sentiment_label"].sum() / len(filtered) * 100)
                if len(filtered) > 0
                else 0
            )
            st.metric("Positive Sentiment", f"{pos:.1f}%")

        # Prepare display dataframe
        display_cols = [
            "name",
            "city",
            "stars",
            "date",
            "price_tier",
            "sentiment_text",
            "business_stars",
        ]
        if all(col in filtered.columns for col in display_cols):
            display_df = filtered[display_cols].copy()
            display_df = display_df.sort_values("date", ascending=False)
            display_df.columns = [
                "Restaurant",
                "City",
                "Rating",
                "Date",
                "Price",
                "Sentiment",
                "Yelp Rating",
            ]

            st.markdown(
                f"**📊 Showing {len(display_df):,} of {len(main_data):,} reviews** ({len(display_df) / len(main_data) * 100:.1f}%)"
            )

            st.dataframe(display_df.head(1000), use_container_width=True, height=500)

            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"yelp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            display_insight(
                f"Export filtered data for further analysis. "
                f"Current selection: {len(display_df):,} reviews.",
                "📥",
            )
    else:
        st.warning("Main data file not found. Please run preprocessing script.")
# ============ END OF PART 5 ============
