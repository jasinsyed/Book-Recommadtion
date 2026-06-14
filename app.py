import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
import time
import difflib
import urllib.parse

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM CSS (ENHANCED UI & OVERLAY TRICK)
# ==========================
st.markdown("""
    <style>
    /* Main Page Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: -webkit-linear-gradient(45deg, #4CAF50, #2E86C1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #888888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Card Link Wrapper */
    a.custom-card-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        height: 100%;
    }
    
    /* Beautiful Flexbox Book Cards */
    .book-card {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        padding: 15px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        height: 380px; /* Fixed height for uniform grids */
        display: flex;
        flex-direction: column;
        align-items: center;
        border: 1px solid #333;
        margin-bottom: 20px;
    }
    
    /* Smooth Hover Animation */
    .book-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 20px rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
    }
    
    /* Container for the book cover (CSS OVERLAY TRICK) */
    .book-cover-container {
        width: 130px;
        height: 190px;
        margin: 0 auto 12px auto;
        background-color: #2b2b2b;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    
    /* The text that sits behind the image */
    .no-cover-text {
        color: #888888;
        font-size: 14px;
        font-weight: bold;
        text-align: center;
        position: absolute;
        z-index: 1;
    }
    
    /* The image itself, sitting on top */
    .book-cover-img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: 2;
        color: transparent; /* Hides the default broken image icon */
    }

    .book-title {
        font-size: 15px;
        font-weight: 700;
        color: #ffffff;
        width: 100%;
        display: -webkit-box;
        -webkit-line-clamp: 2; /* Limit to 2 lines */
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    
    .book-author {
        font-size: 13px;
        color: #b3b3b3;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
    }
    
    .book-year {
        font-size: 11px;
        color: #777777;
        margin-bottom: auto; /* Pushes everything below this to the bottom */
    }
    
    /* Gradient Match Badge */
    .match-badge {
        background: linear-gradient(90deg, #4CAF50, #2E86C1);
        color: white;
        font-size: 11px;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .store-badge {
        font-size: 11px;
        color: #4CAF50;
        font-weight: 600;
        margin-top: 8px;
        opacity: 0.8;
        transition: opacity 0.2s;
    }
    .book-card:hover .store-badge {
        opacity: 1;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS (CACHED)
# ==========================
@st.cache_resource
def load_data():
    start = time.time()
    books = pd.read_pickle("cleaned_books.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    title_to_idx = joblib.load("title_to_idx.pkl")
    collab_sim = joblib.load("collab_sim.pkl")
    collab_title_to_idx = joblib.load("collab_title_to_idx.pkl")
    book_titles_list = joblib.load("book_titles_list.pkl")
    
    knn = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute', n_jobs=-1)
    knn.fit(tfidf_matrix)
    return (books, tfidf_matrix, title_to_idx, collab_sim, collab_title_to_idx, book_titles_list, knn)

with st.spinner("Loading library databases..."):
    (books, tfidf_matrix, title_to_idx, collab_sim, collab_title_to_idx, book_titles_list, knn) = load_data()

# ==========================
# INTELLIGENT SEARCH ENGINE
# ==========================
def intelligent_search(query, valid_titles):
    """Finds the best matching book title using a multi-tiered approach."""
    query = query.strip().lower()
    if not query:
        return None, "empty"
        
    # 1. Exact Match Check
    for title in valid_titles:
        if title.lower() == query:
            return title, "exact"
            
    # 2. Starts With Check
    starts_with_matches = [t for t in valid_titles if t.lower().startswith(query)]
    if starts_with_matches:
        return min(starts_with_matches, key=len), "substring"
        
    # 3. Contains Substring Check
    contains_matches = [t for t in valid_titles if query in t.lower()]
    if contains_matches:
        return min(contains_matches, key=len), "substring"
        
    # 4. Fuzzy Spellcheck 
    closest_matches = difflib.get_close_matches(query, valid_titles, n=1, cutoff=0.45)
    if closest_matches:
        return closest_matches[0], "fuzzy"
        
    return None, "none"

# ==========================
# RECOMMENDATION FUNCTIONS
# ==========================
@st.cache_data(ttl=3600, show_spinner=False)
def content_based_recommend(book_title, n=10, min_year=None, max_year=None, author_filter=None):
    book_title = book_title.lower()
    if book_title not in title_to_idx: return pd.DataFrame()
    
    idx = title_to_idx[book_title]
    
    # Catch Duplicate Indices Bug
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    elif isinstance(idx, (list, np.ndarray)):
        idx = idx[0]
        
    # The "Zero Vector" Bug Fix
    if tfidf_matrix[idx].sum() == 0:
        return pd.DataFrame()
        
    fetch_n = min(max(n * 20, 200), tfidf_matrix.shape[0])
    distances, indices = knn.kneighbors(tfidf_matrix[idx].reshape(1, -1), n_neighbors=fetch_n)
    
    top_idx = indices[0][1:]
    sim_scores = 1 - distances[0][1:] 
    
    # INCLUDE ISBN for the Open Library API
    recs = books.iloc[top_idx][["ISBN", "Title", "Author", "Year", "Publisher", "Img-L"]].copy()
    recs["Similarity"] = sim_scores
    
    # Filter out fake 100% matches and completely unrelated books
    recs = recs[(recs["Similarity"] < 0.99) & (recs["Similarity"] > 0.01)]
    recs = recs[recs["Title"].str.lower() != book_title]
    
    # DIVERSITY FILTER: Only one book per author
    recs = recs.drop_duplicates(subset=["Author"], keep="first")
    
    if min_year: recs = recs[recs["Year"] >= min_year]
    if max_year: recs = recs[recs["Year"] <= max_year]
    if author_filter and author_filter.strip():
        recs = recs[recs["Author"].str.contains(author_filter, case=False, na=False)]
    
    return recs.head(n)

@st.cache_data(ttl=3600, show_spinner=False)
def collaborative_recommend(book_title, n=10, min_year=None, max_year=None, author_filter=None):
    book_title = book_title.lower()
    if book_title not in collab_title_to_idx: return pd.DataFrame()
    
    idx = collab_title_to_idx[book_title]
    
    # Catch Duplicate Indices Bug
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    elif isinstance(idx, (list, np.ndarray)):
        idx = idx[0]
        
    sim_scores = collab_sim[idx].copy()
    sim_scores[idx] = 0
    
    # "Zero Vector" Bug for collaborative filtering
    if np.sum(sim_scores) == 0:
        return pd.DataFrame()
    
    fetch_n = min(max(n * 20, 200), len(sim_scores))
    top_idx = np.argsort(sim_scores)[::-1][:fetch_n]
    
    rec_titles = [book_titles_list[i] for i in top_idx]
    recs = books[books["Title"].isin(rec_titles)].drop_duplicates(subset=["Title"]).copy()
    
    score_dict = dict(zip(rec_titles, sim_scores[top_idx]))
    recs["Score"] = recs["Title"].map(score_dict)
    recs = recs.sort_values("Score", ascending=False)
    
    # Filter out fake 100% matches and completely unrelated books
    recs = recs[(recs["Score"] < 0.99) & (recs["Score"] > 0.01)]
    recs = recs[recs["Title"].str.lower() != book_title]
    
    # DIVERSITY FILTER: Only one book per author
    recs = recs.drop_duplicates(subset=["Author"], keep="first")
    
    if min_year: recs = recs[recs["Year"] >= min_year]
    if max_year: recs = recs[recs["Year"] <= max_year]
    if author_filter and author_filter.strip():
        recs = recs[recs["Author"].str.contains(author_filter, case=False, na=False)]
    
    return recs.head(n)

# ==========================
# UI HELPER: RENDER BOOK GRID
# ==========================
def render_book_grid(df):
    """Reusable function to draw a grid of clickable books from a dataframe."""
    cols_per_row = 4
    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(df):
                row = df.iloc[i + j]
                title = row.get("Title", "Unknown")
                author = row.get("Author", "Unknown")
                year = row.get("Year", "")
                img = row.get("Img-L", "")
                isbn = row.get("ISBN", "")
                
                # Sanitize text to prevent HTML attribute breaking
                safe_title = str(title).replace('"', '&quot;')
                safe_author = str(author).replace('"', '&quot;')
                
                # Create the Open Library API backup link
                open_lib_img = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false"
                
                # Double Fallback Logic: Amazon Image -> Open Library API -> CSS Overlay Text
                if pd.isna(img) or str(img).strip() == "" or str(img).strip() == "nan":
                    # If Amazon image is completely missing from DB, fallback to Open Library
                    img = open_lib_img
                else:
                    img = str(img).replace("http://", "https://")
                
                # Using Goodreads search to avoid CAPTCHA blocks
                search_query = f"{title} {author}"
                encoded_query = urllib.parse.quote_plus(search_query)
                store_url = f"https://www.goodreads.com/search?q={encoded_query}"
                
                # Determine match score
                score_html = ""
                for score_col in ["Similarity", "Score"]:
                    if score_col in row:
                        val = row[score_col]
                        display_score = val * 100 if val <= 1 else val
                        score_html = f'<div class="match-badge">⭐ Match: {display_score:.0f}%</div>\n'
                        break
                
                with col:
                    # Formatted strictly left-aligned to prevent Streamlit interpreting as Markdown Code Block
                    # USING PURE CSS OVERLAY TRICK FOR BROKEN IMAGES
                    html_card = f"""<a href="{store_url}" target="_blank" class="custom-card-link" title="Click to view on Goodreads">
<div class="book-card">
<div class="book-cover-container">
<span class="no-cover-text">No Cover</span>
<img src="{img}" class="book-cover-img" alt="">
</div>
<div class="book-title" title="{safe_title}">{safe_title}</div>
<div class="book-author" title="{safe_author}">✍️ {safe_author}</div>
<div class="book-year">📅 {int(year) if pd.notna(year) and year > 0 else 'N/A'}</div>
{score_html}<span class="store-badge">🛒 View in Store ↗</span>
</div>
</a>"""
                    st.markdown(html_card, unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=60)
    st.header("⚙️ Settings")
    
    method = st.radio(
        "Recommendation Engine",
        ["Content Based (Similar Plots)", "Collaborative (Users Also Liked)"]
    )
    
    num_recs = st.slider("Number of books", 4, 24, 8, 4)
    
    st.divider()
    st.subheader("🔧 Advanced Filters")
    min_year = st.number_input("Min Year", min_value=1800, max_value=2025, value=1800, step=10)
    max_year = st.number_input("Max Year", min_value=1800, max_value=2025, value=2025, step=10)
    author_filter = st.text_input("Preferred Author", placeholder="e.g., J.K. Rowling")
    
    st.divider()
    st.caption(f"📚 Total Library: **{len(books):,}** books")

# ==========================
# MAIN PAGE
# ==========================
st.markdown('<p class="main-header">AI Book Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover your next favorite read powered by Machine Learning.</p>', unsafe_allow_html=True)

if "Content" in method:
    valid_titles = books["Title"].dropna().unique().tolist()
else:
    valid_titles = [str(t) for t in book_titles_list]

search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
with search_col2:
    search_query = st.text_input(
        "🔍 Search",
        placeholder="Type a book you love... (e.g., 'The Hobbit' or 'Harry Potter')",
        label_visibility="collapsed"
    )
st.write("") 

# ==========================
# LOGIC: DEFAULT VIEW VS SEARCH
# ==========================
if not search_query.strip():
    st.markdown("### 🔥 Trending Right Now")
    trending_books = books.sample(8, random_state=int(time.time()) % 100)
    render_book_grid(trending_books)
    
    st.divider()
    
    st.markdown("### 🆕 Classic & Latest Releases")
    valid_years = books[(books["Year"] > 1900) & (books["Year"] <= 2024)]
    latest_books = valid_years.sort_values("Year", ascending=False).head(8)
    render_book_grid(latest_books)

else:
    best_match, match_type = intelligent_search(search_query, valid_titles)
    
    if best_match:
        st.divider()
        
        if match_type in ["fuzzy", "substring"]:
            st.markdown(f"### 📖 Because you enjoyed **{best_match}** *(closest match)*")
        else:
            st.markdown(f"### 📖 Because you enjoyed **{best_match}**")
        
        filters = {
            "min_year": min_year if min_year > 1800 else None,
            "max_year": max_year if max_year < 2025 else None,
            "author_filter": author_filter if author_filter else None
        }
        
        with st.spinner("Analyzing library..."):
            start_time = time.time()
            if "Content" in method:
                recs = content_based_recommend(best_match, num_recs, **filters)
            else:
                recs = collaborative_recommend(best_match, num_recs, **filters)
            elapsed = (time.time() - start_time) * 1000
        
        if recs.empty:
            st.info("No matches found with the current filters. Try adjusting the Year or Author settings in the sidebar.")
        else:
            st.caption(f"✨ Found **{len(recs)}** recommendations in {elapsed:.0f} ms")
            render_book_grid(recs)
            
    else:
        st.error(f"We couldn't find a book matching '{search_query}'. Try a different title!")