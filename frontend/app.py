import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("BACKEND_API_URL", "http://localhost:5000/api")

# Set page configuration
st.set_page_config(
    page_title="Task Queue Manager",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Streamlined CSS with only essential styles
st.markdown("""
<style>
    /* Hide theme selector */
    [data-testid="stSidebar"] [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Set overall page background to a softer color */
    .stApp {
        background-color: #f2f2f2;
    }
    
    /* Basic styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }
    
    /* Header and section styles */
    .header, .section-header {
        background-color: #e6e9ed;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Card components */
    .worker-card, .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dde2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .worker-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #b6c0ca;
    }
    
    .metric-card:hover {
        transform: scale(1.03);
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* Grid layouts */
    .worker-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1.5rem;
    }
    
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-badge.active {
        background-color: #c5e1c5;
        color: #2c5f2c;
    }
    
    .status-badge.inactive {
        background-color: #edd0d0;
        color: #8e3a3a;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0.5rem 0;
        color: #445566;
    }
    
    .metric-label {
        color: #667788;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    /* Update info */
    .update-info {
        color: #667788;
        font-weight: 400;
        background-color: #e6e9ed;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
    }
    
    /* Table container */
    .dataframe-container {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border: 1px solid #dde2e6;
    }
    
    /* Make dataframe headers less bright */
    div[data-testid="stDataFrame"] th {
        background-color: #e6e9ed !important;
        color: #445566 !important;
        font-weight: 500 !important;
    }
    
    /* Make dataframe cells less bright */
    div[data-testid="stDataFrame"] td {
        background-color: #f8f9fa !important;
        color: #445566 !important;
    }
    
    /* Add more styling for better card appearance */
    
    /* Add refresh effect animation */
    @keyframes fade {
        0% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .refresh-animation {
        animation: fade 0.5s ease;
    }
    
    /* Card header styling */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e6e9ed;
    }
    
    .worker-title {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Status badge improved */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .status-badge.active {
        background-color: #c5e1c5;
        color: #2c5f2c;
    }
    
    .status-badge.inactive {
        background-color: #edd0d0;
        color: #8e3a3a;
    }
    
    /* Info rows styling */
    .card-content {
        padding: 0 0.25rem;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    
    .info-label {
        font-weight: 600;
        color: #667788;
        flex: 1;
    }
    
    .info-value {
        flex: 2;
        text-align: right;
        color: #2c3e50;
    }
    
    /* Load bar styling */
    .load-bar-container {
        flex: 2;
        height: 0.5rem;
        background-color: #e6e9ed;
        border-radius: 0.25rem;
        overflow: hidden;
        position: relative;
    }
    
    .load-bar {
        height: 100%;
        background: linear-gradient(to right, #4caf50, #8bc34a);
        border-radius: 0.25rem;
        transition: width 0.5s ease;
    }
    
    /* When load is high, change color */
    .load-bar.high {
        background: linear-gradient(to right, #ff9800, #f44336);
    }
    
    .load-text {
        position: absolute;
        right: 0.5rem;
        top: -0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #445566;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing data between refreshes
if 'worker_status' not in st.session_state:
    st.session_state.worker_status = None
if 'queue_stats' not in st.session_state:
    st.session_state.queue_stats = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'view_type' not in st.session_state:
    st.session_state.view_type = "Grid View"
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'refresh_rate' not in st.session_state:
    st.session_state.refresh_rate = 5

# Helper functions
def fetch_api_data(endpoint):
    """Fetch data from the API without caching"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"API connection error: {e}")
        return None

def get_worker_metadata(worker, key, default=0):
    """Safely extract metadata from worker"""
    try:
        return worker.get("metadata", {}).get(key, default)
    except:
        return default

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    if not timestamp_str:
        return "Never"
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now()
        delta = now - dt
        
        if delta < timedelta(minutes=1):
            return "Just now"
        elif delta < timedelta(hours=1):
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        elif delta < timedelta(days=1):
            hours = int(delta.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            days = delta.days
            return f"{days} day{'s' if days > 1 else ''} ago"
    except:
        return timestamp_str

def update_data():
    """Update the data in session state"""
    st.session_state.worker_status = fetch_api_data("workers/status")
    st.session_state.queue_stats = fetch_api_data("queue/stats")
    st.session_state.last_update = datetime.now()

def toggle_auto_refresh():
    """Toggle auto refresh state"""
    st.session_state.auto_refresh = not st.session_state.auto_refresh

def change_view_type():
    """Update view type in session state"""
    st.session_state.view_type = st.session_state.view_type_radio

def change_refresh_rate():
    """Update refresh rate in session state"""
    try:
        st.session_state.refresh_rate = int(st.session_state.refresh_rate_input)
    except:
        st.session_state.refresh_rate = 5

# Custom header with forced light theme
st.markdown('<div class="header"><h1>Worker Management</h1></div>', unsafe_allow_html=True)

# Hide streamlit style elements
hide_streamlit_style = """
    <style>
        /* Hide the top-right "Running..." status and hamburger menu */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Optional: hide the 'Running...' icon completely */
        .stStatusWidget {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Refresh controls with improved visibility
col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

with col1:
    last_update_str = st.session_state.last_update.strftime("%H:%M:%S")
    st.markdown(f'<span class="update-info">Last updated: {last_update_str}</span>', unsafe_allow_html=True)

with col2:
    auto_refresh = st.checkbox("Auto refresh", value=st.session_state.auto_refresh, key='auto_refresh_checkbox', on_change=toggle_auto_refresh)

with col3:
    refresh_rate = st.number_input("Refresh every (sec)", 
                                   min_value=1, 
                                   max_value=60, 
                                   value=st.session_state.refresh_rate,
                                   key='refresh_rate_input',
                                   on_change=change_refresh_rate)

with col4:
    st.button("Refresh Now", on_click=update_data)

# Update the data if it's the first run or if auto-refresh is enabled
current_time = datetime.now()
time_since_update = (current_time - st.session_state.last_update).total_seconds()

if st.session_state.worker_status is None or (st.session_state.auto_refresh and time_since_update >= st.session_state.refresh_rate):
    update_data()

# Display data only if we have it
if st.session_state.worker_status and st.session_state.queue_stats:
    # Worker summary metrics with refresh animation class
    st.markdown('<div class="section-header refresh-animation"><h3>System Overview</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="metrics-container refresh-animation">', unsafe_allow_html=True)
    
    # Total Workers
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Workers</div>
        <div class="metric-value">{st.session_state.worker_status["total"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Workers
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Workers</div>
        <div class="metric-value">{st.session_state.worker_status["active"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Queued Tasks
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Queued Tasks</div>
        <div class="metric-value">{st.session_state.queue_stats.get("queued_tasks", 0)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # System Utilization
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">System Utilization</div>
        <div class="metric-value">{st.session_state.queue_stats.get("capacity_percent", 0):.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Worker List Section
    st.markdown('<div class="section-header refresh-animation"><h3>Worker Nodes</h3></div>', unsafe_allow_html=True)
    
    # Table view of workers with enhanced styling
    if st.session_state.worker_status.get("workers"):
        st.markdown('<div class="dataframe-container refresh-animation">', unsafe_allow_html=True)
        
        worker_df = pd.DataFrame([
            {
                "Status": "🟢 Active" if w.get("is_active") else "🔴 Inactive",
                "Hostname": w["hostname"] or "Unknown",
                "ID": w["id"],
                "IP Address": w["ip_address"] or "Unknown",
                "Active Tasks": get_worker_metadata(w, "active_tasks", 0),
                "Max Tasks": get_worker_metadata(w, "max_concurrent_tasks", 0),
                "Load (%)": round((get_worker_metadata(w, "active_tasks", 0) / get_worker_metadata(w, "max_concurrent_tasks", 1)) * 100, 1) if get_worker_metadata(w, "max_concurrent_tasks", 0) > 0 else 0,
                "Last Heartbeat": format_timestamp(w["last_heartbeat"]),
                "Last Heartbeat (Raw)": w["last_heartbeat"]  # For sorting
            }
            for w in st.session_state.worker_status["workers"]
        ])
        
        # Sort by status (active first) then by last heartbeat
        worker_df = worker_df.sort_values(
            by=["Status", "Last Heartbeat (Raw)"], 
            ascending=[False, False]
        )
        
        # Display the styled dataframe
        st.markdown("""
        <style>
            /* Custom styling for dataframe */
            .dataframe-container {
                overflow: hidden;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                background-color: white;
            }
            
            /* Hover styling */
            [data-testid="stDataFrame"] tr:hover td {
                background-color: #f2f7ff !important;
                transition: background-color 0.2s;
            }
            
            /* Status icons */
            [data-testid="stDataFrame"] td:first-child {
                font-size: 16px !important;
            }
            
            /* Headers */
            [data-testid="stDataFrame"] th {
                background-color: #f8f9fa !important;
                color: #445566 !important;
                font-weight: 600 !important;
                text-transform: uppercase;
                font-size: 0.85rem !important;
                letter-spacing: 0.5px;
            }
            
            /* Cell padding */
            [data-testid="stDataFrame"] td {
                padding: 12px 15px !important;
            }
            
            /* Alternating row colors */
            [data-testid="stDataFrame"] tr:nth-child(even) td {
                background-color: #f8f9fa !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Create load color conditions
        def get_load_color(load):
            if load > 70:
                return "#ff4b4b"  # Red for high load
            elif load > 40:
                return "#ffab40"  # Orange for medium load
            else:
                return "#4caf50"  # Green for low load
                
        # Display the dataframe with customized columns
        st.dataframe(
            worker_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status",
                    width="small",
                ),
                "Hostname": st.column_config.TextColumn(
                    "Hostname",
                    width="medium",
                ),
                "ID": st.column_config.TextColumn(
                    "ID", 
                    width="medium",
                ),
                "IP Address": st.column_config.TextColumn(
                    "IP Address",
                    width="medium",
                ),
                "Active Tasks": st.column_config.NumberColumn(
                    "Active Tasks",
                    width="small",
                ),
                "Max Tasks": st.column_config.NumberColumn(
                    "Max Tasks",
                    width="small",
                ),
                "Load (%)": st.column_config.ProgressColumn(
                    "Load",
                    width="medium",
                    help="Current worker load percentage",
                    format="%d%%",
                    min_value=0,
                    max_value=100
                ),
                "Last Heartbeat": st.column_config.TextColumn(
                    "Last Heartbeat",
                    width="medium",
                )
            }
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a custom JavaScript function to refresh the page without full reload
    # This is hidden from the user but runs on a timer
    if st.session_state.auto_refresh:
        refresh_js = f"""
        <script>
            // This JavaScript function will automatically refresh components
            // without causing a full page reload
            function refresh() {{
                window.stStreamlitCallbacks.notifyComponentsIframesReady();
                setTimeout(refresh, {st.session_state.refresh_rate * 1000});
            }}
            
            // Start the refresh cycle
            setTimeout(refresh, {st.session_state.refresh_rate * 1000});
        </script>
        """
        st.markdown(refresh_js, unsafe_allow_html=True)
else:
    st.error("Unable to connect to the backend API. Please check your connection settings.")