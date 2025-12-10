import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="JST Macrohistory Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    """Loads the Excel file and ensures proper data types."""
    try:
        df = pd.read_excel(file)
        # Ensure column names are lower case to match requirements
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def calculate_indices(df):
    """
    Calculates cumulative indices for Nominal and Real returns.
    Assumes input dataframe is sorted by year.
    """
    # 1. Nominal Accumulation (Wealth Index)
    # We assume returns are in decimals (e.g., 0.05 for 5%). 
    # If the user data is in percent (e.g., 5.0), this logic needs adjustment, 
    # but standard JST data is usually decimals.
    
    # We use cumprod() to compound the returns
    # We add a 'base' of 1.0 to start the series calculation
    df['eq_idx_nominal'] = (1 + df['eq_tr']).cumprod()
    df['bond_idx_nominal'] = (1 + df['bond_tr']).cumprod()
    
    # CPI is already an index, but we ensure it's treated as float
    df['cpi_idx'] = df['cpi'].astype(float)
    
    # 2. Real Accumulation
    # Real Wealth = Nominal Wealth / Price Level
    # We simply divide the calculated Nominal Index by the CPI Index
    df['eq_idx_real'] = df['eq_idx_nominal'] / df['cpi_idx']
    df['bond_idx_real'] = df['bond_idx_nominal'] / df['cpi_idx']
    
    return df

def rebase_series(series):
    """
    Rebases a pandas Series so that the first value is 100.
    Used for comparing different assets starting from the same point in time.
    """
    if len(series) == 0:
        return series
    first_value = series.iloc[0]
    if first_value == 0:
        return series # Avoid division by zero
    return (series / first_value) * 100

def format_number_eu(val):
    """Formats float to string with . as thousands and , as decimal."""
    # Standard format first: 1,234.56
    s = f"{val:,.2f}"
    # Swap separators: 1.234,56
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def format_percent_eu(val):
    """Formats float to percentage string with , as decimal."""
    s = f"{val:.2%}"
    return s.replace(".", ",")

def create_chart(data, traces, log_scale):
    """
    Helper to generate Plotly figures for different tabs to avoid repetition.
    traces: list of dicts with keys 'col', 'name', 'color', 'dash'
    """
    fig = go.Figure()
    for t in traces:
        fig.add_trace(go.Scatter(
            x=data['year'],
            y=data[t['col']],
            mode='lines',
            name=t['name'],
            line=dict(color=t['color'], dash=t.get('dash', 'solid'), width=2),
            hovertemplate='%{y:,.2f}<extra></extra>' # Force EU formatting in hover
        ))

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Year",
        # We move yaxis config here to apply tickformat
        yaxis=dict(
            title="Accumulated Return (Base=100)",
            type="log" if log_scale else "linear",
            tickformat=",.0f" # Force full numbers (1.000) instead of SI (1k)
        ),
        separators=",.", # Force comma for decimal, dot for thousands
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------
def main():
    st.title("📊 JST Macrohistory: Return Analyzer")
    
    with st.expander("ℹ️ Data Format & Logic Guide"):
        st.markdown("""
        **How the app interprets your data:**
        * **Returns (`eq_tr`, `bond_tr`):** Expected as **decimals** (e.g., `0.05` for 5%, not `5.0`). The app calculates the cumulative product of `(1 + return)`.
        * **CPI (`cpi`):** Expected as a **Price Index Level** (e.g., `100` or `4.5`).
            * *Note:* The app handles any base year (e.g., 1990=100) automatically. It uses the ratio between years to determine inflation, so the absolute starting number doesn't matter.
        * **Real Returns:** Calculated as `Nominal Index / CPI Index`.
        * **Chart Base:** All charts (including CPI) are **rebased to 100** at the start of your selected period. This allows you to easily compare relative performance regardless of the raw index levels.
        """)

    st.markdown("""
    This tool visualizes **Nominal vs. Real** returns for Equities and Bonds based on the JST Macrohistory dataset format.
    Upload your `.xlsx` file to begin.
    """)

    # 1. File Upload
    uploaded_file = st.file_uploader("Upload JST Data (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df_raw = load_data(uploaded_file)
        
        if df_raw is not None:
            # Validate Columns
            required_cols = {'year', 'country', 'cpi', 'eq_tr', 'bond_tr'}
            if not required_cols.issubset(df_raw.columns):
                st.error(f"Missing columns! The file must contain: {', '.join(required_cols)}")
                st.stop()

            # 2. Sidebar Controls
            st.sidebar.header("Configuration")
            
            # Country Selector
            countries = sorted(df_raw['country'].unique())
            selected_country = st.sidebar.selectbox("Select Country", countries)
            
            # Chart Scale Selector
            use_log_scale = st.sidebar.checkbox("Use Logarithmic Scale", value=True)

            # Filter Data by Country first (to get correct year ranges)
            country_data = df_raw[df_raw['country'] == selected_country].sort_values('year').copy()
            
            # Date Range Selector
            min_year = int(country_data['year'].min())
            max_year = int(country_data['year'].max())
            
            selected_years = st.sidebar.slider(
                "Select Period",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            
            # 3. Data Processing & Calculation
            # Filter by date
            mask = (country_data['year'] >= selected_years[0]) & (country_data['year'] <= selected_years[1])
            chart_data = country_data.loc[mask].copy()

            if chart_data.empty:
                st.warning("No data available for the selected period.")
                st.stop()

            # Perform Calculations
            full_country_data = calculate_indices(country_data)
            chart_data = full_country_data.loc[mask].copy()
            
            # Rebase
            chart_data['Display CPI'] = rebase_series(chart_data['cpi_idx'])
            chart_data['Display Eq Nominal'] = rebase_series(chart_data['eq_idx_nominal'])
            chart_data['Display Eq Real'] = rebase_series(chart_data['eq_idx_real'])
            chart_data['Display Bond Nominal'] = rebase_series(chart_data['bond_idx_nominal'])
            chart_data['Display Bond Real'] = rebase_series(chart_data['bond_idx_real'])

            # 4. Visualization
            st.subheader(f"Asset Performance: {selected_country} ({selected_years[0]} - {selected_years[1]})")
            st.caption("All values rebased to 100 at the start of the selected period.")

            # Define Plot Traces
            t_cpi = {'col': 'Display CPI', 'name': 'CPI (Inflation)', 'color': 'gray', 'dash': 'dot'}
            t_eq_nom = {'col': 'Display Eq Nominal', 'name': 'Equities (Nominal)', 'color': '#1f77b4', 'dash': 'solid'}
            t_eq_real = {'col': 'Display Eq Real', 'name': 'Equities (Real)', 'color': '#1f77b4', 'dash': 'dash'}
            t_bond_nom = {'col': 'Display Bond Nominal', 'name': 'Bonds (Nominal)', 'color': '#ff7f0e', 'dash': 'solid'}
            t_bond_real = {'col': 'Display Bond Real', 'name': 'Bonds (Real)', 'color': '#ff7f0e', 'dash': 'dash'}

            # Create Tabs
            tab_all, tab_nom, tab_real = st.tabs(["All Data", "Nominal Returns", "Real Returns"])

            with tab_all:
                st.markdown("##### Overview: Nominal vs Real")
                fig_all = create_chart(chart_data, [t_cpi, t_eq_nom, t_eq_real, t_bond_nom, t_bond_real], use_log_scale)
                st.plotly_chart(fig_all, use_container_width=True)

            with tab_nom:
                st.markdown("##### Nominal Returns (Before Inflation)")
                fig_nom = create_chart(chart_data, [t_cpi, t_eq_nom, t_bond_nom], use_log_scale)
                st.plotly_chart(fig_nom, use_container_width=True)
                
            with tab_real:
                st.markdown("##### Real Returns (Purchasing Power)")
                fig_real = create_chart(chart_data, [t_cpi, t_eq_real, t_bond_real], use_log_scale)
                st.plotly_chart(fig_real, use_container_width=True)

            # 5. Summary Stats Table
            with st.expander("See Underlying Data & Statistics"):
                # Calculate CAGR for the period
                # CAGR = (End_Value / Start_Value)^(1/n) - 1
                n_years = max(1, len(chart_data) - 1)
                
                stats = []
                for col, name in [
                    ('Display Eq Nominal', 'Equity Nominal'),
                    ('Display Eq Real', 'Equity Real'),
                    ('Display Bond Nominal', 'Bond Nominal'),
                    ('Display Bond Real', 'Bond Real'),
                    ('Display CPI', 'CPI')
                ]:
                    start_val = chart_data[col].iloc[0]
                    end_val = chart_data[col].iloc[-1]
                    total_ret = (end_val / start_val) - 1
                    cagr = (end_val / start_val) ** (1/n_years) - 1
                    
                    stats.append({
                        "Metric": name,
                        "Start Value": format_number_eu(start_val),
                        "End Value": format_number_eu(end_val),
                        "Total Return": format_percent_eu(total_ret),
                        "CAGR": format_percent_eu(cagr)
                    })
                
                st.dataframe(pd.DataFrame(stats))
                st.write("### Raw Data (Processed)")
                st.dataframe(chart_data)

if __name__ == "__main__":
    main()