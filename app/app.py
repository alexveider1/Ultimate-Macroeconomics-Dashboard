import streamlit as st

st.set_page_config(
    page_title="Ultimate Macroeconomics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo("assets/logo.png")


def init_global_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_settings" not in st.session_state:
        st.session_state.user_settings = {
            "theme": "dark",
            "plotly_theme": "plotly",
        }

    if "api_health" not in st.session_state:
        st.session_state.api_health = {
            "db": True,
            "vector_db": True,
            "agent": True,
            "clustering": True,
            "python_sandbox": True,
            "forecaster": True,
            "downloader_extra": True,
        }


def setup_routing():
    p_basic_indicators = st.Page(
        "pages/01_basic_indicators.py", title="General Economics Indicators", icon="📈"
    )
    p_economy_structure = st.Page(
        "pages/02_economy_structure.py", title="Economy Structure", icon="📈"
    )
    p_finance_monetary = st.Page(
        "pages/03_finance_monetary.py", title="Finance and Monetary", icon="📈"
    )
    p_trade = st.Page("pages/04_trade.py", title="Trade and External sector", icon="📈")
    p_demography = st.Page("pages/05_demography.py", title="Demography", icon="📈")
    p_governance = st.Page(
        "pages/06_governance_institutions.py", title="Governance and Institutions", icon="📈"
    )
    p_tech_innovations = st.Page(
        "pages/07_tech_innovation.py", title="Technology and Innovations", icon="📈"
    )
    p_health_wellbeing = st.Page(
        "pages/08_health_wellbeing.py", title="Health and wellbeing", icon="📈"
    )
    p_education_human_capital = st.Page(
        "pages/09_education_human_capital.py",
        title="Education and Human Capital",
        icon="📈",
    )
    p_environment = st.Page(
        "pages/10_environment.py", title="Environment and Sustainability", icon="📈"
    )
    p_agent = st.Page("pages/11_ai_agent_chat.py", title="AI Analyst", icon="🤖")
    p_custom_plot = st.Page(
        "pages/12_custom_plot_builder.py", title="Custom Plot Constructor", icon="📊"
    )
    p_cluster = st.Page(
        "pages/13_clustering_sandbox.py", title="Clustering Sandbox", icon="🔍"
    )
    p_yahoo_finance = st.Page(
        "pages/14_yahoo_finance.py", title="Yahoo Finance", icon="💹"
    )
    p_news = st.Page("pages/15_news.py", title="News Explorer", icon="📰")
    p_settings = st.Page("pages/16_settings.py", title="System Settings", icon="⚙️")

    pg = st.navigation(
        {
            "Dashboard": [
                p_basic_indicators,
                p_economy_structure,
                p_finance_monetary,
                p_trade,
                p_demography,
                p_governance,
                p_tech_innovations,
                p_health_wellbeing,
                p_education_human_capital,
                p_environment,
            ],
            "Other data": [p_yahoo_finance, p_news],
            "Constructors": [p_custom_plot, p_cluster],
            "AI": [p_agent],
            "Settings": [p_settings],
        }
    )

    return pg


def _show_data_disclaimer():
    if st.session_state.get("disclaimer_accepted"):
        return

    @st.dialog("Data Disclaimer", width="large")
    def _disclaimer_dialog():
        st.warning(
            "**Important:** The developer of this dashboard is not responsible for "
            "the accuracy, completeness, or quality of the data and news displayed. "
            "All information is sourced from third-party providers and is presented "
            "as-is. It is the user's responsibility to evaluate whether any given "
            "source or data point is reliable before making decisions based on it."
        )
        if st.button("I understand", type="primary", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()

    _disclaimer_dialog()


init_global_state()
_show_data_disclaimer()

pg = setup_routing()
pg.run()
