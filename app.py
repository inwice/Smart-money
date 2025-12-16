import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ตั้งค่าหน้าเว็บให้กว้าง
st.set_page_config(page_title="Smart Money Detector (HMM)", layout="wide")

# ==========================================
# 1. CLASS & FUNCTIONS
# ==========================================

class SmartMoneyHMM:
    def __init__(self, ticker, period='1y', interval='1d', n_states=4):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.n_states = n_states
        self.df = None
        self.model = None
        self.accum_state_id = None
        self.accum_stats = {}

    def fetch_data(self):
        try:
            # ดึงข้อมูลจาก yfinance
            self.df = yf.download(self.ticker, period=self.period, interval=self.interval, progress=False)
            
            # จัดการ Multi-level columns (สำหรับ yfinance เวอร์ชั่นใหม่)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)
            
            if self.df.empty:
                return False
                
            self.df = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            # ลบแถวที่ Volume เป็น 0 (วันหยุดหรือข้อมูลผิดพลาด)
            self.df = self.df[self.df['Volume'] > 0]
            return True
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
            return False

    def add_indicators(self):
        df = self.df.copy()
        
        # 1. Log Returns (Price Action)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Bollinger Bands (Volatility)
        window = 20
        std = df['Close'].rolling(window).std()
        sma = df['Close'].rolling(window).mean()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        range_bb = upper - lower
        df['BB_Width'] = range_bb / sma
        
        # 3. RSI (Momentum)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 4. Relative Volume (Volume Profile Proxy)
        df['Rel_Vol'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, 1)

        # Drop NaN
        self.df = df.dropna()

    def train_hmm(self):
        # Features สำหรับ HMM: เน้น Volatility, Returns, RSI
        feature_cols = ['Log_Ret', 'BB_Width', 'RSI', 'Rel_Vol']
        X = self.df[feature_cols].values
        
        # Scale ข้อมูล
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train HMM
        self.model = GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=1000, random_state=42)
        self.model.fit(X_scaled)
        
        # ทำนาย States
        self.df['HMM_State'] = self.model.predict(X_scaled)

    def identify_accumulation_state(self):
        """ระบุว่า State ไหนคือ Accumulation (Smart Money เก็บของ)"""
        state_stats = []
        
        for i in range(self.n_states):
            state_data = self.df[self.df['HMM_State'] == i]
            if len(state_data) == 0: continue
            
            stats = {
                'State': i,
                'Volatility': state_data['Log_Ret'].std(), # ความผันผวน
                'Avg_RSI': state_data['RSI'].mean(),
                'Avg_Rel_Vol': state_data['Rel_Vol'].mean(),
                'Count': len(state_data)
            }
            state_stats.append(stats)
            
        # Logic: Accumulation คือช่วงที่ Volatility ต่ำที่สุด (ราคานิ่งๆ)
        sorted_stats = sorted(state_stats, key=lambda x: x['Volatility'])
        best_candidate = sorted_stats[0]
        
        self.accum_state_id = best_candidate['State']
        
        # คำนวณ VWAP เฉพาะช่วง Accumulation
        accum_data = self.df[self.df['HMM_State'] == self.accum_state_id]
        vwap = (accum_data['Close'] * accum_data['Volume']).sum() / accum_data['Volume'].sum()
        
        self.accum_stats = {
            'vwap': vwap,
            'count': len(accum_data),
            'volatility': best_candidate['Volatility']
        }

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.title("🤖 Smart Money Detector (HMM AI)")
st.markdown("""
ระบบค้นหาพฤติกรรม **Smart Money (Accumulation)** โดยใช้ **Hidden Markov Model (Unsupervised Learning)** เพื่อแบ่งสภาวะตลาดจาก Price Action, Volume, และ Volatility
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    ticker = st.text_input("ชื่อหุ้น (Symbol)", value="BTC-USD", help="เช่น CPALL.BK, DELTA.BK, AAPL, BTC-USD")
    
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("ย้อนหลัง", options=['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
    with col2:
        interval = st.selectbox("Timeframe", options=['1d', '1wk'], index=0)
        
    n_states = st.slider("จำนวน States (Market Regimes)", 2, 6, 4, help="ปกติ 4 (Accumulation, Markup, Distribution, Markdown)")
    
    run_btn = st.button("🚀 วิเคราะห์ผล", type="primary")

# --- Main Process ---
if run_btn:
    with st.spinner(f"กำลังดึงข้อมูลและวิเคราะห์ {ticker}..."):
        model = SmartMoneyHMM(ticker, period, interval, n_states)
        success = model.fetch_data()
        
        if success:
            model.add_indicators()
            model.train_hmm()
            model.identify_accumulation_state()
            
            # --- ส่วนแสดงผล Metrics ---
            df = model.df
            last_price = df['Close'].iloc[-1]
            accum_price = model.accum_stats['vwap']
            
            # คำนวณ % Gap จากต้นทุนเจ้ามือ
            gap_percent = ((last_price - accum_price) / accum_price) * 100
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("ราคาปัจจุบัน", f"{last_price:,.2f}")
            col_m2.metric("ต้นทุน Smart Money (Acc. VWAP)", f"{accum_price:,.2f}", f"{gap_percent:+.2f}%")
            col_m3.metric("Accumulation State ID", f"State {model.accum_state_id}", help="คือ State ที่มีความผันผวนต่ำที่สุด")
            
            # --- สร้างกราฟด้วย Plotly ---
            
            # กำหนดสีให้แต่ละ State
            colors = ['#EF4444', '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899'] # Red, Blue, Green, Orange...
            state_colors = {i: colors[i % len(colors)] for i in range(n_states)}
            
            # Create Subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3],
                                subplot_titles=(f"Price Action & Market Regimes ({ticker})", "RSI & Volume"))

            # 1. Main Price Chart (Line)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='gray', width=1),
                opacity=0.5
            ), row=1, col=1)

            # 2. Add Colored Markers for States
            for state_id in range(n_states):
                mask = df['HMM_State'] == state_id
                state_name = f"State {state_id}"
                if state_id == model.accum_state_id:
                    state_name += " (Accumulation 🟢)"
                
                fig.add_trace(go.Scatter(
                    x=df.index[mask], y=df['Close'][mask],
                    mode='markers',
                    name=state_name,
                    marker=dict(color=state_colors[state_id], size=6 if state_id == model.accum_state_id else 4),
                    opacity=0.8
                ), row=1, col=1)

            # 3. Bollinger Bands Area (Optional - visual guide)
            # เราจะไม่ plot เส้นรกๆ แต่จะ plot accumulation area
            
            # 4. RSI (Lower Chart)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Update Layout
            fig.update_layout(
                height=700,
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
            )
            
            # Range Selector buttons
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # --- ตารางข้อมูลดิบ ---
            with st.expander("ดูข้อมูลดิบ (Data Table)"):
                display_df = df[['Close', 'Volume', 'RSI', 'Rel_Vol', 'BB_Width', 'HMM_State']].copy()
                display_df = display_df.sort_index(ascending=False)
                
                # Highlight Accumulation rows
                def highlight_accum(row):
                    if row['HMM_State'] == model.accum_state_id:
                        return ['background-color: #1a472a'] * len(row) # สีเขียวเข้ม
                    return [''] * len(row)
                
                st.dataframe(display_df.style.apply(highlight_accum, axis=1), use_container_width=True)

        else:
            st.warning("ไม่พบข้อมูล กรุณาตรวจสอบชื่อหุ้นหรือช่วงเวลา")
else:
    st.info("👈 กรุณากรอกชื่อหุ้นที่ Sidebar ด้านซ้าย แล้วกดปุ่ม 'วิเคราะห์ผล'")
