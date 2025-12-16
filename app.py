import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ตั้งค่าหน้าเว็บ
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
        self.state_props = {} # เก็บสีและชื่อของแต่ละ State

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period=self.period, interval=self.interval, progress=False)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)
            
            if self.df.empty: return False
            self.df = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            self.df = self.df[self.df['Volume'] > 0]
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

    def add_indicators(self):
        df = self.df.copy()
        # 1. Log Returns
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        # 2. Bollinger Bands Width (Volatility)
        window = 20
        sma = df['Close'].rolling(window).mean()
        std = df['Close'].rolling(window).std()
        df['BB_Width'] = (4 * std) / sma # 4 std = (Upper - Lower)
        # 3. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        df['RSI'] = 100 - (100 / (1 + rs))
        # 4. Rel Volume
        df['Rel_Vol'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, 1)

        self.df = df.dropna()

    def train_hmm(self):
        feature_cols = ['Log_Ret', 'BB_Width', 'RSI', 'Rel_Vol']
        X = self.df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.model = GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=1000, random_state=42)
        self.model.fit(X_scaled)
        self.df['HMM_State'] = self.model.predict(X_scaled)

    def interpret_states(self):
        """วิเคราะห์คุณสมบัติของแต่ละ State เพื่อระบุสีและหน้าที่"""
        state_stats = {}
        for i in range(self.n_states):
            mask = self.df['HMM_State'] == i
            if not mask.any(): continue
            state_stats[i] = {
                'volatility': self.df.loc[mask, 'BB_Width'].mean(),
                'return': self.df.loc[mask, 'Log_Ret'].mean(),
                'rsi': self.df.loc[mask, 'RSI'].mean()
            }
        
        self.state_props = {} 

        # --- Logic การให้สีตามทฤษฎี Wyckoff (สำหรับ 4 States) ---
        if self.n_states == 4:
            # 1. Accumulation (สีเขียว): ความผันผวนต่ำที่สุด
            accum_id = min(state_stats, key=lambda k: state_stats[k]['volatility'])
            self.state_props[accum_id] = {'color': '#10B981', 'label': 'Accumulation (เก็บของ)'} # Green
            
            # ลบ Accum ออกจากรายการที่จะหาต่อ
            remaining = [k for k in state_stats if k != accum_id]
            
            # 2. Markdown (สีแดง): ผลตอบแทนเฉลี่ยแย่ที่สุด (ลบมากสุด)
            markdown_id = min(remaining, key=lambda k: state_stats[k]['return'])
            self.state_props[markdown_id] = {'color': '#EF4444', 'label': 'Markdown (ขาลง/ทุบ)'} # Red
            
            remaining = [k for k in remaining if k != markdown_id]
            
            # 3. Distribution (สีส้ม): ความผันผวนสูงที่สุดในกลุ่มที่เหลือ (ผันผวนบนยอดดอย)
            dist_id = max(remaining, key=lambda k: state_stats[k]['volatility'])
            self.state_props[dist_id] = {'color': '#F97316', 'label': 'Distribution (ระบายของ)'} # Orange
            
            remaining = [k for k in remaining if k != dist_id]
            
            # 4. Markup (สีฟ้า): ตัวสุดท้าย (มักเป็นขาขึ้นปกติ)
            if remaining:
                markup_id = remaining[0]
                self.state_props[markup_id] = {'color': '#3B82F6', 'label': 'Markup (ขาขึ้น)'} # Blue
            
            self.accum_state_id = accum_id

        else:
            # กรณีไม่ได้เลือก 4 States ให้ใช้ Logic ง่ายๆ
            # หา Accumulation แล้วที่เหลือสุ่มสี
            accum_id = min(state_stats, key=lambda k: state_stats[k]['volatility'])
            self.accum_state_id = accum_id
            
            colors = ['#6B7280', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6']
            for i in state_stats:
                if i == accum_id:
                    self.state_props[i] = {'color': '#10B981', 'label': 'Accumulation (เก็บของ)'}
                else:
                    color = colors[i % len(colors)]
                    self.state_props[i] = {'color': color, 'label': f'State {i}'}

        # คำนวณ Stats ของ Accumulation เพื่อใช้แสดงผล
        accum_data = self.df[self.df['HMM_State'] == self.accum_state_id]
        if not accum_data.empty:
            vwap = (accum_data['Close'] * accum_data['Volume']).sum() / accum_data['Volume'].sum()
            self.accum_stats = {'vwap': vwap, 'count': len(accum_data)}

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.title("🤖 Smart Money Detector (AI Colors)")
st.caption("Auto-detect Market Regimes: Accumulation (Green), Markup (Blue), Distribution (Orange), Markdown (Red)")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Symbol", value="BTC-USD")
    col1, col2 = st.columns(2)
    with col1: period = st.selectbox("Period", ['6mo', '1y', '2y', '5y'], index=1)
    with col2: interval = st.selectbox("Timeframe", ['1d', '1wk'], index=0)
    # ล็อคไว้ที่ 4 เพื่อให้สีตรงตามทฤษฎี
    n_states = st.slider("States", 2, 6, 4, disabled=False) 
    run_btn = st.button("Analyze", type="primary")

# --- Main ---
if run_btn:
    with st.spinner(f"Analyzing {ticker}..."):
        model = SmartMoneyHMM(ticker, period, interval, n_states)
        if model.fetch_data():
            model.add_indicators()
            model.train_hmm()
            model.interpret_states() # เรียกใช้ฟังก์ชันแปลความหมายสี
            
            # Metrics
            df = model.df
            last_price = df['Close'].iloc[-1]
            accum_vwap = model.accum_stats.get('vwap', 0)
            gap = ((last_price - accum_vwap) / accum_vwap) * 100 if accum_vwap else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"{last_price:,.2f}")
            c2.metric("Smart Money Cost (VWAP)", f"{accum_vwap:,.2f}", f"{gap:+.2f}%")
            c3.metric("Current State", model.state_props[df['HMM_State'].iloc[-1]]['label'])

            # Plotly Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

            # 1. Price Line
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='gray', width=1), opacity=0.3), row=1, col=1)

            # 2. Colored Dots
            # วาดทีละ State เพื่อให้ชื่อใน Legend ถูกต้อง
            # เรียงลำดับ Accumulation ขึ้นก่อน
            sorted_states = sorted(model.state_props.keys(), key=lambda x: 0 if x == model.accum_state_id else 1)
            
            for state_id in sorted_states:
                mask = df['HMM_State'] == state_id
                props = model.state_props[state_id]
                
                fig.add_trace(go.Scatter(
                    x=df.index[mask], 
                    y=df['Close'][mask],
                    mode='markers',
                    name=props['label'],
                    marker=dict(color=props['color'], size=5),
                    opacity=0.9
                ), row=1, col=1)

            # 3. RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#A78BFA', width=1)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

            fig.update_layout(height=600, template="plotly_dark", hovermode="x unified", margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # คำอธิบาย
            st.info("""
            **Color Guide:**
            - 🟢 **Accumulation (สีเขียว):** เก็บของ/พักตัว (ซื้อเมื่อราคาอยู่ใกล้ VWAP)
            - 🔵 **Markup (สีฟ้า):** ขาขึ้น/ไล่ราคา (ถือ Run Trend)
            - 🟠 **Distribution (สีส้ม):** ผันผวนสูง/ระบายของ (ระวังดอย/ทยอยขาย)
            - 🔴 **Markdown (สีแดง):** ขาลง/ทุบ (ห้ามรับมีด)
            """)
        else:
            st.error("Data not found.")


