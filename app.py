import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from datetime import timedelta
import re
from FinMind.data import DataLoader
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ==========================================
# 介面設計：Apple.com 現代簡約風格 (自適應深淺色)
# ==========================================
st.set_page_config(page_title="Quantfolio | 雙引擎量化決策", page_icon="", layout="wide")

apple_css = """
<style>
    /* 引入 Apple 風格字體 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* 隱藏預設的 Streamlit 選單與 Footer，看起來更像獨立 App */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 數字卡片 (Metric) 蘋果擬真卡片化設計 */
    div[data-testid="metric-container"] {
        background-color: rgba(150, 150, 150, 0.08); /* 適應深淺色的半透明背景 */
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(150, 150, 150, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.05);
    }

    /* 標題與按鈕的 Apple 風格 */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    .stButton>button {
        border-radius: 980px; /* 藥丸型按鈕 */
        font-weight: 600;
        padding: 0.5rem 2rem;
        background-color: #0071E3; /* Apple Blue */
        color: white;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0077ED;
        color: white;
    }
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

# ==========================================
# 核心邏輯工具
# ==========================================
def parse_investment_horizon(text):
    text = text.replace(" ", "")
    if any(k in text for k in ["存股", "長期", "不賣", "退休"]): return "5年以上"
    if any(k in text for k in ["短線", "當沖", "隔日沖"]): return "1-3個月"
    if "半年" in text: return "3-6個月"
    replace_map = {"一": "1", "兩": "2", "二": "2", "三": "3", "四": "4", "五": "5", "十年": "10年"}
    for k, v in replace_map.items(): text = text.replace(k, v)
    if "年" in text:
        match = re.search(r'(\d+(?:\.\d+)?)\s*年', text)
        if match:
            y = float(match.group(1))
            if y < 1: return "3-6個月"
            elif y <= 1: return "1年"
            elif y <= 3: return "1-3年"
            elif y <= 5: return "3-5年"
            else: return "5年以上"
        return "1-3年"
    if "月" in text:
        match = re.search(r'(\d+(?:\.\d+)?)\s*個?月', text)
        if match:
            m = float(match.group(1))
            if m <= 3: return "1-3個月"
            elif m <= 6: return "3-6個月"
            elif m <= 12: return "1年"
            elif m <= 36: return "1-3年"
            else: return "3-5年"
        return "3-6個月"
    if any(k in text for k in ["天", "日", "周", "週"]): return "1-3個月"
    return "1年"

def get_horizon_years(horizon_str):
    return {"1-3個月": 0.25, "3-6個月": 0.5, "1年": 1.0, "1-3年": 2.0, "3-5年": 4.0, "5年以上": 5.0}.get(horizon_str, 1.0)

# ==========================================
# 模組 B：個股與ETF共用量化評估系統 (重構為回傳 Dictionary)
# ==========================================
class UnifiedEvaluator:
    def __init__(self, raw_ticker, ticker_yf, stock_name, market_label, matched_horizon_str, fm, is_etf):
        self.raw_ticker, self.ticker = raw_ticker, ticker_yf
        self.stock_name, self.market_label, self.is_etf = stock_name, market_label, is_etf
        self.matched_horizon = matched_horizon_str
        self.fm, self.stock = fm, yf.Ticker(self.ticker)
        self.df, self.info = pd.DataFrame(), {}
        self.report_data = {}
        
        self.horizons = {
            "1-3個月": {"fund": 0.1, "tech": 0.6, "chip": 0.3},
            "3-6個月": {"fund": 0.2, "tech": 0.5, "chip": 0.3},
            "1年": {"fund": 0.4, "tech": 0.4, "chip": 0.2},
            "1-3年": {"fund": 0.6, "tech": 0.3, "chip": 0.1},
            "3-5年": {"fund": 0.8, "tech": 0.2, "chip": 0.0},
            "5年以上": {"fund": 0.9, "tech": 0.1, "chip": 0.0}
        }

    def fetch_data(self):
        self.df = self.stock.history(period="1y") # 延長為 1 年以利畫 K 線圖
        try: self.info = self.stock.info or {}
        except: self.info = {}
        if self.df.empty: 
            if self.ticker.endswith('.TW'):
                self.ticker = self.ticker.replace('.TW', '.TWO')
                self.stock = yf.Ticker(self.ticker)
                self.df = self.stock.history(period="1y")
            if self.df.empty: raise ValueError(f"無法獲取 {self.raw_ticker} 的股價數據。")

    def calculate_indicators(self):
        df = self.df
        df['MA5'], df['MA20'], df['MA60'] = df['Close'].rolling(5).mean(), df['Close'].rolling(20).mean(), df['Close'].rolling(60).mean()
        low_min, high_max = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
        price_diff = (high_max - low_min).replace(0, np.nan)
        df['RSV'] = ((df['Close'] - low_min) / price_diff) * 100
        df['RSV'] = df['RSV'].fillna(50)
        df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    def analyze_all(self):
        # 這裡簡化評分邏輯以符合 UI 化需求，保留核心精神
        score_f, score_t, score_c = 0, 0, 0
        details_f, details_t, details_c = [], [], []
        latest, prev = self.df.iloc[-1], self.df.iloc[-2]

        # 1. 基本面/殖利率 (ETF 不看 PE/PB，主要看 Yield 與長期抗跌)
        pe = self.info.get('trailingPE', 999) if not self.is_etf else 999
        yield_pct = (self.info.get('dividendYield', 0) or 0) * 100
        roe = (self.info.get('returnOnEquity', 0) or 0) * 100

        if not self.is_etf:
            if 0 < pe < 20: score_f += 15; details_f.append(f"估值偏低 (PE: {pe:.1f})")
            elif 20 <= pe <= 30: score_f += 5; details_f.append(f"估值合理 (PE: {pe:.1f})")
            if roe > 10: score_f += 10; details_f.append(f"高資本回報 (ROE: {roe:.1f}%)")
        else:
            score_f += 15 # ETF 給予基本面基礎分
            
        if yield_pct > 4: score_f += 5; details_f.append(f"具備高殖利率保護 ({yield_pct:.1f}%)")

        # 2. 技術面
        if latest['Close'] > latest['MA20']: score_t += 15; details_t.append("站上月線，短線強勢")
        if latest['K'] < 30: score_t += 15; details_t.append(f"KD 超賣區，具備反彈契機 (K:{latest['K']:.1f})")
        if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0: score_t += 10; details_t.append("MACD 柱狀體翻紅")

        # 3. 籌碼面 (簡化為 FinMind 近三日法人)
        inst_data = []
        if self.market_label != "美股/全球":
            start_date = (datetime.date.today() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
            try:
                df_inst = self.fm.taiwan_stock_institutional_investors(stock_id=self.raw_ticker, start_date=start_date)
                if not df_inst.empty:
                    recent = df_inst.groupby('name')['buy', 'sell'].sum()
                    net_foreign = recent.loc['外資及陸資(不含外資自營商)']['buy'] - recent.loc['外資及陸資(不含外資自營商)']['sell'] if '外資及陸資(不含外資自營商)' in recent.index else 0
                    net_sitc = recent.loc['投信']['buy'] - recent.loc['投信']['sell'] if '投信' in recent.index else 0
                    inst_data = [net_foreign/1000, net_sitc/1000] # 千張
                    if net_foreign > 0 and net_sitc > 0: score_c += 20; details_c.append("土洋法人同步買超")
                    elif net_sitc > 0: score_c += 10; details_c.append("投信積極建倉")
            except: score_c += 10 # API失敗預設分
        else: score_c += 10

        # 計算權重總分
        w = self.horizons[self.matched_horizon]
        f_pct, t_pct, c_pct = min(score_f/30*100, 100), min(score_t/40*100, 100), min(score_c/20*100, 100)
        total_score = (f_pct * w['fund']) + (t_pct * w['tech']) + (c_pct * w['chip'])

        advice = "強烈建議進場" if total_score >= 75 else ("建議進場" if total_score >= 60 else ("觀望為主" if total_score >= 45 else "不建議進場"))

        # 打包回傳
        self.report_data = {
            "current_price": latest['Close'],
            "price_change": latest['Close'] - prev['Close'],
            "price_change_pct": (latest['Close'] - prev['Close']) / prev['Close'] * 100,
            "volume": latest['Volume'],
            "total_score": total_score,
            "advice": advice,
            "details": {"fund": details_f, "tech": details_t, "chip": details_c},
            "pe": pe, "yield": yield_pct, "roe": roe,
            "inst_data": inst_data,
            "support": self.df['Low'].tail(20).min(),
            "df": self.df
        }

class MasterRoutingSystem:
    def __init__(self):
        self.fm = DataLoader()
    def auto_detect(self, raw_ticker):
        ticker_yf, is_etf, stock_name, market_label = f"{raw_ticker}.TW", False, raw_ticker, "台股"
        if not any(char.isdigit() for char in raw_ticker):
            ticker_yf, market_label = raw_ticker, "美股/全球"
        try:
            info = yf.Ticker(ticker_yf).info or {}
            is_etf = True if info.get('quoteType') == 'ETF' else False
            stock_name = info.get('shortName', info.get('longName', raw_ticker))
        except: pass
        return ticker_yf, is_etf, stock_name, market_label

# ==========================================
# 網頁 UI 綁定層 (Apple Style Layout)
# ==========================================
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'> Quantfolio 智能決策</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    raw_ticker = st.text_input("搜尋股票或 ETF (如: 2330, 0050, AAPL)", value="2330").strip().upper().replace('.TW', '')
with col2:
    horizon_input = st.selectbox("投資時間年限規劃", ["1-3個月 (短線/波段)", "3-6個月 (中短線)", "1年 (中期佈局)", "1-3年 (中長線)", "3-5年 (長線成長)", "5年以上 (存股退休)"], index=2)

if st.button("Generate Analysis 產生分析"):
    if raw_ticker:
        with st.spinner("Analyzing market data..."):
            system = MasterRoutingSystem()
            matched_horizon = horizon_input.split(" ")[0]
            ticker_yf, is_etf, stock_name, market_label = system.auto_detect(raw_ticker)
            
            evaluator = UnifiedEvaluator(raw_ticker, ticker_yf, stock_name, market_label, matched_horizon, system.fm, is_etf)
            evaluator.fetch_data()
            evaluator.calculate_indicators()
            evaluator.analyze_all()
            data = evaluator.report_data
            df = data['df']

            # --- 頂部摘要卡片 (Hero Section) ---
            st.markdown(f"### {stock_name} ({raw_ticker}) <span style='color: gray; font-size: 0.6em;'>{market_label}</span>", unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("目前報價", f"{data['current_price']:,.2f}", f"{data['price_change']:+.2f} ({data['price_change_pct']:+.2f}%)")
            m2.metric("量化綜合評分", f"{data['total_score']:.1f} / 100", data['advice'], delta_color="off" if data['total_score']>=60 else "inverse")
            m3.metric("黃金防守區間", f"{data['support']:,.2f}", "近20日低點", delta_color="off")
            m4.metric("本益比 (PE)" if not is_etf else "屬性", f"{data['pe']:.1f}" if data['pe']!=999 else "N/A", "ETF/無" if is_etf else "")

            st.write("---")

            # --- 蘋果風格的多頁籤設計 ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 K線與價量", "🎯 量化診斷報告", "🏦 籌碼與主力", "💰 股利與基本", "⚡ 進階盤口(明細)"])

            with tab1:
                st.markdown("#### 互動式 K 線與指標 (Interactive Chart)")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                # K 線
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1.5), name="月線 (MA20)"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1.5), name="季線 (MA60)"), row=1, col=1)
                # 成交量
                colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
                
                fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, xaxis_rangeslider_visible=False, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.markdown("#### 機器學習權重解析 (AI Diagnostics)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info("**基本價值面**")
                    for d in data['details']['fund']: st.write(f"• {d}")
                    if not data['details']['fund']: st.write("無明顯特徵")
                with c2:
                    st.warning("**技術型態面**")
                    for d in data['details']['tech']: st.write(f"• {d}")
                    if not data['details']['tech']: st.write("盤整中立")
                with c3:
                    st.success("**籌碼大戶面**")
                    for d in data['details']['chip']: st.write(f"• {d}")
                    if not data['details']['chip']: st.write("籌碼無異常")

            with tab3:
                st.markdown("#### 近期三大法人進出 (Institutional Momentum)")
                if data['inst_data']:
                    st.write("以下為 FinMind 串接之台灣集保/交易所近期籌碼變化：")
                    ci1, ci2 = st.columns(2)
                    ci1.metric("外資近10日淨買賣 (千張)", f"{data['inst_data'][0]:+,.1f}")
                    ci2.metric("投信近10日淨買賣 (千張)", f"{data['inst_data'][1]:+,.1f}")
                else:
                    st.write("海外標的或 API 限制，暫無台灣法人數據。")

            with tab4:
                st.markdown("#### 股利與資產品質 (Dividends & Quality)")
                cd1, cd2, cd3 = st.columns(3)
                cd1.metric("近四季 ROE", f"{data['roe']:.2f}%" if data['roe']>0 else "N/A")
                cd2.metric("預估殖利率 (Yield)", f"{data['yield']:.2f}%" if data['yield']>0 else "N/A")
                
                perf_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 if len(df)>20 else 0
                perf_6m = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                cd3.metric("近半年績效表現", f"{perf_6m:+.2f}%", f"近一月 {perf_1m:+.2f}%")

            with tab5:
                st.markdown("#### ⚡ 盤口與即時明細 (Tick & Depth)")
                st.info("💡 **架構設計展示**：您所要求的「五檔、即時明細、券商分點」屬於台灣證交所**盤中高頻付費數據**。")
                st.write("目前架構採用 Yahoo Finance 與 FinMind 作為免費開源資料源，僅能取得「日K級別」與「盤後籌碼」。")
                st.write("若需解鎖本頁籤功能，後續可為您串接如 **Shioaji (永豐金 API)** 或 **Fugle (富果 API)** 的即時 Websocket 服務。")
                
                # 用當日高低點模擬替代展示
                st.write("---")
                st.markdown("**當日極值區間展示 (日報價模擬)**")
                sd1, sd2, sd3 = st.columns(3)
                sd1.metric("今日最高", f"{df['High'].iloc[-1]:.2f}")
                sd2.metric("今日最低", f"{df['Low'].iloc[-1]:.2f}")
                sd3.metric("今日振幅", f"{(df['High'].iloc[-1] - df['Low'].iloc[-1])/df['Low'].iloc[-1]*100:.2f}%")