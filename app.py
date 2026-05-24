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
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ==========================================
# 1. 介面設計：與原版 Apple 現代簡約風格對齊 (優化深淺色適配)
# ==========================================
st.set_page_config(page_title="台美股及ETF量化進場決策系統", layout="wide")

apple_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "SF Pro TC", "PingFang TC", "Helvetica Neue", sans-serif;
    }
    
    /* 隱藏預設的 Streamlit 選單與 Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 按鈕樣式 (Apple 藥丸型) */
    .stButton>button {
        border-radius: 980px;
        background-color: #0071E3;
        color: white;
        font-weight: 600;
        padding: 10px 28px;
        border: none;
        transition: background-color 0.3s ease, transform 0.1s ease;
    }
    .stButton>button:hover {
        background-color: #0077ED;
        color: white;
        transform: scale(1.02);
    }
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

# ==========================================
# 2. 資料快取層 (解決重複請求與限流問題)
# ==========================================
@st.cache_data(ttl=3600)
def load_taiwan_stock_info():
    """快取台灣股票基本資訊"""
    try:
        fm = DataLoader()
        return fm.taiwan_stock_info()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_yf_data(ticker_str, period="2y"):
    """快取 yfinance 的歷史價格與 Info 數據 (觀測窗擴大至 2 年以支援 MA60 與長線分析)"""
    ticker = yf.Ticker(ticker_str)
    df = ticker.history(period=period)
    try:
        info = ticker.info
    except:
        info = {}
    return df, info

@st.cache_data(ttl=1800)
def fetch_finmind_institutional(stock_id, days_back=40):
    """快取 FinMind 三大法人資料"""
    try:
        fm = DataLoader()
        start_date = (datetime.date.today() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        return fm.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_finmind_big_holders(stock_id, days_back=40):
    """快取 FinMind 大戶持股比例"""
    try:
        fm = DataLoader()
        start_date = (datetime.date.today() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        return fm.taiwan_stock_holding_shares_per(stock_id=stock_id, start_date=start_date)
    except:
        return pd.DataFrame()


# ==========================================
# 3. 核心邏輯層 (移除重導向，改為執行緒安全的字串回傳)
# ==========================================
def parse_investment_horizon(text):
    text = text.replace(" ", "")
    if any(k in text for k in ["存股", "長期", "不賣", "退休"]): return "5年以上"
    if any(k in text for k in ["短線", "當沖", "隔日沖"]): return "1-3個月"
    if "半年" in text: return "3-6個月"

    replace_map = {"一": "1", "兩": "2", "二": "2", "三": "3", "四": "4", "五": "5"}
    for k, v in replace_map.items(): text = text.replace(k, v)
    if text == "十年": text = "10年"

    if "年" in text:
        match = re.search(r'(\d+(?:\.\d+)?)\s*年', text)
        if match:
            years = float(match.group(1))
            if years < 1: return "3-6個月"
            elif years <= 1: return "1年"
            elif years <= 3: return "1-3年"
            elif years <= 5: return "3-5年"
            else: return "5年以上"
        return "1-3年"

    if "月" in text:
        match = re.search(r'(\d+(?:\.\d+)?)\s*個?月', text)
        if match:
            months = float(match.group(1))
            if months <= 3: return "1-3個月"
            elif months <= 6: return "3-6個月"
            elif months <= 12: return "1年"
            elif months <= 36: return "1-3年"
            else: return "3-5年"
        return "3-6個月"

    if any(k in text for k in ["天", "日", "周", "週"]): return "1-3個月"
    return "1年" 

def get_horizon_years(horizon_str):
    mapping = {"1-3個月": 0.25, "3-6個月": 0.5, "1年": 1.0, "1-3年": 2.0, "3-5年": 4.0, "5年以上": 5.0}
    return mapping.get(horizon_str, 1.0)


class ETFAnalyzer:
    def __init__(self, raw_ticker, ticker_yf, etf_name, user_years, df_data):
        self.raw_ticker, self.ticker_yf = raw_ticker, ticker_yf
        self.is_tw_stock = ticker_yf.endswith('.TW') or ticker_yf.endswith('.TWO')
        self.etf_name, self.user_years = etf_name, user_years
        self.data = df_data.copy()
        self.score = 0
        self.evaluation_details = []
        self.institutional_data = None

    def process_institutional_summary(self, df_inst):
        """從快取後的 FinMind 數據中提取最新單日籌碼概要"""
        if df_inst.empty or not self.is_tw_stock:
            return
        try:
            latest_date = df_inst['date'].max()
            df_latest = df_inst[df_inst['date'] == latest_date]
            
            foreign = df_latest[df_latest['name'] == 'Foreign_Investor']['buy'] - df_latest[df_latest['name'] == 'Foreign_Investor']['sell']
            sitc = df_latest[df_latest['name'] == 'Investment_Trust']['buy'] - df_latest[df_latest['name'] == 'Investment_Trust']['sell']
            dealer = df_latest[df_latest['name'] == 'Dealer_Self']['buy'] - df_latest[df_latest['name'] == 'Dealer_Self']['sell']
            
            self.institutional_data = {
                'foreign': int(foreign.sum()) if not foreign.empty else 0,
                'sitc': int(sitc.sum()) if not sitc.empty else 0,
                'dealer': int(dealer.sum()) if not dealer.empty else 0,
                'date': latest_date
            }
        except:
            self.institutional_data = None

    def calculate_indicators(self):
        df = self.data
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['BIAS_20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100
        delta = df['Close'].diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        
        avg_loss = avg_loss.replace(0, 0.00001)
        df['RSI_14'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        low_9, high_9 = df['Low'].rolling(window=9).min(), df['High'].rolling(window=9).max()
        price_diff = (high_9 - low_9).replace(0, np.nan) 
        df['RSV'] = ((df['Close'] - low_9) / price_diff) * 100
        df['RSV'] = df['RSV'].fillna(50)
        
        df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['SMA_60'] = df['Close'].rolling(window=60).mean()
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    def analyze_score(self):
        if len(self.data) < 2: return
        latest, prev = self.data.iloc[-1], self.data.iloc[-2]
        bias = latest['BIAS_20']
        if pd.notna(bias):
            if bias < -5: self.score += 2; self.evaluation_details.append(f"[+2分] 20日乖離率為 {bias:.2f}%，處低檔超跌。")
            elif bias > 5: self.score -= 2; self.evaluation_details.append(f"[-2分] 20日乖離率為 {bias:.2f}%，偏離過高。")
            else: self.evaluation_details.append(f"[ 0分] 20日乖離率正常 ({bias:.2f}%)。")
            
        rsi, k, d = latest['RSI_14'], latest['K'], latest['D']
        if pd.notna(rsi):
            if rsi < 30: self.score += 2; self.evaluation_details.append(f"[+2分] RSI進入超賣區 ({rsi:.1f})。")
            elif rsi > 70: self.score -= 2; self.evaluation_details.append(f"[-2分] RSI進入超買區 ({rsi:.1f})。")
        if pd.notna(k) and pd.notna(d):
            if k < 30 and k > d and prev['K'] <= prev['D']: self.score += 3; self.evaluation_details.append("[+3分] KD低檔黃金交叉。")
            elif k > 80 and k < d and prev['K'] >= prev['D']: self.score -= 3; self.evaluation_details.append("[-3分] KD高檔死亡交叉。")
            
        macd, sig = latest['MACD'], latest['MACD_Signal']
        if pd.notna(macd) and pd.notna(sig):
            if macd > sig and prev['MACD'] <= prev['MACD_Signal']: self.score += 3; self.evaluation_details.append("[+3分] MACD低檔黃金交叉。")
            elif macd < sig and prev['MACD'] >= prev['MACD_Signal']: self.score -= 3; self.evaluation_details.append("[-3分] MACD高檔死亡交叉。")
            
        if self.institutional_data:
            f_net, s_net, d_net = self.institutional_data['foreign']/1000, self.institutional_data['sitc']/1000, self.institutional_data['dealer']/1000
            self.evaluation_details.append(f"[籌碼現況] 近期買賣超(千張): 外資 {f_net:+.1f}、投信 {s_net:+.1f}、自營商 {d_net:+.1f}。")
            if f_net > 0 and s_net > 0: self.score += 2; self.evaluation_details.append("[+2分] 外資與投信同步買超。")
            elif s_net > 500: self.score += 1; self.evaluation_details.append("[+1分] 投信積極買超建倉。")

    def get_time_based_advice(self, term_type):
        s = self.score
        if term_type == "1-3個月":
            if s >= 6: return "強烈建議進場 (技術籌碼轉強，具備短線爆發力)"
            elif s >= 2: return "建議進場 (短線趨勢偏多，沿均線操作)"
            elif s >= -1: return "中性觀望 (無明顯方向，等待表態)"
            else: return "強烈不建議進場 (短線有修正風險)"
        elif term_type == "3-6個月":
            if s >= 4: return "強烈建議進場 (波段買點浮現)"
            elif s >= 1: return "建議分批進場 (波段趨勢尚可)"
            else: return "建議觀望 (波段仍有下探風險)"
        elif term_type == "1年":
            if s >= 2: return "建議單筆+定期定額進場 (位階偏低)"
            elif s > -4: return "建議定期定額進場 (透過紀律扣款平滑成本)"
            else: return "建議小額定期定額 (短線偏高，保留主要資金)"
        else:
            return "強烈建議進場 (長線財務與護城河穩定。短線技術指標僅供承接點參考，時間是 ETF 最大護城河，建議紀律扣款！)"

    def generate_report_string(self):
        latest, recent_low = self.data.iloc[-1], self.data['Low'].tail(20).min()
        sma20, current_price = latest['SMA_20'], latest['Close']
        g_low = recent_low * 0.98 if latest['RSI_14'] < 30 else min(recent_low, current_price * 0.95)
        g_high = recent_low * 1.02 if latest['RSI_14'] < 30 else (sma20 if pd.notna(sma20) else current_price)
        matched_tier = "5年以上" if self.user_years > 5.0 else ("3-5年" if self.user_years > 3.0 else ("1-3年" if self.user_years > 1.0 else ("1年" if self.user_years > 0.5 else ("3-6個月" if self.user_years > 0.25 else "1-3個月"))))

        out = io.StringIO()
        out.write("="*70 + "\n")
        out.write(f"📊 【 {self.etf_name} ({self.ticker_yf}) 】 ETF 多維度進場決策報告\n")
        out.write("="*70 + "\n")
        out.write(f"💰 目前收盤價: {current_price:.2f} 元\n")
        out.write(f"✨ 建議短線黃金承接區段: {g_low:.2f} ~ {g_high:.2f} 元\n")
        out.write(f"🎯 技術與籌碼綜合總分: {self.score} 分\n")
        out.write("-" * 70 + "\n")
        out.write("🔍 【技術面與籌碼評估細節】:\n")
        for detail in self.evaluation_details: 
            out.write(f"  {detail}\n")
        out.write("-" * 70 + "\n")
        out.write("⏳ 【依投資時間年限之進場建議】:\n")
        out.write(f"  ⭐ 專屬客製化 ({self.user_years} 年) 建議: 👉 【 {self.get_time_based_advice(matched_tier)} 】\n\n")
        for term in ["1-3個月", "3-6個月", "1年", "1-3年", "3-5年", "5年以上"]:
            out.write(f"  ({term}) 進場建議: {self.get_time_based_advice(term)}\n")
        out.write("="*70 + "\n")
        return out.getvalue()


class StockEvaluator:
    def __init__(self, raw_ticker, ticker_yf, stock_name, market_label, matched_horizon_str, df_data, yf_info):
        self.raw_ticker, self.ticker = raw_ticker, ticker_yf
        self.stock_name, self.market_label = stock_name, market_label
        self.matched_horizon, self.user_horizon_text = matched_horizon_str, matched_horizon_str
        self.df = df_data.copy()
        self.info = yf_info if isinstance(yf_info, dict) else {}
        self.horizons = {
            "1-3個月": {"fund": 0.1, "tech": 0.6, "chip": 0.3, "desc": "極短線極度依賴技術型態與法人動能"},
            "3-6個月": {"fund": 0.2, "tech": 0.5, "chip": 0.3, "desc": "短中線重視技術指標與大戶籌碼流向"},
            "1年": {"fund": 0.4, "tech":