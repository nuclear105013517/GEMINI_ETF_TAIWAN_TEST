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

@st.cache_data(ttl=3600)
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
        """整合優化：從快取後的 FinMind 數據中提取最新單日籌碼概要"""
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
        
        # 防止分母為 0
        avg_loss = avg_loss.replace(0, 0.00001)
        df['RSI_14'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        low_9, high_9 = df['Low'].rolling(window=9).min(), df['High'].rolling(window=9).max()
        price_diff = (high_9 - low_9).replace(0, np.nan) 
        df['RSV'] = ((df['Close'] - low_9) / price_diff) * 100
        df['RSV'] = df['RSV'].fillna(50)
        
        df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['SMA_60'] = df['Close'].rolling(window=60).mean() # 擴充中長期線
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
            # 針對長線(5年以上)與存股邏輯優化提示
            return "強烈建議進場 (長線財務與護城河穩定。短線技術指標僅供承接點參考，時間是 ETF 最大護城河，建議紀律扣款！)"

    def generate_report_string(self):
        """重構：改為安全回傳字串"""
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
            "1年": {"fund": 0.4, "tech": 0.4, "chip": 0.2, "desc": "中線投資需要基本面獲利支撐，搭配技術多頭"},
            "1-3年": {"fund": 0.6, "tech": 0.3, "chip": 0.1, "desc": "中長線以長期獲利與估值為主，技術面抓低點"},
            "3-5年": {"fund": 0.8, "tech": 0.2, "chip": 0.0, "desc": "長線投資高度看重 ROE 與護城河，忽略籌碼波動"},
            "5年以上": {"fund": 0.9, "tech": 0.1, "chip": 0.0, "desc": "存股極長線幾乎純看財務體質與殖利率"}
        }
        self.chip_details = []
        self.chip_score = 10

    def calculate_indicators(self):
        self.df['MA20'] = self.df['Close'].rolling(20).mean()
        self.df['MA60'] = self.df['Close'].rolling(60).mean()
        low_min, high_max = self.df['Low'].rolling(9).min(), self.df['High'].rolling(9).max()
        price_diff = (high_max - low_min).replace(0, np.nan)
        self.df['RSV'] = ((self.df['Close'] - low_min) / price_diff) * 100
        self.df['RSV'] = self.df['RSV'].fillna(50)
        self.df['K'] = self.df['RSV'].ewm(com=2, adjust=False).mean()
        self.df['D'] = self.df['K'].ewm(com=2, adjust=False).mean()

    def analyze_fundamentals(self):
        score, details = 0, []
        pe = self.info.get('trailingPE', self.info.get('forwardPE'))
        pb = self.info.get('priceToBook')
        roe = self.info.get('returnOnEquity')
        yield_pct = self.info.get('dividendYield')
        eps = self.info.get('trailingEps')
        
        # 安全防禦型數值轉換
        pe_val = float(pe) if pe is not None else None
        pb_val = float(pb) if pb is not None else None
        roe_val = float(roe) * 100 if roe is not None else None
        yield_val = float(yield_pct) * 100 if yield_pct is not None else None

        # 產業估值防禦彈性提示
        if pe_val is not None:
            if 0 < pe_val < 18: score += 10; details.append(f"本益比({pe_val:.2f}) < 18，估值具吸引力 (+10分)")
            elif 18 <= pe_val <= 28: score += 5; details.append(f"本益比({pe_val:.2f}) 處於合理成長或穩健防禦區間 (+5分)")
            elif pe_val <= 0: details.append(f"本益比({pe_val:.2f}) 為負值，近期經營可能處於虧損狀態 (0分)")
            else: details.append(f"本益比({pe_val:.2f}) 偏高，長線進場需注意溢價風險 (0分)")
        else:
            score += 5; details.append("本益比數據未明，給予中性基本分 (+5分)")

        if pb_val is not None and 0 < pb_val < 2.5: score += 5; details.append(f"股價淨值比({pb_val:.2f}) 相對安全 (+5分)")
        else: details.append(f"股價淨值比偏高或暫無數據 (0分)")
            
        if roe_val is not None:
            if roe_val > 15: score += 10; details.append(f"近四季 ROE({roe_val:.2f}%) > 15%，資本回報效率極佳 (+10分)")
            elif roe_val > 8: score += 5; details.append(f"近四季 ROE({roe_val:.2f}%) 表現穩定 (+5分)")
            else: details.append(f"近四季 ROE({roe_val:.2f}%) 偏低 (0分)")
        else:
            score += 5; details.append("ROE 數據未明，給予權益中性基本分 (+5分)")
            
        if yield_val is not None and yield_val > 4: score += 5; details.append(f"預估殖利率({yield_val:.2f}%) > 4%，具備良好下檔保護 (+5分)")
        else: details.append(f"預估殖利率較低或未載明，屬成長型或海外標的 (0分)")
            
        return score, details, eps

    def analyze_technicals(self):
        score, details = 0, []
        latest, prev = self.df.iloc[-1], self.df.iloc[-2]
        
        # 解決 MA60 在數據不足時的 NaN 預防
        if pd.notna(latest['MA20']) and pd.notna(latest['MA60']):
            if latest['Close'] > latest['MA20'] > latest['MA60']:
                score += 15; details.append("股價 > 月線 > 季線，技術面呈標準多頭排列趨勢 (+15分)")
            elif latest['Close'] < latest['MA20'] and latest['MA20'] < latest['MA60']:
                details.append("技術面呈空頭走勢，均線壓力重，不建議盲目接刀 (0分)")
            else:
                score += 5; details.append("均線交織，目前處於區間盤整、重塑型態階段 (+5分)")
        else:
            score += 5; details.append("中長期均線尚在計算生成中，給予技術中立分 (+5分)")
        
        if latest['K'] < 30:
            score += 10; details.append(f"KD 指標進入低檔超賣區 (K:{latest['K']:.1f})，隨時可能反彈 (+10分)")
            if latest['K'] > latest['D'] and prev['K'] <= prev['D']: score += 5; details.append("【訊號】KD 在低檔區完成黃金交叉 (+5分)")
        elif latest['K'] > 80: details.append(f"KD 指標進入高檔超買區 (K:{latest['K']:.1f})，注意過熱回檔 (0分)")
        else: score += 5; details.append(f"KD 指標位於中性區間 (K:{latest['K']:.1f}) (+5分)")
        
        recent_low, recent_high = self.df['Low'].tail(20).min(), self.df['High'].tail(20).max()
        if latest['Close'] <= recent_low * 1.03: score += 10; details.append(f"當前股價接近近20日支撐點({recent_low:.2f})，相對抗跌 (+10分)")
        return score, details, recent_low, recent_high

    def process_finmind_chips(self, df_inst, df_shares):
        """重構：與外部快取層串接，安全處理籌碼計算"""
        score, details = 0, []
        if self.market_label == "美股/全球":
            self.chip_score = 10
            self.chip_details = ["海外市場不適用台灣集保與三大法人分析，給予中立基本分 (+10分)"]
            return

        try:
            if not df_inst.empty:
                df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
                recent_3_days_net = df_inst.groupby('date')['net_buy'].sum().tail(3).sum() / 1000
                if recent_3_days_net > 100: score += 10; details.append(f"三大法人近 3 日強勢買超共 {recent_3_days_net:,.0f} 張，籌碼點火 (+10分)")
                elif recent_3_days_net > -100: score += 5; details.append(f"三大法人近 3 日呈現中性橫盤，淨變動 {recent_3_days_net:,.0f} 張 (+5分)")
                else: details.append(f"三大法人近 3 日大舉調節賣超 {abs(recent_3_days_net):,.0f} 張，上方壓力重 (0分)")
            else:
                score += 5; details.append("近期無三大法人進出數據，給予中性基本分 (+5分)")

            if not df_shares.empty:
                df_big_holders = df_shares[df_shares['HoldingSharesLevel'] == 'more than 1,000,001'].sort_values(by='date')
                if len(df_big_holders) >= 2:
                    diff = df_big_holders['percent'].iloc[-1] - df_big_holders['percent'].iloc[-2]
                    if diff > 0.1: score += 10; details.append(f"千張大戶持股比率單週增加 {diff:+.2f}%，大戶進場 (+10分)")
                    elif diff < -0.1: details.append(f"千張大戶持股比率單週減少 {abs(diff):-.2f}%，大戶出貨 (0分)")
                    else: score += 5; details.append(f"千張大戶持股比率維持穩定 (+5分)")
                else: score += 5; details.append("集保大戶歷史數據週轉不足，給予中性基本分 (+5分)")
            else:
                score += 5; details.append("未獲取到股權分散表數據，給予中性基本分 (+5分)")
        except:
            score = 10; details = ["籌碼資料解析異常，啟動防護機制給予中立基本分 (+10分)"]
            
        self.chip_score = score
        self.chip_details = details

    def _get_advice_level(self, total_score):
        if total_score >= 75: return "強烈建議進場"
        elif total_score >= 60: return "建議進場"
        elif total_score >= 45: return "平平 (觀望為主)"
        elif total_score >= 30: return "不建議進場"
        else: return "強烈不建議進場"

    def generate_report_string(self):
        """重構：移除非安全性重導向，直接構建 Markdown/純文字"""
        fund_score, fund_details, eps = self.analyze_fundamentals()
        tech_score, tech_details, support, resistance = self.analyze_technicals()
        
        f_pct = (fund_score / 30) * 100
        t_pct = (tech_score / 40) * 100
        c_pct = (self.chip_score / 20) * 100
        
        user_w = self.horizons[self.matched_horizon]
        weighted_f = f_pct * user_w['fund']
        weighted_t = t_pct * user_w['tech']
        weighted_c = c_pct * user_w['chip']
        
        max_f, max_t, max_c = user_w['fund'] * 100, user_w['tech'] * 100, user_w['chip'] * 100
        user_total = weighted_f + weighted_t + weighted_c
        eps_display = f"{eps:.2f}" if eps is not None else "暫無資料"

        out = io.StringIO()
        out.write("="*75 + "\n")
        out.write(f" 📊 【 {self.stock_name} ({self.ticker}) 】 個股量化策略多空評估報告 \n")
        out.write("="*75 + "\n")
        out.write(f" 現前股價: {self.df['Close'].iloc[-1]:,.2f} 元  |  近四季累積 EPS: {eps_display} 元\n")
        out.write(f" 💡 尋找低點效益 - 建議黃金進場價格區間: {support:,.2f} ~ {support * 1.03:,.2f} 元\n")
        out.write("-" * 75 + "\n")
        out.write("【 依照各投資年限之進場建議與戰略分析 】\n")
        out.write(f" (1) 針對您的投資年限【{self.user_horizon_text}】 (模型對應: {self.matched_horizon})\n")
        out.write(f"    ⭐ 專屬綜合評分: {user_total:.1f}/100 | 進場建議: 👉 【 {self._get_advice_level(user_total)} 】\n")
        out.write(f"    💡 策略說明: {user_w['desc']} (權重: 基本面{max_f:.0f}% / 技術面{max_t:.0f}% / 籌碼面{max_c:.0f}%)\n")
        out.write("-" * 75 + "\n")
        
        seq = 2
        for horizon, w in self.horizons.items():
            h_total = (f_pct * w['fund']) + (t_pct * w['tech']) + (c_pct * w['chip'])
            out.write(f" ({seq}) {horizon} 投資建議: 【 {self._get_advice_level(h_total)} 】 (評分: {h_total:.1f}/100)\n")
            seq += 1

        out.write("="*75 + "\n")
        out.write(f"\n🔍 【第一維度：基本面價值評估】 (得分: {weighted_f:.1f} / {max_f:.0f})\n")
        for d in fund_details: out.write(f"  • {d}\n")
        out.write(f"\n📈 【第二維度：技術面擇時評估】 (得分: {weighted_t:.1f} / {max_t:.0f})\n")
        for d in tech_details: out.write(f"  • {d}\n")
        out.write(f"  • 參考波段壓力位(近20日高點): {resistance:,.2f} 元\n")
        out.write(f"\n📊 【第三維度：籌碼面法人大戶追蹤】 (得分: {weighted_c:.1f} / {max_c:.0f})\n")
        if max_c == 0: out.write("  • 說明：由於長線存股，系統已將短期籌碼權重歸零，改為專注基本財務。\n")
        for d in self.chip_details: out.write(f"  • {d}\n")
        out.write("="*75 + "\n")
        return out.getvalue()


class MasterRoutingSystem:
    def __init__(self):
        self._stock_info_cache = load_taiwan_stock_info()
        
    def auto_detect_type(self, raw_ticker):
        is_tw = any(char.isdigit() for char in raw_ticker)
        ticker_yf, is_etf, stock_name, market_label = raw_ticker, False, "未知名稱", "美股/全球"
        
        if is_tw:
            try:
                if not self._stock_info_cache.empty:
                    target = self._stock_info_cache[self._stock_info_cache['stock_id'] == raw_ticker]
                    if not target.empty:
                        stock_name = target.iloc[0]['stock_name']
                        industry = str(target.iloc[0].get('industry_category', '')).upper()
                        market_type = str(target.iloc[0].get('type', '')).lower()
                        market_label = "上櫃" if 'tpex' in market_type or 'otc' in market_type else "上市"
                        ticker_yf = f"{raw_ticker}.TWO" if market_label == "上櫃" else f"{raw_ticker}.TW"
                        if 'ETF' in industry or raw_ticker.startswith('00'): is_etf = True
                    else:
                        ticker_yf = f"{raw_ticker}.TW"
                        if raw_ticker.startswith('00'): is_etf = True
            except:
                ticker_yf = f"{raw_ticker}.TW"
                if raw_ticker.startswith('00'): is_etf = True
                
        try:
            # 引入快取下載
            _, info = fetch_yf_data(ticker_yf)
            if info.get('quoteType') == 'ETF': is_etf = True
            elif info.get('quoteType') == 'EQUITY': is_etf = False
            if stock_name == "未知名稱": 
                stock_name = info.get('shortName', info.get('longName', raw_ticker))
        except: 
            pass
            
        return ticker_yf, is_etf, stock_name, market_label


# ==========================================
# 4. 網頁 UI 綁定與呈現層 (原生適配優化)
# ==========================================
st.markdown("<h1 style='text-align: center; margin-bottom: 5px; font-weight: 700; letter-spacing: -1px;'>台美股及ETF量化進場決策系統</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-bottom: 30px;'>結合基本面、技術面與籌碼面的法人級量化模型</p>", unsafe_allow_html=True)

# 使用 Streamlit 內建 border 卡片，完美適配深淺色主題
with st.container(border=True):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        raw_ticker = st.text_input("搜尋股票或 ETF 代號 (如: 0050, 2330, AAPL)", value="2412").strip().upper().replace('.TW', '').replace('.TWO', '')
    with col2:
        horizon_input = st.text_input("預計投資年限 (如: 10年, 半年, 存股, 當沖)", value="1年").strip()
    with col3:
        st.write("") 
        st.write("") 
        run_btn = st.button("🚀 開始分析", use_container_width=True)

if run_btn and raw_ticker:
    with st.spinner("正在連接市場資料庫並進行多維度量化運算..."):
        try:
            system = MasterRoutingSystem()
            matched_horizon = parse_investment_horizon(horizon_input)
            horizon_years = get_horizon_years(matched_horizon)
            ticker_yf, is_etf, stock_name, market_label = system.auto_detect_type(raw_ticker)
            
            # 使用統一快取層下載數據（2年數據視窗）
            df_chart, yf_info = fetch_yf_data(ticker_yf, period="2y")
            
            if df_chart.empty:
                st.error(f"無法從市場獲取標的代號 {ticker_yf} 的有效交易數據，請檢查輸入是否有誤。")
            else:
                # 預先下載籌碼資料備用
                df_inst = fetch_finmind_institutional(raw_ticker) if market_label != "美股/全球" else pd.DataFrame()
                df_shares = fetch_finmind_big_holders(raw_ticker) if market_label != "美股/全球" else pd.DataFrame()
                
                # 初始化計算引擎
                if is_etf:
                    analyzer = ETFAnalyzer(raw_ticker, ticker_yf, stock_name, horizon_years, df_chart)
                    analyzer.process_institutional_summary(df_inst)
                    analyzer.calculate_indicators()
                    analyzer.analyze_score()
                    report_output = analyzer.generate_report_string()
                    core_engine = analyzer 
                else:
                    evaluator = StockEvaluator(raw_ticker, ticker_yf, stock_name, market_label, matched_horizon, df_chart, yf_info)
                    evaluator.calculate_indicators()
                    evaluator.process_finmind_chips(df_inst, df_shares)
                    report_output = evaluator.generate_report_string()
                    core_engine = evaluator

                st.success(f"分析完成！已成功對齊 {stock_name} ({raw_ticker}) 的跨景氣週期數據。")
                
                # 建立五個切換頁籤 (全面活化並填補原版空白)
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📑 量化決策報告", 
                    "📈 K線與價量指標", 
                    "💰 基本面與股利", 
                    "🏦 法人籌碼趨勢", 
                    "⚡ 歷史波動與價量明細"
                ])

                # 頁籤一：量化報告
                with tab1:
                    st.markdown("### 專家級多維度量化決策報告")
                    st.code(report_output, language="text")

                # 頁籤二：K線指標
                with tab2:
                    st.markdown("### 互動式動態 K 線與均線指標")
                    if 'Close' in df_chart.columns and len(df_chart) > 0:
                        # 取最近 120 天作為圖表最適可視範圍
                        df_plot = df_chart.tail(120)
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                        
                        # K線
                        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="K線"), row=1, col=1)
                        
                        # 加上 MA20 與 MA60 均線
                        if not is_etf:
                            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], line=dict(color='#FFA500', width=1.5), name="20日均線"), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA60'], line=dict(color='#4169E1', width=1.5), name="60日均線"), row=1, col=1)
                        else:
                            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_20'], line=dict(color='#FFA500', width=1.5), name="20日均線"), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_60'], line=dict(color='#4169E1', width=1.5), name="60日均線"), row=1, col=1)

                        # 成交量
                        colors = ['#00B500' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i] else '#FF3333' for i in range(len(df_plot))]
                        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
                        
                        fig.update_layout(height=550, margin=dict(l=10, r=10, t=10, b=10), showlegend=True, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

                # 頁籤三：基本面
                with tab3:
                    st.markdown("### 基本財務指標與價值評估")
                    c1, c2, c3, c4 = st.columns(4)
                    
                    pe = yf_info.get('trailingPE', yf_info.get('forwardPE', 'N/A'))
                    pb = yf_info.get('priceToBook', 'N/A')
                    yield_pct = yf_info.get('dividendYield')
                    yield_display = f"{yield_pct * 100:.2f}%" if yield_pct else "N/A"
                    nav = yf_info.get('navPrice', 'N/A')
                    
                    if isinstance(pe, float): pe = f"{pe:.2f}"
                    if isinstance(pb, float): pb = f"{pb:.2f}"
                    
                    c1.metric("本益比 (PE)", pe)
                    c2.metric("股價淨值比 (PB)", pb)
                    c3.metric("預估殖利率", yield_display)
                    c4.metric("最新資產淨值 (NAV)", f"{nav}" if is_etf else "個股不適用")
                    
                    st.write("---")
                    if len(df_chart) > 120:
                        perf_1m = (df_chart['Close'].iloc[-1] / df_chart['Close'].iloc[-20] - 1) * 100
                        perf_6m = (df_chart['Close'].iloc[-1] / df_chart['Close'].iloc[-120] - 1) * 100
                        perf_2y = (df_chart['Close'].iloc[-1] / df_chart['Close'].iloc[0] - 1) * 100
                        
                        st.markdown("#### 🚀 週期歷史績效追蹤")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("近 1 個月表現", f"{perf_1m:+.2f}%")
                        p2.metric("近 2 季表現", f"{perf_6m:+.2f}%")
                        p3.metric("近 2 年長期表現", f"{perf_2y:+.2f}%")

                # 頁籤四：籌碼面 (活化：利用 FinMind 資料繪製法人趨勢柱狀圖，不再顯示限制警告)
                with tab4:
                    st.markdown("### 近期法人籌碼動態歷史趨勢")
                    if market_label == "美股/全球":
                        st.info("海外全球標的未提供集保與三大法人分點資料，本頁面聚焦於台股市場。")
                    else:
                        if not df_inst.empty:
                            df_inst['net_shares'] = df_inst['buy'] - df_inst['sell']
                            df_pivot = df_inst.pivot_table(index='date', columns='name', values='net_shares', aggfunc='sum').tail(20)
                            
                            # 重新映射簡稱
                            name_map = {'Foreign_Investor': '外資', 'Investment_Trust': '投信', 'Dealer_Self': '自營商'}
                            df_pivot = df_pivot.rename(columns=name_map)
                            
                            fig_chip = go.Figure()
                            for col in df_pivot.columns:
                                if col in ['外資', '投信', '自營商']:
                                    fig_chip.add_trace(go.Bar(x=df_pivot.index, y=df_pivot[col] / 1000, name=col))
                            
                            fig_chip.update_layout(barmode='group', title="近 20 日三大法人買賣超動態 (張)", xaxis_title="日期", yaxis_title="淨買賣超 (張)", height=400, margin=dict(t=30, b=10))
                            st.plotly_chart(fig_chip, use_container_width=True)
                        else:
                            st.warning("暫時無法取得該標的之歷史法人詳細數據，請稍後重試。")

                # 頁籤五：波動與明細 (活化：計算真實歷史數據波動與明細統計)
                with tab5:
                    st.markdown("### 歷史波動度與近期交易價量明細")
                    if len(df_chart) > 0:
                        # 計算年化波動率
                        log_ret = np.log(df_chart['Close'] / df_chart['Close'].shift(1))
                        volatility_2y = log_ret.std() * np.sqrt(252) * 100
                        
                        today = df_chart.iloc[-1]
                        v1, v2, v3, v4 = st.columns(4)
                        v1.metric("2年歷史年化波動度", f"{volatility_2y:.2f}%")
                        v2.metric("近2年最高收盤價", f"{df_chart['Close'].max():,.2f}")
                        v3.metric("近2年最低收盤價", f"{df_chart['Close'].min():,.2f}")
                        v4.metric("歷史交易日總計", f"{len(df_chart)} 天")
                        
                        st.write("---")
                        st.markdown("#### 近 10 日價量數據結算明細")
                        df_detail = df_chart.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False)
                        df_detail['Volume'] = df_detail['Volume'].apply(lambda x: f"{int(x):,}")
                        st.dataframe(df_detail, use_container_width=True)

        except Exception as general_e:
            st.error(f"系統執行過程中發生未預期錯誤: {general_e}")
            st.info("請確認網路連線穩定，或稍微縮短搜尋頻率以防止資料庫伺服器拒絕連線。")