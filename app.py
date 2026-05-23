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
from contextlib import redirect_stdout

warnings.filterwarnings('ignore')

# ==========================================
# 介面設計：莫蘭迪色系 CSS (Morandi Palette)
# ==========================================
st.set_page_config(page_title="台美股/ETF 量化決策系統", page_icon="🏦", layout="centered")

morandi_css = """
<style>
    /* 全局背景與字體顏色 (莫蘭迪灰白) */
    .stApp {
        background-color: #F2F1EC;
        color: #4A4A4A;
    }
    /* 標題顏色 (莫蘭迪深灰綠) */
    h1, h2, h3 {
        color: #6B7B75 !important;
    }
    /* 按鈕樣式 (莫蘭迪藍灰) */
    .stButton>button {
        background-color: #79898C;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5D6C6F;
        border-color: #5D6C6F;
    }
    /* 輸入框背景 (莫蘭迪淺灰) */
    .stTextInput>div>div>input {
        background-color: #E8E8E3;
        color: #333333;
        border: 1px solid #C4C4C4;
    }
    /* 終端機輸出區塊設計 (柔和沙色) */
    pre {
        background-color: #E2DFD8 !important;
        color: #3A403E !important;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #8B9D95;
    }
</style>
"""
st.markdown(morandi_css, unsafe_allow_html=True)

# ==========================================
# 核心邏輯
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
    def __init__(self, raw_ticker, ticker_yf, etf_name, user_years):
        self.raw_ticker, self.ticker_yf = raw_ticker, ticker_yf
        self.is_tw_stock = ticker_yf.endswith('.TW') or ticker_yf.endswith('.TWO')
        self.etf_name, self.user_years = etf_name, user_years
        self.data, self.news, self.score, self.evaluation_details, self.institutional_data = None, [], 0, [], None

    def fetch_data(self):
        print(f"[1/2] 正在抓取 {self.ticker_yf} 的市場歷史資料...")
        etf = yf.Ticker(self.ticker_yf)
        self.data = etf.history(period="6mo")
        if self.data is None or self.data.empty:
            if self.ticker_yf.endswith('.TW'):
                self.ticker_yf = self.ticker_yf.replace('.TW', '.TWO')
                etf = yf.Ticker(self.ticker_yf)
                self.data = etf.history(period="6mo")
        if self.data is None or len(self.data) < 20: raise ValueError(f"無法獲取 {self.ticker_yf} 足夠資料。")

    def fetch_institutional_data(self):
        if not self.is_tw_stock: return
        print(f"[2/2] 正在同步 {self.ticker_yf} 的三大法人籌碼資料...")
        pure_ticker, headers = self.raw_ticker, {'User-Agent': 'Mozilla/5.0'}
        def to_shares(val):
            try: return int(str(val).replace(',', '').replace(' ', ''))
            except: return 0
        for days_back in range(5):
            target_date = self.data.index[-1]
            if target_date.tz is not None: target_date = target_date.tz_localize(None)
            target_date -= timedelta(days=days_back)
            try:
                if self.ticker_yf.endswith('.TW'):
                    date_str = target_date.strftime("%Y%m%d")
                    url = f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={date_str}&selectType=ALL"
                    res = requests.get(url, headers=headers, timeout=5).json()
                    if res.get('stat') == 'OK' and 'data' in res:
                        fields = res['fields']
                        idx_id = fields.index("證券代號") if "證券代號" in fields else 0
                        idx_f = next((i for i, f in enumerate(fields) if "外" in f and "買賣超" in f), 4)
                        idx_s = next((i for i, f in enumerate(fields) if "投信" in f and "買賣超" in f), 10)
                        idx_d = next((i for i, f in enumerate(fields) if "自營商" in f and "買賣超" in f), 11)
                        for row in res['data']:
                            if row[idx_id].strip().replace('"', '') == pure_ticker:
                                self.institutional_data = {'foreign': to_shares(row[idx_f]), 'sitc': to_shares(row[idx_s]), 'dealer': to_shares(row[idx_d]), 'date': date_str}
                                return
                elif self.ticker_yf.endswith('.TWO'):
                    roc_date = f"{target_date.year - 1911}/{target_date.strftime('%m/%d')}"
                    url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json&se=EW&t=D&d={roc_date}"
                    res = requests.get(url, headers=headers, timeout=5).json()
                    if 'aaData' in res and len(res['aaData']) > 0:
                        for row in res['aaData']:
                            if row[0] == pure_ticker:
                                self.institutional_data = {'foreign': to_shares(row[4]) if len(row)>4 else 0, 'sitc': to_shares(row[10]) if len(row)>10 else 0, 'dealer': to_shares(row[13]) if len(row)>13 else 0, 'date': roc_date}
                                return
            except: continue

    def calculate_indicators(self):
        df = self.data
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['BIAS_20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100
        delta = df['Close'].diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        df['RSI_14'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        low_9, high_9 = df['Low'].rolling(window=9).min(), df['High'].rolling(window=9).max()
        price_diff = (high_9 - low_9).replace(0, np.nan) 
        df['RSV'] = ((df['Close'] - low_9) / price_diff) * 100
        df['RSV'] = df['RSV'].fillna(50)
        
        # [修復] 將 K 與 D 拆為兩行計算，避免 KeyError
        df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        
        df['STD_20'] = df['Close'].rolling(window=20).std()
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
        else: return "強烈建議進場 (長線投資隨時皆是買點，時間是 ETF 的最大護城河！)"

    def display_report(self):
        latest, recent_low = self.data.iloc[-1], self.data['Low'].tail(20).min()
        sma20, current_price = latest['SMA_20'], latest['Close']
        g_low = recent_low * 0.98 if latest['RSI_14'] < 30 else min(recent_low, current_price * 0.95)
        g_high = recent_low * 1.02 if latest['RSI_14'] < 30 else (sma20 if pd.notna(sma20) else current_price)
        matched_tier = "5年以上" if self.user_years > 5.0 else ("3-5年" if self.user_years > 3.0 else ("1-3年" if self.user_years > 1.0 else ("1年" if self.user_years > 0.5 else ("3-6個月" if self.user_years > 0.25 else "1-3個月"))))

        print("\n" + "="*70)
        print(f"📊 【 {self.etf_name} ({self.ticker_yf}) 】 ETF 多維度進場決策報告")
        print("="*70)
        print(f"💰 目前收盤價: {current_price:.2f} 元")
        print(f"✨ 建議短線黃金承接區段: {g_low:.2f} ~ {g_high:.2f} 元")
        print(f"🎯 技術與籌碼綜合總分: {self.score} 分")
        print("-" * 70)
        print("🔍 【技術面與籌碼評估細節】:")
        for detail in self.evaluation_details: print("  " + detail)
        print("-" * 70)
        print("⏳ 【依投資時間年限之進場建議】:")
        print(f"  ⭐ 專屬客製化 ({self.user_years} 年) 建議: 👉 【 {self.get_time_based_advice(matched_tier)} 】\n")
        for term in ["1-3個月", "3-6個月", "1年", "1-3年", "3-5年", "5年以上"]:
            print(f"  ({term}) 進場建議: {self.get_time_based_advice(term)}")
        print("="*70)

class StockEvaluator:
    def __init__(self, raw_ticker, ticker_yf, stock_name, market_label, matched_horizon_str, fm):
        self.raw_ticker, self.ticker = raw_ticker, ticker_yf
        self.stock_name, self.market_label = stock_name, market_label
        self.matched_horizon, self.user_horizon_text = matched_horizon_str, matched_horizon_str
        self.fm, self.stock, self.df, self.info = fm, yf.Ticker(self.ticker), pd.DataFrame(), {}
        self.horizons = {
            "1-3個月": {"fund": 0.1, "tech": 0.6, "chip": 0.3, "desc": "極短線極度依賴技術型態與法人動能"},
            "3-6個月": {"fund": 0.2, "tech": 0.5, "chip": 0.3, "desc": "短中線重視技術指標與大戶籌碼流向"},
            "1年": {"fund": 0.4, "tech": 0.4, "chip": 0.2, "desc": "中線投資需要基本面獲利支撐，搭配技術多頭"},
            "1-3年": {"fund": 0.6, "tech": 0.3, "chip": 0.1, "desc": "中長線以長期獲利與估值為主，技術面抓低點"},
            "3-5年": {"fund": 0.8, "tech": 0.2, "chip": 0.0, "desc": "長線投資高度看重 ROE 與護城河，忽略籌碼波動"},
            "5年以上": {"fund": 0.9, "tech": 0.1, "chip": 0.0, "desc": "存股極長線幾乎純看財務體質與殖利率"}
        }

    def fetch_data(self):
        print(f"[1/2] 正在透過 yfinance 獲取 {self.stock_name} 的市場技術與基本面數據...")
        self.df = self.stock.history(period="6mo")
        try: self.info = self.stock.info or {}
        except: self.info = {}
        if self.df.empty: raise ValueError(f"無法獲取股價數據。")

    def calculate_indicators(self):
        self.df['MA20'], self.df['MA60'] = self.df['Close'].rolling(20).mean(), self.df['Close'].rolling(60).mean()
        low_min, high_max = self.df['Low'].rolling(9).min(), self.df['High'].rolling(9).max()
        price_diff = (high_max - low_min).replace(0, np.nan)
        self.df['RSV'] = ((self.df['Close'] - low_min) / price_diff) * 100
        self.df['RSV'] = self.df['RSV'].fillna(50)
        
        # [修復] 將 K 與 D 拆為兩行計算，避免 KeyError
        self.df['K'] = self.df['RSV'].ewm(com=2, adjust=False).mean()
        self.df['D'] = self.df['K'].ewm(com=2, adjust=False).mean()

    def analyze_fundamentals(self):
        score, details = 0, []
        pe = self.info.get('trailingPE', self.info.get('forwardPE'))
        pb, roe, yield_pct, eps = self.info.get('priceToBook'), self.info.get('returnOnEquity'), self.info.get('dividendYield'), self.info.get('trailingEps')
        pe_val = pe if pe is not None else 999
        pb_val = pb if pb is not None else 999
        roe_val = roe * 100 if roe is not None else 0
        yield_val = yield_pct * 100 if yield_pct is not None else 0

        if 0 < pe_val < 18: score += 10; details.append(f"本益比({pe_val:.2f}) < 18，估值偏低具吸引力 (+10分)")
        elif 18 <= pe_val <= 28: score += 5; details.append(f"本益比({pe_val:.2f}) 處於合理成長區間 (+5分)")
        elif pe_val <= 0: details.append(f"本益比({pe_val:.2f}) 為負或無效，近期可能虧損 (0分)")
        else: details.append(f"本益比({pe_val:.2f}) 偏高，需注意溢價風險 (0分)")
        if 0 < pb_val < 2.5: score += 5; details.append(f"股價淨值比({pb_val:.2f}) 相對安全 (+5分)")
        else: details.append(f"股價淨值比({pb_val:.2f}) 偏高或無效 (0分)")
        if roe_val > 15: score += 10; details.append(f"近四季 ROE({roe_val:.2f}%) > 15%，資本回報效率極佳 (+10分)")
        elif roe_val > 8: score += 5; details.append(f"近四季 ROE({roe_val:.2f}%) 表現穩定 (+5分)")
        else: details.append(f"近四季 ROE({roe_val:.2f}%) 偏低 (0分)")
        if yield_val > 4: score += 5; details.append(f"預估殖利率({yield_val:.2f}%) > 4%，具備下檔保護 (+5分)")
        else: details.append(f"預估殖利率({yield_val:.2f}%) 較低，屬成長型標的 (0分)")
        return score, details, eps

    def analyze_technicals(self):
        score, details = 0, []
        latest, prev = self.df.iloc[-1], self.df.iloc[-2]
        if latest['Close'] > latest['MA20'] > latest['MA60']:
            score += 15; details.append("股價 > 月線 > 季線，技術面呈標準多頭排列趨勢 (+15分)")
        elif latest['Close'] < latest['MA20'] and latest['MA20'] < latest['MA60']:
            details.append("技術面呈空頭走勢，不建議盲目接刀 (0分)")
        else: score += 5; details.append("均線交織，目前處於區間盤整階段 (+5分)")
        
        if latest['K'] < 30:
            score += 10; details.append(f"KD 指標進入低檔超賣區 (K:{latest['K']:.1f})，隨時可能反彈 (+10分)")
            if latest['K'] > latest['D'] and prev['K'] <= prev['D']: score += 5; details.append("【訊號】KD 在低檔區完成黃金交叉 (+5分)")
        elif latest['K'] > 80: details.append(f"KD 指標進入高檔超買區 (K:{latest['K']:.1f})，注意過熱回檔 (0分)")
        else: score += 5; details.append(f"KD 指標位於中性區間 (K:{latest['K']:.1f}) (+5分)")
        
        recent_low, recent_high = self.df['Low'].tail(20).min(), self.df['High'].tail(20).max()
        if latest['Close'] <= recent_low * 1.03: score += 10; details.append(f"當前股價接近近20日支撐點({recent_low:.2f})，相對抗跌 (+10分)")
        return score, details, recent_low, recent_high

    def analyze_chips(self):
        score, details = 0, []
        if self.market_label == "美股/全球":
            score += 10; details.append("海外市場不適用台灣集保與三大法人分析，給予中立基本分 (+10分)")
            return score, details

        print(f"[2/2] 正在透過 FinMind API 獲取 {self.stock_name} 真實法人籌碼數據...")
        start_date = (datetime.date.today() - datetime.timedelta(days=40)).strftime("%Y-%m-%d")
        try:
            df_inst = self.fm.taiwan_stock_institutional_investors(stock_id=self.raw_ticker, start_date=start_date)
            if not df_inst.empty:
                df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
                recent_3_days_net = df_inst.groupby('date')['net_buy'].sum().tail(3).sum() / 1000
                if recent_3_days_net > 100: score += 10; details.append(f"三大法人近 3 日強勢買超共 {recent_3_days_net:,.0f} 張，籌碼點火 (+10分)")
                elif recent_3_days_net > -100: score += 5; details.append(f"三大法人近 3 日呈現中性橫盤，淨變動 {recent_3_days_net:,.0f} 張 (+5分)")
                else: details.append(f"三大法人近 3 日大舉調節賣超 {abs(recent_3_days_net):,.0f} 張，上方壓力重 (0分)")
            else: score += 5; details.append("近期無三大法人進出數據，給予中性基本分 (+5分)")

            df_shares = self.fm.taiwan_stock_holding_shares_per(stock_id=self.raw_ticker, start_date=start_date)
            if not df_shares.empty:
                df_big_holders = df_shares[df_shares['HoldingSharesLevel'] == 'more than 1,000,001'].sort_values(by='date')
                if len(df_big_holders) >= 2:
                    diff = df_big_holders['percent'].iloc[-1] - df_big_holders['percent'].iloc[-2]
                    if diff > 0.1: score += 10; details.append(f"千張大戶持股比率單週增加 {diff:+.2f}%，大戶進場 (+10分)")
                    elif diff < -0.1: details.append(f"千張大戶持股比率單週減少 {abs(diff):-.2f}%，大戶出貨 (0分)")
                    else: score += 5; details.append(f"千張大戶持股比率維持穩定 (+5分)")
                else: score += 5; details.append("集保大戶歷史數據不足，給予中性基本分 (+5分)")
            else: score += 5; details.append("未獲取到股權分散表數據，給予中性基本分 (+5分)")
        except: score += 10; details.append("API 限流或無籌碼資料，啟動安全防護機制改給中立計分 (+10分)")
        return score, details

    def _get_advice_level(self, total_score):
        if total_score >= 75: return "強烈建議進場"
        elif total_score >= 60: return "建議進場"
        elif total_score >= 45: return "平平 (觀望為主)"
        elif total_score >= 30: return "不建議進場"
        else: return "強烈不建議進場"

    def evaluate(self):
        self.fetch_data()
        self.calculate_indicators()
        fund_score, fund_details, eps = self.analyze_fundamentals()
        tech_score, tech_details, support, resistance = self.analyze_technicals()
        chip_score, chip_details = self.analyze_chips()

        f_pct, t_pct, c_pct = (fund_score / 30) * 100, (tech_score / 40) * 100, (chip_score / 20) * 100
        user_w = self.horizons[self.matched_horizon]
        weighted_f, weighted_t, weighted_c = f_pct * user_w['fund'], t_pct * user_w['tech'], c_pct * user_w['chip']
        max_f, max_t, max_c = user_w['fund'] * 100, user_w['tech'] * 100, user_w['chip'] * 100

        user_total = weighted_f + weighted_t + weighted_c
        eps_display = f"{eps:.2f}" if eps is not None else "暫無資料"

        print("\n" + "="*75)
        print(f" 【 {self.stock_name} ({self.ticker}) 】 個股量化策略多空評估報告 ")
        print("="*75)
        print(f" 現前股價: {self.df['Close'].iloc[-1]:,.2f} 元  |  近四季累積 EPS: {eps_display} 元")
        print(f" 💡 尋找低點效益 - 建議黃金進場價格區間: {support:,.2f} ~ {support * 1.03:,.2f} 元")
        print("-" * 75)
        print("【 依照各投資年限之進場建議與戰略分析 】")
        print(f" (1) 針對您的投資年限【{self.user_horizon_text}】 (模型對應: {self.matched_horizon})")
        print(f"    ⭐ 專屬綜合評分: {user_total:.1f}/100 | 進場建議: 👉 【 {self._get_advice_level(user_total)} 】")
        print(f"    💡 策略說明: {user_w['desc']} (權重: 基本面{max_f:.0f}% / 技術面{max_t:.0f}% / 籌碼面{max_c:.0f}%)")
        print("-" * 75)
        
        seq = 2
        for horizon, w in self.horizons.items():
            h_total = (f_pct * w['fund']) + (t_pct * w['tech']) + (c_pct * w['chip'])
            print(f" ({seq}) {horizon} 投資建議: 【 {self._get_advice_level(h_total)} 】 (評分: {h_total:.1f}/100)")
            seq += 1

        print("="*75)
        print(f"\n🔍 【第一維度：基本面價值評估】 (得分: {weighted_f:.1f} / {max_f:.0f})")
        for d in fund_details: print(f"  • {d}")
        print(f"\n📈 【第二維度：技術面擇時評估】 (得分: {weighted_t:.1f} / {max_t:.0f})")
        for d in tech_details: print(f"  • {d}")
        print(f"  • 參考波段壓力位(近20日高點): {resistance:,.2f} 元")
        print(f"\n📊 【第三維度：籌碼面法人大戶追蹤】 (得分: {weighted_c:.1f} / {max_c:.0f})")
        if max_c == 0: print("  • 說明：由於長線存股，系統已將短期籌碼權重歸零，不影響最終評分。")
        for d in chip_details: print(f"  • {d}")
        print("="*75)

class MasterRoutingSystem:
    def __init__(self):
        self.fm, self._stock_info_cache = DataLoader(), None
    def auto_detect_type(self, raw_ticker):
        print("\n正在連接市場資料庫，自動辨識證券資訊...")
        is_tw = any(char.isdigit() for char in raw_ticker)
        ticker_yf, is_etf, stock_name, market_label = raw_ticker, False, "未知名稱", "美股/全球"
        if is_tw:
            try:
                if self._stock_info_cache is None: self._stock_info_cache = self.fm.taiwan_stock_info()
                target = self._stock_info_cache[self._stock_info_cache['stock_id'] == raw_ticker]
                if not target.empty:
                    stock_name = target.iloc[0]['stock_name']
                    industry, market_type = str(target.iloc[0].get('industry_category', '')).upper(), str(target.iloc[0].get('type', '')).lower()
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
            info = yf.Ticker(ticker_yf).info or {}
            if info.get('quoteType') == 'ETF': is_etf = True
            elif info.get('quoteType') == 'EQUITY': is_etf = False
            if stock_name == "未知名稱": stock_name = info.get('shortName', info.get('longName', raw_ticker))
        except: pass
        return ticker_yf, is_etf, stock_name, market_label

# ==========================================
# 網頁 UI 綁定層
# ==========================================
st.title("🏦 台美股/ETF 雙引擎量化進場決策系統")
st.write("這是一套結合基本面、技術面與籌碼面的法人級量化模型。")

col1, col2 = st.columns(2)
with col1:
    raw_ticker = st.text_input("1. 請輸入股票或 ETF 代號 (如: 0050, 2330, TSLA)", value="2412").strip().upper().replace('.TW', '').replace('.TWO', '')
with col2:
    horizon_input = st.text_input("2. 請輸入預計投資的時間年限 (如: 10年, 半年, 存股, 當沖)", value="1年").strip()

if st.button("🚀 開始分析"):
    if raw_ticker:
        with st.spinner("正在連接市場資料庫並進行大量運算，請稍候..."):
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    system = MasterRoutingSystem()
                    matched_horizon = parse_investment_horizon(horizon_input)
                    horizon_years = get_horizon_years(matched_horizon)
                    ticker_yf, is_etf, stock_name, market_label = system.auto_detect_type(raw_ticker)
                    
                    print(f"✅ 成功辨識：{stock_name} ({ticker_yf}) - 屬於【{market_label}】市場")
                    
                    if is_etf:
                        print(f"🎯 系統判定為【ETF】，啟動【ETF量化籌碼與波段分析引擎】...\n")
                        analyzer = ETFAnalyzer(raw_ticker, ticker_yf, stock_name, horizon_years)
                        analyzer.fetch_data()
                        analyzer.fetch_institutional_data()
                        analyzer.calculate_indicators()
                        analyzer.analyze_score()
                        analyzer.display_report()
                    else:
                        print(f"🎯 系統判定為【個股】，啟動【個股多維度價值與技術分析引擎】...\n")
                        evaluator = StockEvaluator(raw_ticker, ticker_yf, stock_name, market_label, matched_horizon, system.fm)
                        evaluator.evaluate()
                        
                except Exception as e:
                    print(f"\n❌ 系統執行過程中發生錯誤: {e}")
                    print("請檢查您的網路連線，或確認輸入的證券代號是否有效。")
            
            report_output = f.getvalue()
            st.code(report_output, language="text")
            st.success("分析完成！")
    else:
        st.warning("請先輸入股票代號！")