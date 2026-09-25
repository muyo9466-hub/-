import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="台美股支撐壓力分析器", layout="wide")
st.title("📈 台美股智慧支撐與壓力分析 App")

# ------------------------------------------------------------------
# 側邊欄輸入
# ------------------------------------------------------------------
st.sidebar.header("設定")
ticker_input = st.sidebar.text_input("輸入股票代號 (美股如 AAPL，台股如 2330.TW)", value="2330.TW")
period = st.sidebar.selectbox("回測區間（抓多長的歷史資料）", ["3mo", "6mo", "1y"], index=1)

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "🔎 分析焦點",
    ["🔍 短線放大鏡（近3個月）", "🌍 全景放大鏡（6個月～1年）"],
    help="切換後，下方「進階參數」的預設回溯天數會自動調整；你仍可以手動微調每一項。",
)
is_short_mode = mode.startswith("🔍")

if is_short_mode:
    st.sidebar.caption("目前模式：精準抓取近期爆量長紅K與短線支撐，適合短線波段操作。")
    default_spike_lookback = 30
    default_pressure_lookback = 60
else:
    st.sidebar.caption("目前模式：搭配 MA60 季線與大波段前高，適合判斷中長期趨勢格局。")
    default_spike_lookback = 90
    default_pressure_lookback = 120

with st.sidebar.expander("進階參數（可微調演算法靈敏度）", expanded=False):
    st.caption("轉折點 / 群聚 / 爆量判定")
    pivot_window = st.slider("轉折點偵測窗口（左右各 N 根 K 棒）", 2, 10, 4, help="數字越大，找到的前高/前低越「重要」但數量越少")
    cluster_tol_pct = st.slider("價位群聚容忍度 (%)", 0.3, 3.0, 1.0, step=0.1, help="價位差距在此百分比內視為同一個支撐/壓力區")
    vol_spike_mult = st.slider("爆量倍數門檻（相對 20 日均量）", 1.5, 4.0, 2.0, step=0.1)
    body_atr_mult = st.slider("長紅K實體門檻（相對 14 日 ATR）", 0.3, 1.5, 0.7, step=0.1, help="數字越大，篩選出的紅K實體越長、訊號越強")

    st.caption("各指標回溯天數（依你指定的規則設定）")
    consolidation_lookback = st.slider(
        "近端整理平台／近期高低點 回溯天數", 10, 40, 20,
        help="固定概念：抓近 20 日高低點作為短線壓力/支撐參考，此處可再微調"
    )
    spike_scan_lookback = st.slider(
        "爆量長紅K 掃描範圍（天）", 20, 120, default_spike_lookback, key=f"spike_{mode}",
        help="在最近幾天內尋找符合『爆量＋長紅＋大實體』條件的K棒"
    )
    st.markdown("**第二支撐 (MA60／季線)：固定抓 60 日**（不可調整，這是季線的定義）")
    pressure_lookback = st.slider(
        "第一／第二壓力（前高、波段高點）回溯天數", 60, 120, default_pressure_lookback, key=f"pressure_{mode}",
        help="在最近 60～120 天內尋找轉折高點，作為壓力位的來源"
    )

cluster_tolerance = cluster_tol_pct / 100


# ------------------------------------------------------------------
# 資料抓取（含快取）
# ------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data


# ------------------------------------------------------------------
# 技術運算
# ------------------------------------------------------------------
def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def find_pivots(df: pd.DataFrame, left: int = 4, right: int = 4):
    """找出局部轉折高點（前高）與轉折低點（前低）。
    用左右各 N 根 K 棒的窗口比較，是簡化版的 zig-zag 轉折點偵測。"""
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index
    n = len(df)
    pivot_highs, pivot_lows = [], []

    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            pivot_highs.append((idx[i], float(highs[i])))

        window_l = lows[i - left:i + right + 1]
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            pivot_lows.append((idx[i], float(lows[i])))

    return pivot_highs, pivot_lows


def find_volume_spike_candles(df: pd.DataFrame, vol_mult: float, body_atr_mult: float, lookback: int) -> pd.DataFrame:
    """精準版爆量長紅K篩選：
    1) 成交量 > N 倍 20 日均量
    2) 收盤 > 開盤（多方力道）
    3) 實體長度 > N 倍 14 日 ATR（避免十字線/小實體誤判為「長紅」）
    回傳依成交量排序的候選 K 棒。"""
    window = df.tail(lookback).copy()
    atr14 = compute_atr(df, 14).reindex(window.index)
    body = window["Close"] - window["Open"]

    mask = (
        (window["Volume"] > window["Vol_MA20"] * vol_mult)
        & (body > 0)
        & (body > atr14 * body_atr_mult)
    )
    return window[mask].sort_values("Volume", ascending=False)


def cluster_levels(levels: list, tolerance: float = 0.01) -> list:
    """把彼此價差在 tolerance 比例內的候選價位合併成同一個支撐/壓力區，
    並記錄有幾個不同來源「重合」在這裡（重合越多，該價位通常越關鍵）。"""
    if not levels:
        return []

    levels_sorted = sorted(levels, key=lambda x: x["price"])
    clusters, current = [], [levels_sorted[0]]

    for lv in levels_sorted[1:]:
        ref = current[-1]["price"]
        if ref != 0 and abs(lv["price"] - ref) / ref <= tolerance:
            current.append(lv)
        else:
            clusters.append(current)
            current = [lv]
    clusters.append(current)

    result = []
    for c in clusters:
        avg_price = sum(x["price"] for x in c) / len(c)
        result.append({
            "price": avg_price,
            "touches": len(c),
            "tags": [x["tag"] for x in c],
            "dates": [x.get("date") for x in c if x.get("date") is not None],
        })
    return result


def describe_level(level) -> str:
    if level is None:
        return "（無足夠資料）"
    tag_str = "、".join(sorted(set(level["tags"])))
    touch_str = f"（{level['touches']} 個訊號重合，較關鍵）" if level["touches"] > 1 else ""
    return f"`{level['price']:.2f}` — {tag_str}{touch_str}"


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
if st.button("開始分析", type="primary"):
    ticker = ticker_input.strip().upper()
    if not ticker:
        st.warning("請輸入股票代號。")
        st.stop()

    with st.spinner("正在抓取即時數據與計算關鍵價位..."):
        try:
            df = load_data(ticker, period)
        except Exception as e:
            st.error(f"抓取資料時發生錯誤：{e}")
            st.stop()

        if df.empty:
            st.error("找不到該代號的資料，請確認格式是否正確（美股如 AAPL，台股上市加 .TW、上櫃加 .TWO）。")
            st.stop()

        if len(df) < 60:
            st.warning(
                f"⚠️ 目前資料只有 {len(df)} 根 K 棒，不足 60 根，"
                "MA60 季線支撐可能無法完整顯示，建議改選 6mo 或 1y。"
            )

        # --- 基本指標 ---
        df["MA60"] = df["Close"].rolling(window=60).mean()  # 第二支撐：固定 60 日，不受模式或參數影響
        df["Vol_MA20"] = df["Volume"].rolling(window=20).mean()
        df["High_Volume"] = df["Volume"] > (df["Vol_MA20"] * vol_spike_mult)

        current_price = float(df["Close"].iloc[-1])
        ma60_val = df["MA60"].iloc[-1]

        recent_high = float(df["High"].tail(consolidation_lookback).max())
        recent_low = float(df["Low"].tail(consolidation_lookback).min())

        # --- 爆量長紅K（精準版，回溯天數依模式/參數決定） ---
        spike_candidates = find_volume_spike_candles(df, vol_spike_mult, body_atr_mult, lookback=spike_scan_lookback)
        if not spike_candidates.empty:
            key_spike = spike_candidates.iloc[0]
            spike_low = float(key_spike["Low"])
            spike_date = key_spike.name
        else:
            spike_low, spike_date = None, None

        # --- 前高/前低轉折點（回溯天數依模式/參數決定，用來判斷「前高轉支撐」） ---
        pivot_lookback = min(len(df), pressure_lookback)
        pdf = df.tail(pivot_lookback)
        pivot_highs, pivot_lows = find_pivots(pdf, left=pivot_window, right=pivot_window)

        # --- 彙整所有候選價位，交給群聚演算法整理 ---
        candidates = []
        for date, price in pivot_highs:
            # 若此前高目前已被價格站上（price < current_price），
            # 代表壓力已被突破，依技術分析「極性反轉」原則，它會轉為支撐。
            tag = "前高轉支撐" if price < current_price else "前高壓力"
            candidates.append({"price": price, "tag": tag, "date": date})

        for date, price in pivot_lows:
            tag = "前低支撐" if price < current_price else "前低轉壓力"
            candidates.append({"price": price, "tag": tag, "date": date})

        candidates.append({"price": recent_high, "tag": f"近{consolidation_lookback}日高點", "date": None})
        candidates.append({"price": recent_low, "tag": f"近{consolidation_lookback}日低點", "date": None})
        if pd.notna(ma60_val):
            candidates.append({"price": float(ma60_val), "tag": "MA60季線", "date": None})
        if spike_low is not None:
            candidates.append({"price": spike_low, "tag": "爆量長紅K低點", "date": spike_date})

        clustered = cluster_levels(candidates, tolerance=cluster_tolerance)

        supports = sorted([c for c in clustered if c["price"] < current_price], key=lambda x: -x["price"])
        resistances = sorted([c for c in clustered if c["price"] > current_price], key=lambda x: x["price"])

        s1 = supports[0] if len(supports) > 0 else None
        s2 = supports[1] if len(supports) > 1 else None
        r1 = resistances[0] if len(resistances) > 0 else None
        r2 = resistances[1] if len(resistances) > 1 else None

        # ------------------------------------------------------------------
        # 繪圖
        # ------------------------------------------------------------------
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="K線"
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA60"], mode="lines", name="MA60季線",
            line=dict(color="orange", width=1.2, dash="dot")
        ))

        if not spike_candidates.empty:
            fig.add_trace(go.Scatter(
                x=spike_candidates.index, y=spike_candidates["Low"] * 0.995,
                mode="markers", name="爆量長紅K",
                marker=dict(symbol="triangle-up", size=10, color="cyan"),
            ))

        # 支撐/壓力水平線：第一支撐/壓力用實線，第二用虛線，顏色分開避免混淆
        level_styles = [
            (s1, "第一支撐", "#00cc66", "solid"),
            (s2, "第二支撐", "#66ffb3", "dash"),
            (r1, "第一壓力", "#ff4d4d", "solid"),
            (r2, "第二壓力", "#ff9999", "dash"),
        ]
        for level, label, color, dash in level_styles:
            if level is None:
                continue
            fig.add_hline(
                y=level["price"],
                line=dict(color=color, width=1.8, dash=dash),
                annotation_text=f"{label} {level['price']:.2f}",
                annotation_position="right",
                annotation_font_color=color,
            )

        fig.update_layout(
            title=f"{ticker} 股價走勢與支撐壓力圖　｜　{mode}",
            xaxis_title="日期", yaxis_title="價格",
            template="plotly_dark", height=650,
        )

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------------
        # 摘要看板
        # ------------------------------------------------------------------
        st.subheader("🎯 關鍵價位觀測站")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🛡️ 支撐防守區")
            st.markdown(f"- **現價參考**: `{current_price:.2f}`")
            st.markdown(f"- **第一支撐**: {describe_level(s1)}")
            st.markdown(f"- **第二支撐**: {describe_level(s2)}")
            if spike_low is not None:
                st.markdown(f"- **爆量長紅K低點**（近 {spike_scan_lookback} 天內）: `{spike_low:.2f}`（{spike_date.strftime('%Y-%m-%d')}）")
            else:
                st.markdown(f"- **爆量長紅K低點**: 近 {spike_scan_lookback} 天內無符合條件的爆量長紅K")

        with col2:
            st.markdown("### ⚔️ 上檔壓力區")
            st.markdown(f"- **第一壓力**: {describe_level(r1)}")
            st.markdown(f"- **第二壓力**: {describe_level(r2)}")
            st.markdown(f"- **近{consolidation_lookback}日高點**: `{recent_high:.2f}`")

        with st.expander("📋 所有偵測到的支撐/壓力價位（含來源與重合次數）"):
            rows = [{
                "價位": round(c["price"], 2),
                "類型": "壓力" if c["price"] > current_price else "支撐",
                "來源標籤": "、".join(sorted(set(c["tags"]))),
                "重合次數": c["touches"],
            } for c in sorted(clustered, key=lambda x: -x["price"])]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.caption(
            f"⚠️ 免責聲明：以上數據僅供參考，不構成投資建議。目前分析焦點：「{mode}」，"
            f"壓力位搜尋回溯 {pressure_lookback} 天、爆量K掃描回溯 {spike_scan_lookback} 天、"
            f"近端高低點回溯 {consolidation_lookback} 天、MA60 季線固定 60 天。"
            "「重合次數」代表有多少不同方法都指向同一價位區，僅供輔助判斷，不保證未來走勢會在此反應。"
        )
