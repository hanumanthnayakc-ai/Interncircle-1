"""
End-to-end e-commerce analysis: cleaning, revenue trends, RFM customer segments,
churn, cohort retention and revenue concentration.

Usage:
    python ecommerce_analysis.py                    # runs on a built-in SYNTHETIC dataset
    python ecommerce_analysis.py orders.csv [outdir]

Expected columns in orders.csv (one row per order line):
    order_id, customer_id, order_date, product_category, quantity, unit_price, region
Optional: channel
Returns are rows with quantity <= 0 (they are separated from sales).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

CHURN_DAYS = 90  # no purchase for this long = churned

# ---- styling (validated categorical slots 1-2, single-hue sequential) -------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def header(fig, title, subtitle, left=0.06):
    fig.text(left, 0.97, title, fontsize=15, fontweight="bold", va="top")
    fig.text(left, 0.925, subtitle, fontsize=10.5, va="top", color=INK2)


def legend_row(fig, items, y=0.885, left=0.06):
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in items]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, y), ncol=len(items),
               fontsize=10.5, labelcolor=INK2, handlelength=1.0, handleheight=1.0,
               columnspacing=1.6)


# =============================================================================
# 1. SYNTHETIC DATA  (only used when no CSV is supplied)
# =============================================================================
def make_synthetic(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.period_range("2025-01", "2026-10", freq="M")
    season = {1: .85, 2: .8, 3: .9, 4: .9, 5: .95, 6: .9, 7: .9, 8: .95, 9: 1.0,
              10: 1.15, 11: 1.6, 12: 1.4}
    cats = {"Electronics": (180, .5), "Home & Kitchen": (55, .5), "Fashion": (45, .45),
            "Beauty": (28, .4), "Sports": (60, .5), "Books": (15, .35)}
    cat_names = list(cats)
    types = ["loyal", "regular", "occasional", "one_time"]
    type_p = [.10, .35, .35, .20]
    rate = {"loyal": 1.3, "regular": .5, "occasional": .16, "one_time": 0}
    churn = {"loyal": .015, "regular": .05, "occasional": .07, "one_time": 1}
    value_mult = {"loyal": 1.25, "regular": 1.0, "occasional": .9, "one_time": .85}
    acq_w = np.linspace(1, 2.2, len(months))
    acq_w /= acq_w.sum()

    rows, oid = [], 0
    for cid in range(1, 3501):
        t = rng.choice(types, p=type_p)
        m = rng.choice(len(months), p=acq_w)
        region = rng.choice(["North", "South", "East", "West"], p=[.30, .25, .20, .25])
        channel = rng.choice(["Web", "Mobile app", "Marketplace"], p=[.45, .40, .15])
        pref = rng.choice(cat_names, p=[.20, .25, .25, .12, .10, .08])
        first, alive = True, True
        while m < len(months) and alive:
            per = months[m]
            n_orders = 1 if first else rng.poisson(rate[t] * season[per.month])
            first = False
            for _ in range(n_orders):
                oid += 1
                day = int(rng.integers(1, per.days_in_month + 1))
                date = pd.Timestamp(per.year, per.month, day)
                for _ in range(int(rng.integers(1, 4))):
                    cat = pref if rng.random() < .5 else rng.choice(cat_names)
                    base, sig = cats[cat]
                    price = round(float(rng.lognormal(np.log(base), sig)) * value_mult[t], 2)
                    rows.append((f"O{oid:06d}", cid, date, cat, 1 + int(rng.poisson(.4)),
                                 price, region, channel))
            alive = rng.random() >= churn[t]
            m += 1
    df = pd.DataFrame(rows, columns=["order_id", "customer_id", "order_date", "product_category",
                                     "quantity", "unit_price", "region", "channel"])
    # add realistic mess
    ret = df.sample(frac=.02, random_state=1).copy()
    ret["quantity"] = -ret["quantity"]
    ret["order_id"] = "R" + ret["order_id"]
    dup = df.sample(frac=.01, random_state=2)
    df.loc[df.sample(frac=.02, random_state=3).index, "customer_id"] = np.nan
    df = pd.concat([df, ret, dup], ignore_index=True).sample(frac=1, random_state=4)
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


# =============================================================================
# 2. CLEANING
# =============================================================================
def clean(raw: pd.DataFrame):
    log = {"raw_rows": len(raw)}
    df = raw.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    log["bad_dates_dropped"] = int(df["order_date"].isna().sum())
    df = df.dropna(subset=["order_date"])
    n = len(df)
    df = df.drop_duplicates()
    log["duplicate_rows_dropped"] = n - len(df)
    n = len(df)
    df = df.dropna(subset=["customer_id"])
    log["missing_customer_id_dropped"] = n - len(df)
    df["customer_id"] = df["customer_id"].astype(int)
    df["revenue"] = df["quantity"] * df["unit_price"]
    returns = df[(df["quantity"] <= 0) | (df["unit_price"] <= 0)]
    sales = df.drop(returns.index)
    log["return_rows"] = len(returns)
    log["returned_value"] = round(float(returns["revenue"].abs().sum()), 2)
    log["clean_sales_rows"] = len(sales)
    sales = sales.assign(month=sales["order_date"].dt.to_period("M"))
    return sales, log


# =============================================================================
# 3. ANALYSIS
# =============================================================================
def segment_of(r, f):
    if r >= 4 and f >= 4: return "Champions"
    if r >= 3 and f >= 3: return "Loyal"
    if r >= 4 and f <= 2: return "New / promising"
    if r <= 2 and f >= 3: return "At risk"
    if r <= 2 and f <= 2: return "Lost / hibernating"
    return "Needs attention"


def quintile(series, higher_is_better=True):
    rank = series.rank(method="first", ascending=higher_is_better)
    return pd.qcut(rank, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def analyse(sales: pd.DataFrame):
    snapshot = sales["order_date"].max() + pd.Timedelta(days=1)

    # ---- monthly trend
    monthly = sales.groupby("month").agg(revenue=("revenue", "sum"),
                                         orders=("order_id", "nunique"),
                                         customers=("customer_id", "nunique"))
    monthly["aov"] = monthly["revenue"] / monthly["orders"]
    monthly["mom_pct"] = monthly["revenue"].pct_change() * 100
    monthly["rolling3"] = monthly["revenue"].rolling(3).mean()

    # ---- customer table + RFM
    cust = sales.groupby("customer_id").agg(
        first_order=("order_date", "min"), last_order=("order_date", "max"),
        orders=("order_id", "nunique"), revenue=("revenue", "sum"))
    cust["recency_days"] = (snapshot - cust["last_order"]).dt.days
    cust["R"] = quintile(cust["recency_days"], higher_is_better=False)
    cust["F"] = quintile(cust["orders"] + cust["revenue"] / (cust["revenue"].max() * 10))
    cust["M"] = quintile(cust["revenue"])
    cust["segment"] = [segment_of(r, f) for r, f in zip(cust["R"], cust["F"])]
    cust["churned"] = cust["recency_days"] > CHURN_DAYS
    cust["eligible"] = cust["first_order"] <= snapshot - pd.Timedelta(days=CHURN_DAYS)

    seg = cust.groupby("segment").agg(customers=("revenue", "size"), revenue=("revenue", "sum"),
                                      avg_orders=("orders", "mean"),
                                      avg_recency=("recency_days", "mean"))
    seg["customer_share"] = seg["customers"] / seg["customers"].sum() * 100
    seg["revenue_share"] = seg["revenue"] / seg["revenue"].sum() * 100
    seg = seg.sort_values("revenue", ascending=False)

    # ---- churn among customers old enough to have churned
    el = cust[cust["eligible"]].copy()
    el["band"] = pd.cut(el["orders"], [0, 1, 2, 5, 10_000], labels=["1 order", "2 orders", "3-5 orders", "6+ orders"])
    churn_band = el.groupby("band", observed=True)["churned"].agg(["mean", "sum", "size"])
    churn_band["mean"] *= 100

    # ---- cohorts
    sales = sales.join(cust["first_order"].dt.to_period("M").rename("cohort"), on="customer_id")
    sales["k"] = (sales["month"] - sales["cohort"]).apply(lambda x: x.n)
    active = sales.groupby(["cohort", "k"])["customer_id"].nunique().unstack()
    retention = active.div(active[0], axis=0) * 100
    last_month = sales["month"].max()
    for c in retention.index:  # blank the future
        retention.loc[c, [k for k in retention.columns if c + k > last_month]] = np.nan

    # ---- concentration
    rev_sorted = cust["revenue"].sort_values(ascending=False).values
    cum_rev = np.cumsum(rev_sorted) / rev_sorted.sum() * 100
    cum_cust = np.arange(1, len(rev_sorted) + 1) / len(rev_sorted) * 100
    top20 = float(cum_rev[int(len(rev_sorted) * .2) - 1])

    by_cat = sales.groupby("product_category")["revenue"].sum().sort_values(ascending=False)
    by_region = sales.groupby("region")["revenue"].sum().sort_values(ascending=False)
    by_channel = (sales.groupby("channel")["revenue"].sum().sort_values(ascending=False)
                  if "channel" in sales else None)

    # ---- repeat-purchase rate and YoY
    repeat_rate = float((cust["orders"] >= 2).mean() * 100)
    last_year = last_month.year
    ytd_now = monthly.loc[monthly.index.year == last_year, "revenue"].sum()
    months_now = [m.month for m in monthly.index if m.year == last_year]
    ytd_prev = monthly.loc[[m for m in monthly.index
                            if m.year == last_year - 1 and m.month in months_now], "revenue"].sum()
    yoy = (ytd_now / ytd_prev - 1) * 100 if ytd_prev else float("nan")

    metrics = {
        "snapshot_date": str(snapshot.date()),
        "period": f"{monthly.index.min()} to {monthly.index.max()}",
        "total_revenue": round(float(sales["revenue"].sum()), 2),
        "orders": int(sales["order_id"].nunique()),
        "customers": int(len(cust)),
        "avg_order_value": round(float(sales.groupby('order_id')['revenue'].sum().mean()), 2),
        "repeat_customer_rate_pct": round(repeat_rate, 1),
        "ytd_yoy_growth_pct": round(float(yoy), 1),
        "top20_customers_revenue_share_pct": round(top20, 1),
        "overall_churn_rate_pct": round(float(el["churned"].mean() * 100), 1),
        "peak_month": str(monthly["revenue"].idxmax()),
        "peak_month_revenue": round(float(monthly["revenue"].max()), 2),
    }
    return dict(monthly=monthly, cust=cust, seg=seg, churn_band=churn_band,
                retention=retention, cum=(cum_cust, cum_rev), top20=top20, by_cat=by_cat,
                by_region=by_region, by_channel=by_channel, metrics=metrics)


# =============================================================================
# 4. CHARTS
# =============================================================================
def k_fmt(v, _): return f"{v / 1000:.0f}k"


def charts(r, out: Path):
    m = r["monthly"]
    x = m.index.to_timestamp()

    # 1. revenue trend
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.08, right=0.95)
    ax.plot(x, m["revenue"], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(x, m["rolling3"], color=ORANGE, lw=2, solid_capstyle="round")
    ax.yaxis.set_major_formatter(FuncFormatter(k_fmt))
    ax.set_ylim(0, m["revenue"].max() * 1.18)
    pk = m["revenue"].idxmax().to_timestamp()
    ax.annotate(f"Peak {pk:%b %Y}: {m['revenue'].max() / 1000:,.0f}k", (pk, m["revenue"].max()),
                xytext=(-10, 14), textcoords="offset points", ha="right", color=INK, fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Monthly revenue (thousands)")
    header(fig, f"Revenue keeps climbing; {pk:%B %Y} is the peak month",
           f"Monthly revenue; year-to-date growth {r['metrics']['ytd_yoy_growth_pct']:+.0f}% vs same months last year")
    legend_row(fig, [("Monthly revenue", BLUE), ("3-month average", ORANGE)])
    fig.savefig(out / "01_revenue_trend.png", dpi=160)
    plt.close(fig)

    # 2. segment share of customers vs revenue
    s = r["seg"].sort_values("revenue_share")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(top=0.76, bottom=0.1, left=0.2, right=0.93)
    y = np.arange(len(s))
    h = 0.36
    ax.barh(y + h / 2, s["revenue_share"], height=h, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.barh(y - h / 2, s["customer_share"], height=h, color=ORANGE, edgecolor=SURFACE, linewidth=2)
    for i, (rs, cs) in enumerate(zip(s["revenue_share"], s["customer_share"])):
        ax.text(rs + .5, i + h / 2, f"{rs:.0f}%", va="center", fontsize=9.5, color=INK2)
        ax.text(cs + .5, i - h / 2, f"{cs:.0f}%", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(y, s.index)
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Share of total (%)")
    top = r["seg"].sort_values("revenue", ascending=False).iloc[0]
    header(fig, "A small set of segments carries the revenue",
           f"Share of customers vs share of revenue by RFM segment; "
           f"{top.name} alone is {top['revenue_share']:.0f}% of revenue", left=0.04)
    legend_row(fig, [("Share of revenue", BLUE), ("Share of customers", ORANGE)], left=0.04)
    fig.savefig(out / "02_segments_revenue_vs_customers.png", dpi=160)
    plt.close(fig)

    # 3. cohort retention heatmap (months 1-12)
    ret = r["retention"].loc[:, 1:12]
    ret.index = [str(p.to_timestamp().strftime("%b %y")) for p in ret.index]
    fig, ax = plt.subplots(figsize=(10, 7.6))
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.1, right=0.92)
    sns.heatmap(ret, cmap=SEQ, vmin=0, vmax=np.nanpercentile(ret.values, 98), linewidths=2,
                linecolor=SURFACE, ax=ax, square=False,
                cbar_kws={"shrink": .6, "label": "% of cohort ordering"})
    ax.grid(False)
    ax.tick_params(length=0, labelsize=9.5, colors=INK2)
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("Months after first purchase")
    ax.set_ylabel("First-purchase cohort")
    ax.collections[0].colorbar.outline.set_visible(False)
    avg = r["retention"].mean()
    header(fig, f"About {avg[1]:.0f}% of a cohort orders again in month 1, {avg[12]:.0f}% by month 12",
           "Share of each monthly cohort that placed an order, by months since first purchase "
           "(title = average across cohorts)", left=0.06)
    fig.savefig(out / "03_cohort_retention.png", dpi=160)
    plt.close(fig)

    # 4. concentration curve
    cc, cr = r["cum"]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    fig.subplots_adjust(top=0.8, bottom=0.11, left=0.12, right=0.95)
    ax.plot([0, 100], [0, 100], color=BASE, lw=1.5)
    ax.plot(np.r_[0, cc], np.r_[0, cr], color=BLUE, lw=2.5)
    ax.plot([20], [r["top20"]], marker="o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=5)
    ax.annotate(f"Top 20% of customers\n= {r['top20']:.0f}% of revenue", (20, r["top20"]),
                xytext=(14, -26), textcoords="offset points", color=INK, fontsize=10.5)
    ax.text(62, 52, "Even split", color=MUTED, fontsize=10, rotation=34)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Cumulative share of customers (highest spenders first, %)")
    ax.set_ylabel("Cumulative share of revenue (%)")
    ax.grid(axis="x", visible=True)
    header(fig, "Revenue is concentrated in top customers",
           "Cumulative revenue by customer rank", left=0.12)
    fig.savefig(out / "04_revenue_concentration.png", dpi=160)
    plt.close(fig)

    # 5. category and region
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.15, right=0.96, wspace=0.55)
    for ax, ser, title in ((axes[0], r["by_cat"], "By product category"),
                           (axes[1], r["by_region"], "By region")):
        share = (ser / ser.sum() * 100).sort_values()
        ax.barh(share.index, share.values, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(share.values):
            ax.text(v + .4, i, f"{v:.0f}%", va="center", fontsize=9.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=11.5, pad=8)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.tick_params(axis="y", length=0, colors=INK2)
        ax.set_xlabel("Share of revenue (%)")
    header(fig, "Where the revenue comes from", "Share of total revenue (returns excluded)", left=0.04)
    fig.savefig(out / "05_category_region_mix.png", dpi=160)
    plt.close(fig)

    # 6. churn by purchase frequency
    cb = r["churn_band"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.1, right=0.96)
    ax.bar(cb.index.astype(str), cb["mean"], color=BLUE, width=0.55, edgecolor=SURFACE, linewidth=2)
    for i, (v, n) in enumerate(zip(cb["mean"], cb["size"])):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10.5, color=INK)
        ax.text(i, 2.5, f"n={n:,}", ha="center", fontsize=9, color="#ffffff")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(f"Churned (no order in {CHURN_DAYS}+ days), %")
    ax.set_xlabel("Lifetime orders")
    header(fig, "Churn falls steeply with each repeat purchase",
           f"Customers with no order in {CHURN_DAYS}+ days, among those acquired at least {CHURN_DAYS} days ago",
           left=0.1)
    fig.savefig(out / "06_churn_by_frequency.png", dpi=160)
    plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    out = Path(args[1] if len(args) > 1 else "output")
    out.mkdir(parents=True, exist_ok=True)

    if args:
        raw = pd.read_csv(args[0])
        print(f"Loaded {args[0]}: {len(raw):,} rows")
    else:
        raw = make_synthetic()
        raw.to_csv(out / "synthetic_orders.csv", index=False)
        print(f"SYNTHETIC dataset generated: {len(raw):,} rows (saved to {out/'synthetic_orders.csv'})")

    sales, log = clean(raw)
    print("\nCleaning log:", json.dumps(log, indent=2))
    res = analyse(sales)

    pd.set_option("display.width", 140)
    print("\nMetrics:", json.dumps(res["metrics"], indent=2))
    print("\nSegments:\n", res["seg"].round(1))
    print("\nChurn by lifetime orders (eligible customers):\n", res["churn_band"].round(1))
    print("\nMonthly revenue:\n", res["monthly"][["revenue", "orders", "customers", "aov", "mom_pct"]].round(1))
    print("\nRevenue by category:\n", (res["by_cat"] / res["by_cat"].sum() * 100).round(1))
    print("\nRevenue by region:\n", (res["by_region"] / res["by_region"].sum() * 100).round(1))
    print("\nMean retention by month since first purchase (%):\n",
          res["retention"].mean().round(1).loc[1:12])

    res["cust"].to_csv(out / "customer_rfm_segments.csv")
    res["monthly"].to_csv(out / "monthly_revenue.csv")
    (out / "metrics.json").write_text(json.dumps({"cleaning": log, **res["metrics"]}, indent=2))
    charts(res, out)
    print("\nOutputs in", out.resolve())
