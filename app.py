import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="폐암 환자 군집 분석 시스템",
    page_icon="🫁",
    layout="centered",
)

# ─────────────────────────────────────────
# CSS 스타일
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* 제목 */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-desc {
        text-align: center;
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* 섹션 헤더 */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 2px solid #eee;
        padding-bottom: 0.4rem;
    }

    /* 결과 박스 */
    .result-box-risk {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        color: #856404;
    }
    .result-box-healthy {
        background-color: #d1e7dd;
        border: 1px solid #198754;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        color: #0f5132;
    }
    .result-box-very-healthy {
        background-color: #cfe2ff;
        border: 1px solid #0d6efd;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        color: #084298;
    }

    /* 범례 설명 */
    .legend-desc {
        font-size: 0.9rem;
        color: #444;
        margin-top: 0.4rem;
    }

    /* 버튼 커스텀 */
    .stButton > button {
        width: 100%;
        background-color: #fff;
        border: 1.5px solid #aaa;
        border-radius: 8px;
        font-size: 1rem;
        padding: 0.55rem;
        font-family: 'Noto Sans KR', sans-serif;
        cursor: pointer;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 샘플 데이터 & 모델 학습 (캐싱)
# ─────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)

    # 군집 0: 매우 건강군 – 흡연·음주 적음
    c0 = np.column_stack([
        np.random.normal(5, 2, 30),   # 흡연량
        np.random.normal(1, 0.8, 30), # 음주량
        np.random.normal(45, 8, 30),  # 나이
    ])
    # 군집 1: 위험군 – 흡연 중간·음주 중간
    c1 = np.column_stack([
        np.random.normal(15, 3, 30),
        np.random.normal(5, 1.2, 30),
        np.random.normal(55, 7, 30),
    ])
    # 군집 2: 건강군 – 흡연·음주 높음
    c2 = np.column_stack([
        np.random.normal(25, 3, 30),
        np.random.normal(7, 1.2, 30),
        np.random.normal(60, 6, 30),
    ])

    data = np.vstack([c0, c1, c2])
    labels_true = np.array([0]*30 + [1]*30 + [2]*30)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(data_scaled)

    # 클러스터 레이블을 실제 의미에 맞게 재매핑
    # (KMeans 결과가 항상 0,1,2 순서가 아닐 수 있으므로 중심점으로 매핑)
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    smoking_order = np.argsort(centers_original[:, 0])  # 흡연량 기준 정렬
    label_map = {smoking_order[0]: 0, smoking_order[1]: 1, smoking_order[2]: 2}

    raw_labels = kmeans.labels_
    mapped_labels = np.array([label_map[l] for l in raw_labels])

    return kmeans, scaler, data, mapped_labels, label_map


kmeans, scaler, train_data, train_labels, label_map = train_model()

CLUSTER_NAMES = {0: "매우 건강군", 1: "위험군", 2: "건강군"}
CLUSTER_COLORS = {0: "#4e9af1", 1: "#f4c430", 2: "#2ecc71"}
RESULT_BOX_CLASS = {0: "result-box-very-healthy", 1: "result-box-risk", 2: "result-box-healthy"}


# ─────────────────────────────────────────
# 예측 함수
# ─────────────────────────────────────────
def predict_cluster(age, smoking, alcohol):
    x = np.array([[smoking, alcohol, age]])
    x_scaled = scaler.transform(x)
    raw_pred = kmeans.predict(x_scaled)[0]
    return label_map[raw_pred]


# ─────────────────────────────────────────
# 시각화 함수
# ─────────────────────────────────────────
def draw_scatter(smoking_input, alcohol_input, predicted_cluster):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for cluster_id in [0, 1, 2]:
        mask = train_labels == cluster_id
        ax.scatter(
            train_data[mask, 0],
            train_data[mask, 1],
            c=CLUSTER_COLORS[cluster_id],
            s=60,
            alpha=0.75,
            edgecolors="none",
            label=f"{cluster_id}번 군집 ({CLUSTER_NAMES[cluster_id]})",
            zorder=2,
        )

    # 현재 환자 위치 (별 모양)
    ax.scatter(
        smoking_input,
        alcohol_input,
        c=CLUSTER_COLORS[predicted_cluster],
        s=220,
        marker="*",
        edgecolors="#333",
        linewidths=0.8,
        zorder=5,
        label="현재 환자",
    )

    ax.set_xlabel("흡연량", fontsize=11)
    ax.set_ylabel("음주량", fontsize=11)
    ax.set_title("군집 시각화", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, color="#ccc")
    ax.spines[["top", "right"]].set_visible(False)

    legend = ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#ddd",
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# UI 레이아웃
# ─────────────────────────────────────────
st.markdown('<div class="main-title">🫁 폐암 환자 군집 분석 시스템</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-desc">AI가 환자의 특성을 분석하여<br>어떤 군집(유형)에 속하는지 예측합니다.</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── 입력 섹션 ──
st.markdown('<div class="section-header">📋 환자 정보 입력</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("나이", min_value=0.0, max_value=120.0, value=50.0, step=1.0, format="%.2f")
with col2:
    smoking = st.number_input("흡연량", min_value=0.0, max_value=100.0, value=10.0, step=1.0, format="%.2f")
with col3:
    alcohol = st.number_input("음주량", min_value=0.0, max_value=50.0, value=5.0, step=1.0, format="%.2f")

st.divider()

# ── 분석 버튼 ──
if st.button("🔍 군집 분석하기"):
    cluster = predict_cluster(age, smoking, alcohol)
    cluster_name = CLUSTER_NAMES[cluster]
    box_class = RESULT_BOX_CLASS[cluster]

    # 결과 메시지
    st.markdown(
        f'<div class="{box_class}">이 환자는 {cluster}번 군집에 속합니다.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="legend-desc">0번은 매우 건강군, 1번은 위험군, 2번은 건강군입니다.</p>',
        unsafe_allow_html=True,
    )

    # 산점도
    fig = draw_scatter(smoking, alcohol, cluster)
    st.pyplot(fig)

    # 상세 설명
    with st.expander("📊 군집별 상세 설명"):
        st.markdown("""
| 군집 | 이름 | 특징 |
|------|------|------|
| 0번 | 매우 건강군 | 흡연량·음주량이 낮고 건강 위험도가 낮습니다. |
| 1번 | 위험군 | 흡연량·음주량이 중간 수준으로 관리가 필요합니다. |
| 2번 | 건강군 | 흡연량·음주량이 높지만 상대적으로 안정적입니다. |
        """)
