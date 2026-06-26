import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import uuid

# --- ページ設定 ---
st.set_page_config(page_title="空気圧配管 圧力損失シミュレーター", layout="wide")

# ==========================================
# 1. サイドバー (モード選択 ＆ 環境条件)
# ==========================================
st.sidebar.header("シミュレーションモード")
mode = st.sidebar.radio(
    "メニューを選択", 
    [
        "➡️ 直列モデル (1本道)", 
        "🔀 並列・分岐モデル (3エリア拡張版)",
        "📖 使い方・Tips集 (FAQ)"
    ]
)

st.sidebar.markdown("---")
if mode != "📖 使い方・Tips集 (FAQ)":
    st.sidebar.header("1. 環境・流体条件")
    t = st.sidebar.slider("温度 t [℃]", min_value=0.0, max_value=50.0, value=20.0, step=1.0)
    p_gage = st.sidebar.number_input("供給圧力 p1_gage [kPa]", value=50.0)
    phi = st.sidebar.slider("相対湿度 φ", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.sidebar.header("2. グラフ描画範囲")
    q_min = st.sidebar.number_input("最小流量 Q_min [L/min]", value=1.0)
    q_max = st.sidebar.number_input("最大流量 Q_max [L/min]", value=30.0)

    with st.sidebar.expander("⚙️ 初期条件・物理定数の詳細設定"):
        p0_hPa = st.number_input("大気圧 p0 [hPa]", value=1013.25, format="%.2f")
        T0_C = st.number_input("基準温度 T0 [℃]", value=0.0, format="%.2f")
        rho0_std = st.number_input("基準空気密度 ρ0 [kg/m³]", value=1.293, format="%.3f")
        mu0_e5 = st.number_input("基準粘度 μ0 [×10⁻⁵ Pa・s]", value=1.716, format="%.3f")
        C = st.number_input("サザーランド定数 C [K]", value=111.0, format="%.1f")

    # 物理特性の事前計算
    p0 = p0_hPa * 100.0
    T0 = T0_C + 273.15
    mu0 = mu0_e5 * 1e-5
    T = T0 + t
    p1 = p0 + p_gage * 1000.0
    p_s = 610.78 * (10 ** ((7.5 * t) / (t + 237.3)))
    rho0 = rho0_std * (T0 / T) * ((p0 - 0.5040 * phi * p_s) / p0)
    rho1 = rho0 * (p1 / p0)
    mu = mu0 * ((T0 + C) / (T + C)) * ((T / T0) ** 1.5)
    nu1 = mu / rho1

# ==========================================
# 2. データベース ＆ コア計算エンジン
# ==========================================
fitting_db = {
    "直管 (Tube)": {"cat": "pipe"},
    "急拡大 (Expansion)": {"cat": "expansion"},
    "急縮小 (Contraction)": {"cat": "contraction"},
    "ソケット (Union)": {"cat": "fitting", "zeta": 0.5},
    "T字 (直角曲がり)": {"cat": "fitting", "zeta": 1.5},
    "エルボ (Elbow)": {"cat": "fitting", "zeta": 1.0}
}

def calc_total_loss_system(Q_Lmin_ANR, elements, p1_val, rho0_val, nu1_val):
    if Q_Lmin_ANR <= 0 or len(elements) == 0: return 0.0, []
    
    Q_m3s_ANR = Q_Lmin_ANR * 1e-3 / 60.0
    total_dp_kPa = 0.0
    current_p = p1_val
    details = []
    
    for item in elements:
        cat = fitting_db[item["name"]]["cat"]
        rho_current = rho0_val * (current_p / p0)
        
        if cat == "pipe":
            d_m = item["d1"] * 1e-3
            A = np.pi * (d_m**2) / 4.0
            u_local = Q_m3s_ANR / A * (p0 / current_p)
            Re_local = d_m * u_local / nu1_val
            u0_local = Q_m3s_ANR / A
            
            lambda_f = 64.0 / Re_local if Re_local < 2300 else 0.3164 / (Re_local ** 0.25)
            L = item["length"]
            inside_sqrt = current_p**2 - lambda_f * (L / d_m) * rho0_val * p0 * (u0_local**2)
            dp_Pa = current_p - np.sqrt(inside_sqrt) if inside_sqrt > 0 else 0
            
            details.append({"cat": cat, "name": item["name"], "u": u_local, "Re": Re_local, "dp": dp_Pa/1000.0, "lambda_f": lambda_f, "p_in": current_p, "L": L, "d_m": d_m, "u0": u0_local, "rho": rho_current})
            
        elif cat == "expansion" or cat == "contraction":
            d_in_m = item["d1"] * 1e-3
            d_out_m = item["d2"] * 1e-3
            A_in = np.pi * (d_in_m**2) / 4.0
            A_out = np.pi * (d_out_m**2) / 4.0
            
            A1 = min(A_in, A_out)
            A2 = max(A_in, A_out)
            u1 = Q_m3s_ANR / A1 * (p0 / current_p)
            
            if cat == "expansion":
                zeta_e0 = 1.0 
                zeta = zeta_e0 * (1 - A1/A2)**2
            else:
                zeta_c0 = 0.47 
                zeta = zeta_c0 * (1 - A1/A2)
                
            dynamic_p = rho_current * (u1**2) / 2.0
            dp_Pa = zeta * dynamic_p
            
            details.append({"cat": cat, "name": item["name"], "u1": u1, "A1": A1, "A2": A2, "zeta": zeta, "rho": rho_current, "dp": dp_Pa/1000.0, "p_in": current_p})
            
        elif cat == "fitting":
            d_m = item["d1"] * 1e-3
            A = np.pi * (d_m**2) / 4.0
            u_local = Q_m3s_ANR / A * (p0 / current_p)
            zeta = item["zeta"]
            dynamic_p = rho_current * (u_local**2) / 2.0
            dp_Pa = zeta * dynamic_p
            
            details.append({"cat": cat, "name": item["name"], "u": u_local, "zeta": zeta, "rho": rho_current, "dp": dp_Pa/1000.0, "p_in": current_p})

        current_p -= dp_Pa
        total_dp_kPa += (dp_Pa / 1000.0)
        
    return total_dp_kPa, details

def balance_parallel_flow(Q_total, elements_A, elements_B, p1_val, rho0_val, nu1_val):
    if Q_total <= 0: return 0.0, 0.0, 0.0
    Q_A_min = 0.0
    Q_A_max = Q_total
    tolerance = 0.001
    
    for _ in range(100):
        Q_A = (Q_A_min + Q_A_max) / 2.0
        Q_B = Q_total - Q_A
        dp_A, _ = calc_total_loss_system(Q_A, elements_A, p1_val, rho0_val, nu1_val)
        dp_B, _ = calc_total_loss_system(Q_B, elements_B, p1_val, rho0_val, nu1_val)
        diff = dp_A - dp_B
        
        if abs(diff) < tolerance: break
        if diff > 0: Q_A_max = Q_A 
        else: Q_A_min = Q_A         
            
    return Q_A, Q_B, dp_A

def calc_3area_system(Q_total, p1_val, rho0_val, nu1_val):
    if Q_total <= 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], []
    
    dp_in, details_in = calc_total_loss_system(Q_total, st.session_state.elements_in, p1_val, rho0_val, nu1_val)
    p_branch_Pa = p1_val - (dp_in * 1000.0)
    
    Q_A, Q_B, dp_par = balance_parallel_flow(Q_total, st.session_state.elements_A, st.session_state.elements_B, p_branch_Pa, rho0_val, nu1_val)
    p_merge_Pa = p_branch_Pa - (dp_par * 1000.0)
    
    dp_out, details_out = calc_total_loss_system(Q_total, st.session_state.elements_out, p_merge_Pa, rho0_val, nu1_val)
    
    total_dp = dp_in + dp_par + dp_out
    return total_dp, dp_in, dp_par, dp_out, Q_A, Q_B, p_branch_Pa, p_merge_Pa, details_in, details_out

# ==========================================
# 3. UI共通コンポーネント関数
# ==========================================
def update_defaults(prefix, item_id, elements_list, idx):
    new_name = st.session_state[f"{prefix}_n_{item_id}"]
    elements_list[idx]["name"] = new_name
    
    if fitting_db[new_name]["cat"] == "fitting":
        default_zeta = fitting_db[new_name]["zeta"]
        elements_list[idx]["zeta"] = default_zeta
        if f"{prefix}_z_{item_id}" in st.session_state:
            st.session_state[f"{prefix}_z_{item_id}"] = default_zeta

def render_flow_builder(elements_list, prefix):
    for i, item in enumerate(elements_list):
        c_idx, c_type, c_p1, c_p2, b1, b2, b3 = st.columns([0.4, 2.2, 1.5, 1.5, 0.6, 0.6, 0.6])
        
        with c_idx: 
            st.markdown(f"<div style='margin-top:30px;'><b>{i+1}</b></div>", unsafe_allow_html=True)
            
        with c_type:
            item["name"] = st.selectbox(
                "種類", 
                list(fitting_db.keys()), 
                index=list(fitting_db.keys()).index(item["name"]), 
                key=f"{prefix}_n_{item['id']}",
                on_change=update_defaults,
                args=(prefix, item["id"], elements_list, i)
            )
        
        cat = fitting_db[item["name"]]["cat"]
        with c_p1:
            if cat in ["pipe", "fitting"]:
                item["d1"] = st.number_input("内径 φd [mm]", min_value=0.1, value=float(item.get("d1", 4.0)), step=1.0, format="%.1f", key=f"{prefix}_d1_{item['id']}")
                item["d2"] = item["d1"]
            else:
                item["d1"] = st.number_input("入口 φd_in [mm]", min_value=0.1, value=float(item.get("d1", 4.0)), step=1.0, format="%.1f", key=f"{prefix}_din_{item['id']}")
        with c_p2:
            if cat == "pipe":
                item["length"] = st.number_input("長さ L [m]", min_value=0.0, value=float(item.get("length", 0.5)), format="%.2f", key=f"{prefix}_l_{item['id']}")
            elif cat == "fitting":
                item["zeta"] = st.number_input("損失係数 ζ", min_value=0.0, value=float(item.get("zeta", fitting_db[item["name"]]["zeta"])), format="%.2f", key=f"{prefix}_z_{item['id']}")
            else:
                item["d2"] = st.number_input("出口 φd_out [mm]", min_value=0.1, value=float(item.get("d2", 4.0)), step=1.0, format="%.1f", key=f"{prefix}_dout_{item['id']}")
                
        margin_html = "<div style='margin-top:28px;'></div>"
        with b1:
            st.markdown(margin_html, unsafe_allow_html=True)
            if st.button("↑", key=f"{prefix}_u_{item['id']}", disabled=(i == 0)):
                elements_list.insert(i - 1, elements_list.pop(i)); st.rerun()
        with b2:
            st.markdown(margin_html, unsafe_allow_html=True)
            if st.button("↓", key=f"{prefix}_d_{item['id']}", disabled=(i == len(elements_list) - 1)):
                elements_list.insert(i + 1, elements_list.pop(i)); st.rerun()
        with b3:
            st.markdown(margin_html, unsafe_allow_html=True)
            if st.button("❌", key=f"{prefix}_x_{item['id']}"):
                elements_list.pop(i); st.rerun()
                
    if st.button("➕ パーツを追加", key=f"{prefix}_add"):
        elements_list.append({"id": str(uuid.uuid4()), "name": "直管 (Tube)", "d1": 4.0, "d2": 4.0, "length": 0.1, "zeta": 0.0})
        st.rerun()

def render_html_diagram(elements_list):
    if len(elements_list) == 0:
        st.info("パーツがありません")
        return
        
    html_code = '<div style="display: flex; align-items: center; padding: 20px 10px; overflow-x: auto; background: white; border: 1px solid #ccc; border-radius: 8px;">'
    html_code += '<div style="font-weight: bold; margin-right: 15px;">FLOW →</div>'
    for i, item in enumerate(elements_list):
        cat = fitting_db[item["name"]]["cat"]
        if cat == "pipe":
            d_mm, L = item["d1"], item["length"]
            width = max(60, min(150, int(L * 100)))
            height = max(12, min(40, int(d_mm * 4)))
            html_code += f'<div style="width: {width}px; height: {height}px; border: 2px solid #0070C0; position: relative; display: flex; align-items: center; justify-content: center; z-index: 1;">'
            html_code += '<div style="position: absolute; width: 100%; border-top: 2px dotted #0070C0;"></div>'
            html_code += f'<div style="position: absolute; top: -20px; font-size: 11px; font-style: italic;">L={L}m</div>'
            html_code += f'<div style="position: absolute; bottom: -20px; font-size: 11px;">φ={d_mm}</div></div>'
        elif cat in ["expansion", "contraction"]:
            d_in, d_out = item["d1"], item["d2"]
            height = max(20, min(50, int(max(d_in, d_out) * 5)))
            color = "#FF8C00" if cat == "expansion" else "#8A2BE2"
            html_code += f'<div style="width: 50px; height: {height}px; border: 2px dashed {color}; background: #fafafa; margin: 0 -5px; position: relative; display: flex; align-items: center; justify-content: center; z-index: 10;">'
            html_code += f'<div style="position: absolute; width: 100%; border-top: 2px dotted {color};"></div>'
            html_code += f'<div style="position: absolute; bottom: -20px; font-size: 10px;">{d_in}→{d_out}</div></div>'
        else:
            d_mm = item["d1"]
            height = max(20, min(50, int(d_mm * 5)))
            html_code += f'<div style="width: 40px; height: {height}px; border: 2px solid #FF0000; background: white; margin: 0 -10px; position: relative; display: flex; align-items: center; justify-content: center; z-index: 10;">'
            html_code += '<div style="position: absolute; width: 100%; border-top: 2px dotted #FF0000;"></div>'
            html_code += f'<div style="position: absolute; bottom: -20px; font-size: 11px;">φ={d_mm}</div></div>'
    html_code += '</div>'
    st.markdown(html_code, unsafe_allow_html=True)

def render_fluid_properties_proof():
    """環境・流体条件の基礎数値がどう計算されたかを提示する"""
    with st.expander("🌍 【基礎物性】環境・流体条件の事前計算プロセス", expanded=False):
        st.markdown("**① 飽和水蒸気圧 $p_s$ (Tetensの式)**")
        st.latex(rf"p_s = 610.78 \times 10^{{\frac{{7.5 \times {t}}}{{{t} + 237.3}}}} = {p_s:.2f} \ \text{{Pa}}")

        st.markdown("**② 基準状態の湿り空気密度 $\\rho_0$**")
        st.latex(rf"\rho_0 = {rho0_std:.3f} \times \frac{{{T0_C+273.15:.2f}}}{{{t+273.15:.2f}}} \times \frac{{{p0:.1f} - 0.5040 \times {phi} \times {p_s:.2f}}}{{{p0:.1f}}} = {rho0:.4f} \ \text{{kg/m}}^3")

        st.markdown("**③ 空気粘度 $\\mu$ (サザーランドの式)**")
        st.latex(rf"\mu = {mu0_e5*1e-5:.3e} \times \frac{{{T0_C+273.15:.2f}+{C}}}{{{t+273.15:.2f}+{C}}} \times \left(\frac{{{t+273.15:.2f}}}{{{T0_C+273.15:.2f}}}\right)^{{1.5}} = {mu:.4e} \ \text{{Pa・s}}")

        st.markdown("**④ 供給口での空気密度 $\\rho_1$ と動粘度 $\\nu_1$**")
        st.latex(rf"\rho_1 = \rho_0 \times \frac{{p_1\text{{(絶対圧)}}}}{{p_0}} = {rho0:.4f} \times \frac{{{p1:.1f}}}{{{p0:.1f}}} = {rho1:.4f} \ \text{{kg/m}}^3")
        st.latex(rf"\nu_1 = \frac{{\mu}}{{\rho_1}} = \frac{{{mu:.4e}}}{{{rho1:.4f}}} = {nu1:.4e} \ \text{{m}}^2\text{{/s}}")


def render_math_proof(details):
    for i, res in enumerate(details):
        with st.expander(f"パーツ {i+1} : {res['name']} の計算証明"):
            st.markdown(f"**【突入時の空気状態】** 絶対圧力 $p_{{in}} = {res['p_in']:.1f}$ Pa $\\rightarrow$ 局所密度 $\\rho = {res['rho']:.4f}$ kg/m³")
            if res["cat"] == "pipe":
                st.markdown("**【管摩擦損失 (Eq.2)】**")
                st.latex(rf"u_0 = {res['u0']:.2f}\ \text{{m/s}} \quad (\text{{ANR換算流速}})")
                st.latex(rf"Re = {res['Re']:.1f}, \quad \lambda = {res['lambda_f']:.4f}")
                st.latex(r"\Delta p_\lambda = p_{in} - \sqrt{p_{in}^2 - \lambda \frac{L}{d} \rho_0 p_0 u_0^2}")
                st.latex(rf"= \mathbf{{{res['dp']:.4f}\ \text{{kPa}}}}")
            elif res["cat"] == "expansion":
                st.markdown("**【急拡大損失 (Eq.3, 7)】** (ボルダ・カルノーの式)")
                st.latex(rf"u_1 = {res['u1']:.2f}\ \text{{m/s}}")
                st.latex(rf"\zeta_e = \left(1 - \frac{{{res['A1']:.2e}}}{{{res['A2']:.2e}}}\right)^2 = {res['zeta']:.3f}")
                st.latex(rf"\Delta p_e = \zeta_e \frac{{\rho u_1^2}}{{2}} = \mathbf{{{res['dp']:.4f}\ \text{{kPa}}}}")
            elif res["cat"] == "contraction":
                st.markdown("**【急縮小損失 (Eq.4, 8)】**")
                st.latex(rf"u_1 = {res['u1']:.2f}\ \text{{m/s}}")
                st.latex(rf"\zeta_c = 0.5 \left(1 - \frac{{{res['A1']:.2e}}}{{{res['A2']:.2e}}}\right) = {res['zeta']:.3f}")
                st.latex(rf"\Delta p_c = \zeta_c \frac{{\rho u_1^2}}{{2}} = \mathbf{{{res['dp']:.4f}\ \text{{kPa}}}}")
            elif res["cat"] == "fitting":
                st.markdown("**【一般継手損失】**")
                st.latex(rf"u = {res['u']:.2f}\ \text{{m/s}}, \quad \zeta = {res['zeta']:.2f}")
                st.latex(rf"\Delta p_f = \zeta \frac{{\rho u^2}}{{2}} = \mathbf{{{res['dp']:.4f}\ \text{{kPa}}}}")


# ==========================================
# 4. モード別 メインレイアウト
# ==========================================

if mode == "➡️ 直列モデル (1本道)":
    st.title("空気圧配管系の流動損失シミュレーター (直列)")
    
    if 'elements_series' not in st.session_state:
        st.session_state.elements_series = [
            {"id": str(uuid.uuid4()), "name": "直管 (Tube)", "d1": 4.0, "d2": 4.0, "length": 0.5, "zeta": 0.0},
            {"id": str(uuid.uuid4()), "name": "急拡大 (Expansion)", "d1": 4.0, "d2": 8.0, "length": 0.0, "zeta": 0.0},
            {"id": str(uuid.uuid4()), "name": "直管 (Tube)", "d1": 8.0, "d2": 8.0, "length": 0.5, "zeta": 0.0}
        ]
        
    st.subheader("① 流路モデル・ビルダー")
    render_flow_builder(st.session_state.elements_series, "s")
    
    st.subheader("② 構築された流路モデル")
    render_html_diagram(st.session_state.elements_series)
    
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.subheader("③ 流量と全圧力損失のシミュレーション")
        Q_array = np.linspace(q_min, q_max, 30)
        
        results_series = [calc_total_loss_system(q, st.session_state.elements_series, p1, rho0, nu1) for q in Q_array]
        dp_array = [res[0] for res in results_series]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(Q_array, dp_array, marker='o', color='#1f77b4', label="Series Total Loss")
        ax.set_title("Flow Rate Q vs Total Pressure Loss dP", fontsize=12)
        ax.set_xlabel("Flow Rate Q [L/min (ANR)]", fontsize=10)
        ax.set_ylabel("Total Pressure Loss dP [kPa]", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
        
        csv_lines = ["流量 Q [L/min],全圧力損失 dP [kPa]"]
        for q, dp in zip(Q_array, dp_array):
            csv_lines.append(f"{q:.2f},{dp:.3f}")
        csv_data = "\n".join(csv_lines)
        
        st.download_button(
            label="📥 グラフデータをCSVでダウンロード (Excel対応)",
            data=csv_data.encode("utf-8-sig"),
            file_name="series_simulation_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c_right:
        st.subheader("④ モデル計算の証明")
        target_Q = st.number_input("確認したい流量 Q を入力 [L/min (ANR)]", value=15.0, key="q_ser")
        total_dp, details = calc_total_loss_system(target_Q, st.session_state.elements_series, p1, rho0, nu1)
        st.success(f"**システム全体の圧力損失 Δp = {total_dp:.3f} kPa**")
        
        # ▼ここに追加！▼
        render_fluid_properties_proof()
        render_math_proof(details)

elif mode == "🔀 並列・分岐モデル (3エリア拡張版)":
    st.title("空気圧配管系の流動損失シミュレーター (並列 3エリア版)")
    st.markdown("共通入口から流入した空気が分岐点でルートA・Bに分かれ、再び合流して共通出口へ向かうシステム全体をシミュレーションします。")
    
    if 'elements_in' not in st.session_state:
        st.session_state.elements_in = [{"id": str(uuid.uuid4()), "name": "直管 (Tube)", "d1": 6.0, "d2": 6.0, "length": 0.5, "zeta": 0.0}]
    if 'elements_out' not in st.session_state:
        st.session_state.elements_out = [] 
    if 'elements_A' not in st.session_state:
        st.session_state.elements_A = [{"id": str(uuid.uuid4()), "name": "直管 (Tube)", "d1": 4.0, "d2": 4.0, "length": 1.0, "zeta": 0.0}]
    if 'elements_B' not in st.session_state:
        st.session_state.elements_B = [{"id": str(uuid.uuid4()), "name": "直管 (Tube)", "d1": 6.0, "d2": 6.0, "length": 1.0, "zeta": 0.0}]

    st.subheader("🟨 エリア1: 共通入口 (コンプレッサ → 分岐点)")
    render_flow_builder(st.session_state.elements_in, "in")
    render_html_diagram(st.session_state.elements_in)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔀 エリア2: 並列部 (分岐点 → 合流点)")
    col_A, col_B = st.columns(2)
    with col_A:
        st.markdown("#### 🟦 ルートA の構成")
        render_flow_builder(st.session_state.elements_A, "A")
        render_html_diagram(st.session_state.elements_A)
    with col_B:
        st.markdown("#### 🟩 ルートB の構成")
        render_flow_builder(st.session_state.elements_B, "B")
        render_html_diagram(st.session_state.elements_B)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🟪 エリア3: 共通出口 (合流点 → 大気開放)")
    render_flow_builder(st.session_state.elements_out, "out")
    render_html_diagram(st.session_state.elements_out)
        
    st.markdown("---")
    
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.subheader("③ 全流量に対する圧力損失のシミュレーション")
        Q_array = np.linspace(q_min, q_max, 20)
        
        results_parallel = [calc_3area_system(q, p1, rho0, nu1) for q in Q_array]
        dp_array = [res[0] for res in results_parallel]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(Q_array, dp_array, marker='s', color='#d62728', label="Total Pressure Loss (3 Areas)")
        ax.set_title("Total Flow Rate Q vs Total Pressure Loss dP", fontsize=12)
        ax.set_xlabel("Total Flow Rate Q_total [L/min (ANR)]", fontsize=10)
        ax.set_ylabel("Total Pressure Loss dP [kPa]", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
        
        csv_lines = ["全流量 Q_total [L/min],全圧力損失 dP_total [kPa],入口損失 dP_in [kPa],並列部損失 dP_par [kPa],出口損失 dP_out [kPa],ルートA流量 Q_A [L/min],ルートB流量 Q_B [L/min]"]
        for q, res in zip(Q_array, results_parallel):
            tot_dp, dp_in, dp_par, dp_out, Q_A, Q_B = res[0], res[1], res[2], res[3], res[4], res[5]
            csv_lines.append(f"{q:.2f},{tot_dp:.3f},{dp_in:.3f},{dp_par:.3f},{dp_out:.3f},{Q_A:.2f},{Q_B:.2f}")
        csv_data = "\n".join(csv_lines)
        
        st.download_button(
            label="📥 グラフデータをCSVでダウンロード (Excel対応)",
            data=csv_data.encode("utf-8-sig"),
            file_name="parallel_simulation_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c_right:
        st.subheader("④ 流量分配の自動計算 ＆ 計算証明")
        target_Q = st.number_input("システム全体の流量 Q_total を入力 [L/min (ANR)]", value=30.0, key="q_par")
        total_dp, dp_in, dp_par, dp_out, Q_A, Q_B, p_branch_Pa, p_merge_Pa, details_in, details_out = calc_3area_system(target_Q, p1, rho0, nu1)
        
        st.success(f"**システム全体の圧力損失 Δp_total = {total_dp:.3f} kPa**")
        st.info(f"【内訳】 共通入口: {dp_in:.3f} kPa ＋ 並列部: {dp_par:.3f} kPa ＋ 共通出口: {dp_out:.3f} kPa")
        st.info(f"**並列部の流量分配: ルートA = {Q_A:.2f} L/min / ルートB = {Q_B:.2f} L/min**")
        
        _, details_A = calc_total_loss_system(Q_A, st.session_state.elements_A, p_branch_Pa, rho0, nu1)
        _, details_B = calc_total_loss_system(Q_B, st.session_state.elements_B, p_branch_Pa, rho0, nu1)
        
        # ▼ここに追加！▼
        render_fluid_properties_proof()
        
        if len(details_in) > 0:
            st.markdown("##### 🟨 共通入口 の計算証明")
            render_math_proof(details_in)
            
        st.markdown("##### 🟦 ルートA の計算証明")
        render_math_proof(details_A)
        
        st.markdown("##### 🟩 ルートB の計算証明")
        render_math_proof(details_B)
        
        if len(details_out) > 0:
            st.markdown("##### 🟪 共通出口 の計算証明")
            render_math_proof(details_out)

elif mode == "📖 使い方・Tips集 (FAQ)":
    st.title("📖 シミュレーターの使い方・Tips集")
    st.markdown("実験装置のモデリングや、流体力学のゼミ発表で迷いやすいポイントをまとめました。研究室での議論や引き継ぎの参考にしてください。")
    
    with st.expander("💡 1. 急拡大・急縮小パーツの「長さ L」はどう扱うの？", expanded=True):
        st.markdown("""
        **結論：長さの入力は不要です（空欄や `0` で構いません）。**
        
        * **理由：** 急拡大（Expansion）や急縮小（Contraction）による圧力損失は、管の長さによる摩擦ではなく、「太さがガクッと変わった瞬間に発生する渦」が原因だからです。
        * **入力のコツ：** ビルダーで回路を組むときは、`[直管]` $\\rightarrow$ `[急拡大]` $\\rightarrow$ `[直管]` のように、前後の「直管」パーツにそれぞれの長さを入力し、段差の瞬間だけを急拡大・急縮小パーツとして間に挟んでください。
        """)
        
    with st.expander("💡 2. 急拡大・急縮小の「入口」と「出口」の数値の入れ方は？"):
        st.markdown("""
        空気が進む向き（FLOW $\\rightarrow$）に合わせて、**段差の手前と奥の管の内径**を入力します。
        
        * **急拡大の場合：** `入口` に細い方の内径、`出口` に太い方の内径を入力。
        * **急縮小の場合：** `入口` に太い方の内径、`出口` に細い方の内径を入力。
        * ※アプリがこれらの面積比から、自動的に損失係数 $\\zeta$ を理論計算（ボルダ・カルノーの定理など）してくれます。
        """)

    with st.expander("💡 3. 並列分岐の「T字管」はどう配置するのが正解？"):
        st.markdown("""
        **共通入口エリアの最後に「T字」を置くのはNGです。**
        
        * **理由：** 共通入口の最後にT字を置くと、ルートAに行く空気も、ルートBに行く空気も、分岐する前に平等に同じ直角曲がりの大ダメージ（$\\zeta=1.5$）を受けてしまうことになり、現実の物理現象とズレてしまいます。
        * **正しい組み方：** 共通入口は直管のままで終わらせ、**分岐した直後の「ルートA」と「ルートB」それぞれの先頭パーツ**としてT字やエルボを配置します。
        * 直進して通り抜けるルートには $\\zeta=0.4$ 程度の小さな値を、直角に曲がるルートには $\\zeta=1.5$ の大きな値を手入力することで、正しい流量分配がシミュレーションできます。
        """)

    with st.expander("💡 4. 「急縮小」と「急収縮」って違うの？"):
        st.markdown("""
        **物理現象としては全く同じです。**
        
        * **急縮小（きゅうしゅくしょう）：** 流体力学の専門書やJIS規格などで使われる**公式な学術用語**です。「管の形状」が細くなることに着目した言葉です。
        * **急収縮（きゅうしゅうしゅく）：** 空気圧の現場などでよく使われる表現です。空気が持つ「圧縮性」によって、体積自体も縮むイメージからこう呼ばれることがあります。
        * **アドバイス：** ゼミの発表資料や卒業論文では、一貫して公式用語である **「急縮小」** に統一しておくのが最も無難で安全です。
        """)

    with st.expander("💡 5. グラフの横軸にある「ANR」って何？"):
        st.markdown("""
        **ANR（基準大気状態）**とは、「温度20℃、絶対圧101.3 kPa（約1気圧）、相対湿度65%」にある空気状態の世界共通ルールです。
        
        * 空気は圧縮性を持つため、配管内の圧力が変わると体積も流速も変わってしまいます。
        * そのため、「もしこの空気を大気開放して1気圧に戻したら、何リットルになるか？」という **ANR基準（L/min ANR）** に統一して流量を評価します。
        * アプリの裏側では、このANR流量と各地点の圧力を照らし合わせ、配管の奥へ行くほど空気が膨張して流速が上がる現象を厳密に計算しています。
        """)
    with st.expander("💡 6. シミュレーターで使用している公式・計算モデル集"):
        st.markdown("""
        本シミュレーターでは、以下の流体力学の公式を用いて圧力損失を計算しています。
        （※流体は圧縮性を考慮し、各パーツ通過時の圧力降下に伴う密度・流速の変化を逐次計算するモデルを採用しています）
        """)
        
        st.markdown("#### ① レイノルズ数 $Re$ と 流速 $u$ の算出")
        st.markdown("ANR基準の流量から、その地点の圧力 $p$ に応じた実際の流速 $u$ を求め、流れの性質（層流・乱流）を判定します。")
        st.latex(r"u = \frac{Q_{ANR}}{A} \times \frac{p_0}{p}")
        st.latex(r"Re = \frac{d \cdot u}{\nu}")
        
        st.markdown("#### ② 直管の管摩擦損失 (Darcy-Weisbachの式)")
        st.markdown("直管を流れる際の壁面との摩擦による損失です。レイノルズ数 $Re$ によって管摩擦係数 $\\lambda$ の算出式（条件）が切り替わります。")
        st.latex(r"\Delta p_\lambda = \lambda \frac{L}{d} \frac{\rho u^2}{2}")
        st.markdown("- **層流 ($Re < 2300$) の場合：** ハーゲン・ポアズイユの法則に基づく厳密解")
        st.latex(r"\lambda = \frac{64}{Re}")
        st.markdown("- **乱流 ($Re \ge 2300$) の場合：** ブラジウス (Blasius) の実験式")
        st.latex(r"\lambda = \frac{0.3164}{Re^{0.25}}")
        
        st.markdown("#### ③ 急拡大損失 (Borda-Carnotの定理)")
        st.markdown("管が急に太くなる箇所で、流れが剥離して渦が発生することによる損失です。")
        st.latex(r"\Delta p_e = \zeta_e \frac{\rho u_1^2}{2} \quad \left(u_1\text{は拡大前の流速}\right)")
        st.latex(r"\zeta_e = \left(1 - \frac{A_1}{A_2}\right)^2 \quad \left(A_1 < A_2\right)")

        st.markdown("#### ④ 急縮小損失")
        st.markdown("管が急に細くなる箇所で、流れが一度縮流（ベナ・コントラクタ）を形成した後に拡大する際の損失です。")
        st.latex(r"\Delta p_c = \zeta_c \frac{\rho u_1^2}{2} \quad \left(u_1\text{は縮小後の流速}\right)")
        st.latex(r"\zeta_c = 0.5 \left(1 - \frac{A_1}{A_2}\right) \quad \left(A_1 < A_2\right)")

        st.markdown("#### ⑤ 一般継手（エルボ・T字・ソケット等）の損失")
        st.markdown("継手の形状ごとに定められた固有の損失係数 $\\zeta$ （実験値等）を用いて計算します。")
        st.latex(r"\Delta p_f = \zeta \frac{\rho u^2}{2}")
