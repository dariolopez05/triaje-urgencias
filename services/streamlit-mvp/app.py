import streamlit as st


st.set_page_config(page_title="TriageIA", layout="wide")

st.title("TriageIA - Sistema de Triaje Manchester")
st.caption("Curso de especializacion IA-BD 25/26 - Proyecto 3")

st.info(
    "Placeholder de la Iteracion 0. "
    "El dashboard final (audio, triage C1-C5, auditoria etica) se implementa en la Iteracion 7."
)

st.subheader("Niveles Manchester")

cols = st.columns(5)
levels = [
    ("C1 Rojo", "0 min", "Emergencia", "#d62828"),
    ("C2 Naranja", "10 min", "Muy urgente", "#f77f00"),
    ("C3 Amarillo", "60 min", "Urgente", "#fcbf49"),
    ("C4 Verde", "120 min", "Menos urgente", "#43aa8b"),
    ("C5 Azul", "240 min", "No urgente", "#277da1"),
]

for col, (name, time, desc, color) in zip(cols, levels):
    col.markdown(
        f"""
        <div style="border-left: 6px solid {color}; padding: 8px 12px;
                    background: #f8f9fa; border-radius: 4px;">
          <strong>{name}</strong><br>
          <span style="color:#555">{time}</span><br>
          <small>{desc}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
