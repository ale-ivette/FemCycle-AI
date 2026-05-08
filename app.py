import streamlit as st
from datetime import date, timedelta
from backend.cycle import calculate_phase
from backend.agent import build_knowledge_base, build_agent

st.set_page_config(
    page_title="FemCycle AI",
    page_icon="🌊",
    layout="wide"
)

st.title("🌸 FemCycle AI")
st.caption("Your AI training coach — personalized for your cycle")

# Model selector
col_model1, col_model2 = st.columns([2, 3])
with col_model1:
    model_choice = st.radio(
        "AI Model",
        ["OpenAI GPT-3.5", "Claude Opus"],
        horizontal=True
    )
model = "claude" if model_choice == "Claude Opus" else "openai"

# Initialize agent once
if "kb" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.kb = build_knowledge_base()

agent = build_agent(st.session_state.kb, model=model)

# Sidebar — athlete profile
with st.sidebar:
    st.header("Your Profile")

    last_period = st.date_input(
        "Last period start date",
        value=date.today() - timedelta(days=14)
    )

    cycle_length = st.slider(
        "Cycle length (days)",
        min_value=21,
        max_value=35,
        value=28
    )

    sport = st.selectbox(
        "Primary sport",
        ["Marathon running", "Open water swimming", "Triathlon",
         "Cycling", "CrossFit", "Yoga", "General fitness"]
    )

    conditions = st.multiselect(
        "Health conditions (optional)",
        ["None", "Diabetes", "Anemia", "Thyroid condition",
         "Endometriosis", "PCOS"],
        default=["None"]
    )

    energy = st.slider("Energy level today (1-10)", 1, 10, 7)
    sleep = st.slider("Sleep last night (hours)", 4.0, 10.0, 7.5, 0.5)

    goal = st.text_input(
        "Your goal",
        value="Train smart and perform at my best"
    )

# Main area — cycle info
cycle_info = calculate_phase(last_period, cycle_length)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cycle Day", cycle_info["cycle_day"])
with col2:
    st.metric("Phase", f"{cycle_info['emoji']} {cycle_info['phase']}")
with col3:
    st.metric("Days to next period", cycle_info["days_until_next_period"])
with col4:
    st.metric("Cycle length", f"{cycle_length} days")

st.info(f"**{cycle_info['phase']} Phase:** {cycle_info['description']}")

st.divider()

# Training plan
st.subheader("Today's Training Plan")

if st.button("Generate my plan for today", type="primary"):
    with st.spinner("Creating your personalized plan..."):
        question = f"Give me today's training plan for a {sport} athlete in {cycle_info['phase']} phase."

        response = agent({
            "question": question,
            "sport": sport,
            "phase": cycle_info["phase"],
            "cycle_day": cycle_info["cycle_day"],
            "conditions": ", ".join(conditions),
            "energy": energy,
            "sleep": sleep,
            "goal": goal
        })

        st.success(response)
        st.session_state.last_plan = response

# Chat
st.divider()
st.subheader("Ask FemCycle AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask anything about training, nutrition, or recovery...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent({
                "question": question,
                "sport": sport,
                "phase": cycle_info["phase"],
                "cycle_day": cycle_info["cycle_day"],
                "conditions": ", ".join(conditions),
                "energy": energy,
                "sleep": sleep,
                "goal": goal
            })
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})