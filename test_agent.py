from datetime import date
from backend.cycle import calculate_phase
from backend.agent import build_knowledge_base, build_agent

# Test your own data
last_period = date(2025, 4, 20)  # Change to your actual date
cycle_length = 28
sport = "Marathon running and open water swimming"
conditions = "None"
energy = 7
sleep = 7.5
goal = "Train smart around my cycle for peak performance"

# Calculate phase
cycle_info = calculate_phase(last_period, cycle_length)
print(f"\nToday is Day {cycle_info['cycle_day']} of your cycle")
print(f"Phase: {cycle_info['emoji']} {cycle_info['phase']}")
print(f"{cycle_info['description']}")
print(f"Days until next period: {cycle_info['days_until_next_period']}")

# Build agent and get recommendation
print("\nBuilding knowledge base...")
kb = build_knowledge_base()
agent = build_agent(kb)

question = f"Give me today's training plan for a {sport} athlete in {cycle_info['phase']} phase."

print("\nGenerating your personalized training plan...")
response = agent({
    "question": question,
    "sport": sport,
    "phase": cycle_info["phase"],
    "cycle_day": cycle_info["cycle_day"],
    "conditions": conditions,
    "energy": energy,
    "sleep": sleep,
    "goal": goal
})

print("\n" + "="*50)
print("YOUR TRAINING PLAN FOR TODAY")
print("="*50)
print(response)