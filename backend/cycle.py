from datetime import date

def calculate_phase(last_period_date: date, cycle_length: int = 28) -> dict:
    today = date.today()
    days_since_period = (today - last_period_date).days
    cycle_day = (days_since_period % cycle_length) + 1

    if cycle_day <= 5:
        phase = "Menstrual"
        description = "Rest and recovery. Your body is doing important work."
        color = "#E57373"
        emoji = "🌙"
    elif cycle_day <= 13:
        phase = "Follicular"
        description = "Rising energy. Perfect for challenging workouts."
        color = "#81C784"
        emoji = "🌱"
    elif cycle_day <= 16:
        phase = "Ovulation"
        description = "Peak performance. Go for your personal records."
        color = "#FFD54F"
        emoji = "⚡"
    else:
        phase = "Luteal"
        description = "Steady and strong. Focus on technique and recovery."
        color = "#64B5F6"
        emoji = "🌊"

    return {
        "phase": phase,
        "cycle_day": cycle_day,
        "description": description,
        "color": color,
        "emoji": emoji,
        "days_until_next_period": cycle_length - cycle_day + 1
    }