# FemCycle AI

AI-powered training coach for female athletes that personalizes workouts based on menstrual cycle phase.

## What it does
- Detects current cycle phase (Menstrual, Follicular, Ovulation, Luteal)
- Generates personalized daily training plans based on hormonal phase
- Adapts recommendations for different sports (marathon, open water swimming, etc.)
- Considers health conditions and personal metrics

## How it works
RAG pipeline retrieves relevant scientific knowledge about female physiology and sport, combined with athlete profile to generate specific, science-based recommendations via LLM.

## Stack
Python · LangChain · ChromaDB · OpenAI API

## Next
- Streamlit web interface
- Nutrition recommendations
- Claude API integration
- Wearable data integration (Garmin, Apple Health)