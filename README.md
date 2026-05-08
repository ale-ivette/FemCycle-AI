# 🌸 FemCycle AI

AI-powered training coach for female athletes that personalizes workouts 
based on menstrual cycle phase.

Built by a marathon runner and open water swimmer who needed this tool 
and couldn't find it.

## What it does

- Detects your current cycle phase automatically
- Generates a personalized daily training plan based on your hormonal phase
- Adapts recommendations for your sport, energy level, sleep, and health conditions
- Supports two AI models: OpenAI GPT-3.5 and Claude Opus — compare their responses
- Science-based recommendations grounded in female physiology research

## Who it's for

Female athletes who want to train smarter by understanding how their 
cycle affects performance, recovery, and energy.

Especially useful for: marathon runners, open water swimmers, triathletes, 
and any woman who has ever felt like training plans were designed for men.

## How to use it

1. Enter your last period start date in the sidebar
2. Set your cycle length, sport, and health conditions
3. Rate your energy and sleep
4. Click "Generate my plan for today"
5. Ask follow-up questions in the chat

## Architecture

RAG pipeline that retrieves relevant scientific knowledge about female
physiology and sport, combined with your personal athlete profile,
to generate specific science-based recommendations via LLM.
Your profile + Cycle phase
↓
ChromaDB retrieves relevant science
↓
LLM generates personalized plan
↓
Your training plan for today

## Model comparison

FemCycle AI supports two models. RAGAS evaluation results on female 
athletic training questions:

| Metric | OpenAI GPT-3.5 | Claude Opus | Winner |
|---|---|---|---|
| Faithfulness | 0.217 | 0.263 | Claude |
| Answer Relevancy | 0.895 | 0.907 | Claude |
| Context Precision | 0.800 | 0.800 | Tie |
| Context Recall | 0.600 | 0.600 | Tie |

Claude shows better faithfulness — critical for health-related recommendations 
where accuracy matters more than creativity.

## Stack

Python · LangChain · ChromaDB · OpenAI API · Anthropic Claude API · 
Streamlit · RAGAS

## Run locally

1. Clone the repo
2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
3. Install dependencies
pip install -r requirements.txt
4. Add your API keys to `.env`
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
5. Run the app
streamlit run app.py

## Next
- Weekly training plans with competition date targets
- Apple Health and Garmin integration
- Nutrition recommendations with medical condition context
- Deploy with persistent user profiles