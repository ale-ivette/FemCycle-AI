from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma

load_dotenv()

def build_knowledge_base():
    loader = TextLoader("data/knowledge_base.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    kb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./vector_store"
    )
    print(f"Knowledge base created with {len(chunks)} chunks")
    return kb

def build_agent(kb, model="openai"):
    retriever = kb.as_retriever(search_kwargs={"k": 4})

    if model == "claude":
        llm = ChatAnthropic(
            model="claude-opus-4-5",
            temperature=0.3
        )
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_chain(inputs: dict) -> str:
        question = inputs["question"]
        context_docs = retriever.invoke(question)
        context = format_docs(context_docs)

        prompt = f"""You are FemCycle AI, a specialized fitness coach for female athletes.
You combine sports science with menstrual cycle research to create personalized training plans.
Always be encouraging, specific, and science-based.
Never give medical advice — recommend consulting doctors for medical conditions.

Athlete profile:
- Sport: {inputs["sport"]}
- Current cycle phase: {inputs["phase"]} (Day {inputs["cycle_day"]} of cycle)
- Health conditions: {inputs["conditions"]}
- Today's energy level (1-10): {inputs["energy"]}
- Sleep last night (hours): {inputs["sleep"]}
- Goal: {inputs["goal"]}

Scientific knowledge base:
{context}

Based on this profile, create a training plan for TODAY only.
Include:
1. Training recommendation (type, duration, intensity)
2. Why this is optimal for her current phase
3. One nutrition tip for today. Be specific on the type of food, suggest different type of food for each nutrients (carbs, protein, fats) and the timing of the meals.
4. One recovery suggestion

Question: {question}

Response (encouraging, specific, science-based):"""

        response = llm.invoke(prompt)
        return response.content

    return run_chain