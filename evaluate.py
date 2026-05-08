from dotenv import load_dotenv
from datetime import date
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from backend.agent import build_knowledge_base, build_agent

load_dotenv()

# Evaluation questions about female athletic training
EVAL_DATASET = [
    {
        "question": "What training should I do during my menstrual phase?",
        "ground_truth": "During menstrual phase, low intensity training is recommended such as yoga, walking, light swimming, and mobility work. High intensity intervals and heavy lifting should be avoided."
    },
    {
        "question": "When is the best time to do a personal record attempt?",
        "ground_truth": "The ovulation phase (days 14-16) is the best time for personal records as estrogen peaks, giving highest energy, strength, and pain tolerance."
    },
    {
        "question": "How should a marathon runner train during the luteal phase?",
        "ground_truth": "During luteal phase, marathon runners should do easy runs only, maintain base fitness, reduce volume by 30%, and avoid introducing new hard workouts."
    },
    {
        "question": "What nutrition considerations are important for athletes with diabetes?",
        "ground_truth": "Athletes with diabetes should monitor blood glucose more frequently during luteal phase when insulin resistance increases, carry fast-acting glucose during workouts, and consult an endocrinologist."
    },
    {
        "question": "How does sleep affect training recommendations by cycle phase?",
        "ground_truth": "During follicular and ovulation phases 7-8 hours is sufficient. During luteal and menstrual phases 8-9 hours is recommended as progesterone disrupts sleep and recovery is critical."
    }
]

def run_evaluation(model_name="openai"):
    print(f"\nEvaluating model: {model_name.upper()}")
    print("="*50)

    kb = build_knowledge_base()
    agent = build_agent(kb, model=model_name)

    # Common athlete profile for all questions
    profile = {
        "sport": "Marathon running and open water swimming",
        "phase": "Follicular",
        "cycle_day": 8,
        "conditions": "None",
        "energy": 7,
        "sleep": 7.5,
        "goal": "Train smart around my cycle"
    }

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    retriever = kb.as_retriever(search_kwargs={"k": 4})

    for item in EVAL_DATASET:
        question = item["question"]
        profile["question"] = question

        print(f"Q: {question[:60]}...")
        answer = agent(profile)
        retrieved_docs = retriever.invoke(question)
        context = [doc.page_content for doc in retrieved_docs]

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(item["ground_truth"])
        print(f"A: {answer[:80]}...\n")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    embeddings = OpenAIEmbeddings()
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    df = results.to_pandas()
    filename = f"eval_results_{model_name}.csv"
    df.to_csv(filename, index=False)

    scores = df
    print(f"\nRESULTS — {model_name.upper()}")
    print(f"Faithfulness:      {scores['faithfulness'].mean():.3f}")
    print(f"Answer Relevancy:  {scores['answer_relevancy'].mean():.3f}")
    print(f"Context Precision: {scores['context_precision'].mean():.3f}")
    print(f"Context Recall:    {scores['context_recall'].mean():.3f}")

    return df, results

if __name__ == "__main__":
    # Evaluate OpenAI
    df_openai, results_openai = run_evaluation("openai")

    # Evaluate Claude
    df_claude, results_claude = run_evaluation("claude")

    # Compare
    print("\n" + "="*50)
    print("COMPARISON: OpenAI GPT-3.5 vs Claude Opus")
    print("="*50)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for metric in metrics:
        openai_score = df_openai[metric].mean()
        claude_score = df_claude[metric].mean()
        winner = "Claude" if claude_score > openai_score else "OpenAI"
        print(f"{metric:20} OpenAI: {openai_score:.3f} | Claude: {claude_score:.3f} | Winner: {winner}")