"""The first phase of research:
1. Breaks the user question down the (subquestion, keyword)
2. Loop through the (subquestion, keyword) zip,
    2a. web search with the keyword. 
    2b. The results indexed to local vector database
    2c. LLM (evaluator) inspect the findings by looking at the retrieved web search results except
    for the first time, it will look at the whole web search.
    2d. Stop when certain conditions are met"""
import sys
from agents.planner_v2 import QuestionDecomposer
from agents.evaluator import ResearchEvaluator

if __name__ == "__main__":
    # initiate question decomposer
    decomposer = QuestionDecomposer()
    
    user_question = "Real Purposes and significances of NASA Artemis project"

    results = decomposer.decompose_to_dict(user_question)

    sub_kw_pairs = results.get("pairs")

    if not sub_kw_pairs:
        print(f"The subquestion and keyword pairs are not found")
        sys.exit(1)

    # initiate the evaluator
    evaluator = ResearchEvaluator(chroma_collection_name="artemis_space")

    for i, (subquestion, keyword) in enumerate(sub_kw_pairs):
        app = evaluator.build_graph()

        state = {
            "subquestion": subquestion,
            "current_query": keyword,
            "new_search_results": [],
            "iteration": 1,
            "max_iterations": 4,
            "evaluation_history": []
        }

        for step in app.stream(state):
            for node, value in step.items():
                print(f"\n--- {node.upper()} ---")
                if node == "evaluate":
                    res = value["evaluation_history"][-1]
                    print(f"Decision: {'STOP' if res.stop_search else 'CONTINUE'}")
                    print(f"Reason: {res.reasoning}")
