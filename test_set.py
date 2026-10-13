import time
import ollama
import rag
from datetime import datetime


# ============================================================
# THE TEST QUESTIONS
# ============================================================
# Each conversation is a list of turns. The chat history is reset
# between conversations, so follow-up questions only see their own chat.
#
# "expect" is a word that must appear in a correct answer.
# "expect": None means the documents do NOT contain the answer,
# so the correct behavior is to say "I don't have enough information".

CONVERSATIONS = [
    # --- 9 normal questions ---
    [{"q": "Which planet is the largest in our solar system?", "expect": "Jupiter"}],
    [{"q": "What is Olympus Mons?", "expect": "volcano"}],
    [{"q": "How long does Mercury take to orbit the Sun?", "expect": "87.97"}],
    [{"q": "Which planet rotates on its side?", "expect": "Uranus"}],
    [{"q": "How long is the Great Wall of China?", "expect": "21,196"}],
    [{"q": "How many trees are in the Amazon Rainforest?", "expect": "390 billion"}],
    [{"q": "How big is the Sahara Desert?", "expect": "9.2 million"}],
    [{"q": "When did the Great Barrier Reef become a UNESCO World Heritage Site?", "expect": "1981"}],
    [{"q": "What do blue whales eat?", "expect": "krill"}],

    # --- 2 questions that should be refused ---
    [{"q": "What is the height of Mount Everest?", "expect": None}],
    [{"q": "Who won the 2018 football World Cup?", "expect": None}],

    # --- 2 follow-up conversations (2 turns each) ---
    [
        {"q": "Tell me about Jupiter", "expect": "Jupiter"},
        {"q": "How many moons does it have?", "expect": "95"},
    ],
    [
        {"q": "Tell me about the Sahara Desert", "expect": "Sahara"},
        {"q": "Which animals live there?", "expect": "fennec"},
    ],
]


# ============================================================
# SMALL HELPER FUNCTIONS
# ============================================================

def keyword_in_chunks(keyword, matches):
    # Did the right word appear in any chunk we sent to the model?
    for match in matches:
        if keyword.lower() in match["text"].lower():
            return True
    return False


def clean_cell(text, limit):
    # Makes text safe to put inside a markdown table cell:
    # no line breaks, no | characters, and not too long.
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("|", "/")
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


# ============================================================
# RUN ONE QUESTION THROUGH THE SAME STEPS AS THE REAL CHAT LOOP
# ============================================================

def run_turn(turn, history, docs, bm25):
    question = turn["q"]
    expect = turn["expect"]

    start = time.time()

    search_question = rag.rewrite_question(question, history)
    matches = rag.hybrid_retrieve(search_question, docs, bm25)
    matches = rag.rerank(search_question, matches)
    matches = matches[:rag.MAX_CHUNKS_TO_MODEL]
    prompt = rag.build_prompt(search_question, matches, history)

    response = ollama.generate(
        model=rag.GEN_MODEL,
        prompt=prompt,
        options={"temperature": 0, "num_predict": 300}
    )

    seconds = time.time() - start

    answer = response["response"]
    abstained = rag.looks_like_refusal(answer)

    if expect is None:
        # Should refuse. Passing means it refused.
        retrieval_hit = None
        passed = abstained
    else:
        retrieval_hit = keyword_in_chunks(expect, matches)
        passed = (not abstained) and (expect.lower() in answer.lower())

    # Keep a copy of what the model saw, so failures can be studied
    # from the report file without running anything again.
    chunk_info = []
    for match in matches:
        info = {}
        info["source"] = match["source"]
        info["heading"] = match.get("heading", "")
        info["text"] = match["text"]
        info["score"] = float(match.get("rerank_score", 0))
        chunk_info.append(info)

    prompt_tokens = response.get("prompt_eval_count", 0)
    answer_tokens = response.get("eval_count", 0)

    result = {}
    result["question"] = question
    result["search_question"] = search_question
    result["expect"] = expect
    result["answer"] = answer.strip()
    result["abstained"] = abstained
    result["retrieval_hit"] = retrieval_hit
    result["passed"] = passed
    result["seconds"] = round(seconds, 1)
    result["prompt_tokens"] = prompt_tokens
    result["answer_tokens"] = answer_tokens
    result["chunks"] = chunk_info
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    docs = rag.load_documents()
    bm25 = rag.build_bm25_index(docs)
    rag.index_documents()

    results = []
    number = 0

    for conversation in CONVERSATIONS:
        history = []

        for turn in conversation:
            number = number + 1
            result = run_turn(turn, history, docs, bm25)
            results.append(result)

            # Same rule as the real chat loop: only remember real answers.
            if not result["abstained"]:
                history.append((turn["q"], result["answer"]))

            if result["passed"]:
                label = "PASS"
            else:
                label = "FAIL"

            print("")
            print(str(number) + ". [" + label + "] " + result["question"])
            print("   searched for: " + result["search_question"])
            print("   answer: " + result["answer"][:150].replace("\n", " "))
            print("   time: " + str(result["seconds"]) + "s | tokens: "
                  + str(result["prompt_tokens"] + result["answer_tokens"]))

    # ---------- work out the totals ----------
    total = len(results)
    total_passed = 0
    answerable = 0
    answerable_passed = 0
    retrieval_hits = 0
    refusals = 0
    refusals_passed = 0
    total_seconds = 0
    total_prompt_tokens = 0
    total_answer_tokens = 0

    for result in results:
        total_seconds = total_seconds + result["seconds"]
        total_prompt_tokens = total_prompt_tokens + result["prompt_tokens"]
        total_answer_tokens = total_answer_tokens + result["answer_tokens"]

        if result["passed"]:
            total_passed = total_passed + 1

        if result["expect"] is None:
            refusals = refusals + 1
            if result["passed"]:
                refusals_passed = refusals_passed + 1
        else:
            answerable = answerable + 1
            if result["passed"]:
                answerable_passed = answerable_passed + 1
            if result["retrieval_hit"]:
                retrieval_hits = retrieval_hits + 1

    average_seconds = round(total_seconds / total, 1)
    average_tokens = int((total_prompt_tokens + total_answer_tokens) / total)

    # ---------- print the summary in the terminal ----------
    print("")
    print("=" * 50)
    print("SCORE: " + str(total_passed) + " / " + str(total))
    print("Answerable questions passed: " + str(answerable_passed) + " / " + str(answerable))
    print("Right chunk reached the model: " + str(retrieval_hits) + " / " + str(answerable))
    print("Correct refusals: " + str(refusals_passed) + " / " + str(refusals))
    print("Average time per question: " + str(average_seconds) + "s")
    print("Average tokens per question: " + str(average_tokens))

    # ---------- build the report file ----------
    now = datetime.now()
    file_name = "test_results_" + now.strftime("%Y-%m-%d_%H-%M") + ".md"

    lines = []
    lines.append("# Test results")
    lines.append("")
    lines.append("Run on: " + now.strftime("%Y-%m-%d %H:%M"))
    lines.append("")

    lines.append("## Settings")
    lines.append("")
    lines.append("- Answer model: " + str(rag.GEN_MODEL))
    lines.append("- Embedding model: " + str(rag.EMBED_MODEL))
    lines.append("- Chunks sent to the model per question: " + str(rag.MAX_CHUNKS_TO_MODEL))
    lines.append("- Chunks in the database: " + str(len(docs)))
    lines.append("- Questions: " + str(total))
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Measure | Value |")
    lines.append("|---|---|")
    lines.append("| Score | " + str(total_passed) + " / " + str(total) + " |")
    lines.append("| Answerable questions passed | " + str(answerable_passed) + " / " + str(answerable) + " |")
    lines.append("| Right chunk reached the model | " + str(retrieval_hits) + " / " + str(answerable) + " |")
    lines.append("| Correct refusals | " + str(refusals_passed) + " / " + str(refusals) + " |")
    lines.append("| Average time per question | " + str(average_seconds) + " s |")
    lines.append("| Total time | " + str(round(total_seconds, 1)) + " s |")
    lines.append("| Average tokens per question | " + str(average_tokens) + " |")
    lines.append("| Total prompt tokens | " + str(total_prompt_tokens) + " |")
    lines.append("| Total answer tokens | " + str(total_answer_tokens) + " |")
    lines.append("")
    lines.append("Time includes rewriting the question, searching, reranking and writing the answer.")
    lines.append("Prompt tokens are what was sent to the model. Answer tokens are what it wrote.")
    lines.append("")

    lines.append("## All questions")
    lines.append("")
    lines.append("| # | Result | Retrieval | Time (s) | Prompt tokens | Answer tokens | Question | Searched for | Answer (start) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    row = 0
    for result in results:
        row = row + 1

        if result["passed"]:
            label = "PASS"
        else:
            label = "FAIL"

        if result["retrieval_hit"] is None:
            retrieval = "n/a"
        elif result["retrieval_hit"]:
            retrieval = "hit"
        else:
            retrieval = "miss"

        lines.append(
            "| " + str(row)
            + " | " + label
            + " | " + retrieval
            + " | " + str(result["seconds"])
            + " | " + str(result["prompt_tokens"])
            + " | " + str(result["answer_tokens"])
            + " | " + clean_cell(result["question"], 80)
            + " | " + clean_cell(result["search_question"], 80)
            + " | " + clean_cell(result["answer"], 80)
            + " |"
        )

    lines.append("")

    # ---------- details for every failed question ----------
    lines.append("## Failures in detail")
    lines.append("")

    failure_count = 0
    row = 0
    for result in results:
        row = row + 1
        if result["passed"]:
            continue

        failure_count = failure_count + 1

        lines.append("### " + str(row) + ". " + result["question"])
        lines.append("")
        if result["expect"] is None:
            lines.append("- Expected: a refusal")
        else:
            lines.append("- Expected the answer to contain: " + str(result["expect"]))
        lines.append("- Searched for: " + result["search_question"])
        lines.append("- Full answer: " + clean_cell(result["answer"], 600))
        lines.append("- Chunks the model saw:")
        for chunk in result["chunks"]:
            lines.append(
                "  - [" + str(chunk["source"]) + " / " + str(chunk["heading"])
                + "] score " + str(round(chunk["score"], 2)) + ": "
                + clean_cell(chunk["text"], 200)
            )
        lines.append("")

    if failure_count == 0:
        lines.append("No failures in this run.")
        lines.append("")

    report = open(file_name, "w", encoding="utf-8")
    report.write("\n".join(lines))
    report.close()

    print("Report saved to " + file_name)


if __name__ == "__main__":
    main()