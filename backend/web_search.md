### Overall Flow

When a user query is received, the modified flow proceeds as follows:

1. **Initial Processing and Retrieval**  
   The agent checks for greetings, performs query condensation, and verifies customer specifications. It then retrieves and reranks documents based on the query.

2. **Document Answer Generation**  
   The agent uses a decision prompt to decide whether a sufficient answer can be generated from the top documents.  
   - If the answer is satisfactory, it generates a document-based answer and updates the conversation history.  
   - If not, it transitions to the **Web Search Fallback Branch**.

3. **Web Search Fallback Branch**  
   When the document-based answer is inadequate, the agent:
   - Prompts the user with a message such as, “I couldn’t find sufficient information in my documents. Would you like me to search the web for more details?”  
   - Sets a flag (e.g., `state.waiting_for_Internet_Access_confirmation = True`) so that the next user message triggers a confirmation handler.

4. **Handling User Response for Web Search**  
   On the next turn, the agent checks if it is waiting for web search confirmation:
   - If the user responds affirmatively ("yes", "ok", etc.), the agent proceeds to perform a web search.  
   - If the user declines, the agent politely ends the conversation, perhaps inviting the user to ask another question.

5. **Performing the Web Search and Answer Generation**  
   In case of affirmative confirmation, the agent:
   - Invokes a new tool (e.g., `WebSearchTool`, analogous to `RetrieverTool`) that searches the web for the query.
   - Optionally reranks the results similarly to document reranking.
   - Generates an answer using a dedicated web answer prompt template and then updates the conversation history.

---

### Detailed Node Design

#### 1. Ask the User if They Want a Web Search

When no answer is generated from the documents, a new node (named, for example, `ask_Internet_Access`) gets executed:

```python
def ask_Internet_Access(state: AgentState) -> AgentState:
    # Flag that the next message should be a confirmation response.
    state.waiting_for_Internet_Access_confirmation = True
    state.response = (
        "I couldn’t find sufficient information from our documents to answer your query. "
        "Would you like me to search the web for more details? Please respond with 'yes' or 'no'."
    )
    state.should_stop = True  # Halt further processing until user confirms.
    return state
```

#### 2. Handle the User’s Response for Web Search Confirmation

A new node inspects the next user input and decides how to proceed:

```python
def handle_Internet_Access_confirmation(state: AgentState) -> AgentState:
    # Process the confirmation response.
    confirmation = state.question.strip().lower()
    if confirmation in ["yes", "y", "sure", "ok"]:
        state.waiting_for_Internet_Access_confirmation = False
        state.Internet_Access_query = state.question  # Optionally preserve or re-use the original query.
        return Internet_Access(state)
    else:
        state.response = "Okay, please let me know if you have any other questions."
        state.waiting_for_Internet_Access_confirmation = False
        state.should_stop = True
        return state
```

#### 3. Execute the Web Search

This node accesses a web search tool to fetch live results:

```python
def Internet_Access(state: AgentState) -> AgentState:
    try:
        # Invoke a web search tool analogous to RetrieverTool but for online search.
        state.web_documents = WebSearchTool.run({
            "query": state.Internet_Access_query,
            "k": 10
        })
        if not state.web_documents:
            state.response = "The web search did not return any useful results."
            state.should_stop = True
            return state
    except Exception as e:
        state.response = "An error occurred while performing the web search."
        state.should_stop = True
        return state

    # Optionally, add web result reranking here.
    return generate_web_response(state)
```

#### 4. Generate Answer from Web Results

Using a dedicated prompt, the agent constructs an answer from web search results:

```python
def generate_web_response(state: AgentState) -> AgentState:
    TOP_K_WEB = 3
    top_web_docs = state.web_documents[:TOP_K_WEB]
    context = "\n\n".join(doc.page_content for doc in top_web_docs)
    
    # Use a dedicated prompt template for web search context (e.g., WEB_ANSWER_PROMPT).
    _input = (
        RunnableLambda(lambda x: {"context": context, "question": x.Internet_Access_query})
        | WEB_ANSWER_PROMPT
        | llm_o3_mini
        | StrOutputParser()
    )
    answer = _input.invoke(state)
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip()
    state.response = answer
    return state
```

---

### Graph Integration and Edge Updates

To integrate these nodes into your existing state graph:

1. **After the Decision Node:**  
   Update the graph so that after `decide_to_generate`:
   - If `state.answer_generated_from_document_store == "pass"`, proceed to the `generate` node.
   - Else, transition to the new node `ask_Internet_Access`.

2. **Handling the Next Turn Afterwards:**  
   When `state.waiting_for_Internet_Access_confirmation` is `True`, the agent should invoke the `handle_Internet_Access_confirmation` node on the next user input instead of treating it as a completely new query.

3. **Adding Edges Example:**

```python
builder.add_conditional_edges(
    "decide_to_generate",
    lambda s: "generate" if s.answer_generated_from_document_store == "pass" else "ask_Internet_Access"
)

# In the subsequent turn, if the flag is set, use the confirmation handling node:
builder.add_node("handle_Internet_Access_confirmation", handle_Internet_Access_confirmation)
builder.add_conditional_edges(
    "handle_Internet_Access_confirmation",
    lambda s: "Internet_Access" if s.response.lower() in ["yes", "y", "sure", "ok"] else "update_history"
)

builder.add_node("Internet_Access", Internet_Access)
builder.add_edge("Internet_Access", "update_history")
```

---

### Summary

- **Document Answer Success:**  
  The agent generates an answer based on retrieved documents if the decision node indicates sufficient context exists.

- **Document Answer Failure:**  
  If documents do not provide enough information, the agent prompts the user for a web search confirmation.

- **Web Search Branch:**  
  Upon affirmative user response, the web search tool collects live results, which are reranked (if needed), and used to generate a final answer.

- **State Graph Integration:**  
  Appropriate modifications to the state graph ensure that the conversation flow branches seamlessly between document retrieval and web search based on user confirmation.

This design preserves your existing document-based processing pipeline while providing an optimized pathway to perform live web searches as a fallback.
