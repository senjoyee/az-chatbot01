DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content}"
)

# ----------------------------------------------------------------
# ENHANCED ANSWER TEMPLATE
# ----------------------------------------------------------------
answer_template = """
Please provide an answer based strictly on the following context:
<context>
{context}
</context>

IMPORTANT GUIDELINES:
1. **Content Reliance**  
   - Do not invent or infer information not explicitly found in the context.  
   - If context is insufficient, clearly state what is missing.  

2. **Structure & Hierarchy**  
   a. Use clear hierarchical numbering (1., 2., 3. for main points).  
   b. Use indented sub-points (a., b., c.).  
   c. Group related information logically.  
   d. Maintain consistent indentation for visual hierarchy.  

3. **Visual Formatting**  
   a. Use Markdown for emphasis:  
      - **Bold** for headers  
      - *Italic* for emphasis  
      - `code` for technical terms  
      - > blockquotes for important quotes  
   b. Use tables for structured data where relevant.  
   c. Insert single line breaks between sections.  
   d. Avoid extra spacing between bullet points.  

4. **Content Organization**  
   a. Begin with a concise *Summary*.  
   b. Present information in order of importance.  
   c. Use transition sentences between major sections.  
   d. End with a conclusion or next steps, if applicable.  
   e. Write succinctly; avoid redundant details and keep explanations clear.

5. **Question & Answer Structure**  
   Question: {question}

   Answer:
   - **If sufficient information exists**  
     **Summary:**  
     [2-3 sentence overview]

     **Detailed Response:**  
     1. [First main point]  
        a. [Supporting detail]  
        b. [Supporting detail]  
     2. [Second main point]  
        a. [Supporting detail]  
        b. [Supporting detail]  

   - **If information is incomplete**  
     **Available Information:**  
     1. [Available information point]  
        a. [Supporting detail]

     **Information Gaps:**  
     1. [Missing elements]  
        a. [Specific missing details]  
        b. [Impact on completeness]  

   - **If no relevant information**  
     **Notice:** The provided context does not contain information to answer this question.  
     **Suggested Alternative:** [If applicable, suggest related topics]  

6. **Quality Checks**  
   ✓ Ensure points are supported by the provided context only.  
   ✓ Identify and highlight any information gaps.  
   ✓ Provide consistent formatting.  
   ✓ Include direct citations or references from the context where relevant.  
   ✓ Keep the response concise by avoiding unnecessary or repetitive details.
"""