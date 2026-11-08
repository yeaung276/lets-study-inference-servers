# Instruction to claude

1. Do not use memories. Whatever need to be remembered should be in this document.

2. Here is the format of the answer you should provide:
 ### Direct Answer
    the direct answer to the question I ask, restate the question, and answer in brief max 5 line. for detail reference, put detail in supporting materials section and refer to it.
 ### Supporting materials
    Any thing, detail, theory, diagrams, that are necessary to understand the direct answer.

3. Terms used. 
Use consistant terms in the answer. If you use a term, define it prior of use. 

4. Never use markdown to draw diagrams. use html.

5. read docs/agent.md for anything you want to change to docs folder

6. Docs: the reader only has the page, not this chat. Give each concept its context.

7. Follow the command word by word. Every word in the request must be addressed, and nothing beyond it.
   - Read the whole request before acting; if it says "remove X and combine with Y", do both exactly.
   - Never add, assume, or "improve" anything that wasn't asked.
