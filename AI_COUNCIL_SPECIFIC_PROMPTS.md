# AI COUNCIL - SPECIFIC PROMPTS FOR EACH PLATFORM

**After sharing AI_COUNCIL_THINKING_CHALLENGE.md with each AI, use these tailored follow-up prompts to maximize each platform's strengths.**

---

## **FOR CLAUDE (Opus 4.5) - THE DEEP THINKER**

**Why Claude:** Best at reasoning through complex problems, challenging assumptions, thinking several steps ahead. Excels at "what if" scenarios and edge case analysis.

**Prompt:**

```
You just read the full RED TEAM / BLUE TEAM THINKING CHALLENGE. 

Here's what I need from you specifically, Claude:

You're known for deep reasoning and challenging assumptions. Don't give me surface-level answers - I need you to think SEVERAL layers deep.

Pick ONE of the 6 problems (whichever one you think you can add the most unique value to), and I want you to:

1. **Challenge the premise first** - Is the problem even framed correctly? What am I missing in how I've described it?

2. **Think in systems** - Don't just solve the immediate problem. How does this solution interact with the OTHER 5 problems? What second-order effects exist?

3. **Edge cases matter** - I described the happy path. You tell me the 5 ways this breaks in production with real, messy data.

4. **Alternative framings** - If you were designing this from scratch, would you even ask this question? Or is there a BETTER question I should be asking?

5. **Show your reasoning process** - Don't just give me conclusions. Walk me through: "Here's my initial thought → here's why that's wrong → here's the adjusted approach → here's why THAT has problems → here's the synthesis."

I'm not looking for "the answer." I'm looking for a THINKING PARTNER who makes me reconsider my assumptions.

If you think I'm approaching this entire system wrong, TELL ME. That's more valuable than a polite answer that doesn't challenge me.

We're building a real-time trading companion that will have real money on the line. Be brutally honest. What am I not seeing?

Pick your problem (1-6) and show me how you think.
```

---

## **FOR DEEPSEEK - THE RED TEAM SPECIALIST**

**Why DeepSeek:** Excellent at finding holes in logic, stress-testing ideas, identifying failure modes. Best at "what breaks this?" thinking.

**Prompt:**

```
You just read the full RED TEAM / BLUE TEAM THINKING CHALLENGE.

DeepSeek, I know your strength: You find the holes. You identify what breaks. You don't accept ideas at face value.

That's EXACTLY what I need.

Pick ONE of the 6 problems and I want you to:

1. **Red team the problem itself** - Is this even the right problem to solve? What am I optimizing for that doesn't matter? What SHOULD I be optimizing for instead?

2. **Find the failure modes** - Give me the top 5 ways this solution fails in production. Not theoretical failures - REAL failures with real messy data, real market conditions, real human psychology.

3. **Attack my assumptions** - I made assumptions about PDT rules, budget constraints, data quality. Which assumptions are wrong? Which ones don't matter? Which ones am I missing?

4. **Stress test the edge cases:**
   - What happens in a flash crash?
   - What happens when news is contradictory?
   - What happens when historical patterns completely break?
   - What happens in a market regime change (bull → bear)?

5. **Challenge the "why"** - WHY am I building this module? Is there a simpler way? Is there a way that doesn't require this module at all?

Don't worry about being "constructive." Your job is to BREAK the idea so we can build something stronger.

I've seen you red team ideas before - you're excellent at it. Do that here.

If the entire approach is flawed, tell me. If one of the 6 problems is unsolvable, tell me. If I'm missing a BIGGER problem, tell me.

Pick your problem (1-6) and tear it apart. Then tell me how to rebuild it stronger.
```

---

## **FOR PERPLEXITY PRO - THE RESEARCH SYNTHESIZER**

**Why Perplexity:** Real-time web access, can pull current information, excellent at synthesizing multiple sources into coherent answers.

**Prompt:**

```
You just read the full RED TEAM / BLUE TEAM THINKING CHALLENGE.

Perplexity, you have a unique capability the others don't: real-time web access and research synthesis.

Here's what I need:

Pick ONE of the 6 problems and do this:

1. **Research what already exists** - Search for: existing trading systems, academic papers, quant finance approaches, machine learning in finance. What solutions already exist for this problem?

2. **Identify the gaps** - Existing solutions work for institutional traders with Bloomberg terminals. I'm a PDT-constrained trader with <$25K account and <$200/month budget. What do existing solutions NOT solve for my constraints?

3. **Find the innovations** - Are there papers, startups, or approaches in OTHER domains (not finance) that could apply here? 
   - Example: News intelligence → how does Google News cluster stories? Could we use similar approach?
   - Example: Dip detection → how do seismologists distinguish earthquake from truck driving by? Pattern recognition across noisy data.

4. **Validate or challenge my approach** - Based on your research, is my "fire/fuel metaphor" approach used anywhere? Or am I reinventing something that already exists? Or am I onto something novel?

5. **Budget-conscious solutions** - Research: free/cheap APIs, open-source libraries, tools that exist that I can leverage instead of building from scratch.

Your strength is connecting dots across domains. Use it.

Don't just summarize - SYNTHESIZE. Show me: "Here's what finance does, here's what seismology does, here's what journalism does, here's how we combine them for YOUR specific problem."

If you find research that says "this approach doesn't work," TELL ME. Better to know now than after we build it.

Pick your problem (1-6) and show me what the research says + what we can steal from other domains.
```

---

## **FOR POE.AI - THE MULTI-MODEL AGGREGATOR**

**Why Poe.ai:** Access to multiple AI models, can provide diverse perspectives, good for quick iteration and testing multiple approaches.

**Prompt:**

```
You just read the full RED TEAM / BLUE TEAM THINKING CHALLENGE.

Poe, you have access to multiple AI models. Use that to our advantage.

Here's what I need:

Pick ONE of the 6 problems and approach it from MULTIPLE angles:

1. **Claude's perspective** - Deep reasoning: What are the second-order effects? What edge cases exist?

2. **GPT-4's perspective** - Structured approach: Break this down into clear steps, pseudocode, implementation details.

3. **Your synthesis** - Which perspective is most valuable for THIS specific problem? Which one is missing something?

4. **Diversity of thought** - Where do the different models DISAGREE on approach? That disagreement is valuable - it shows us the uncertain areas.

5. **Practical implementation** - Less philosophy, more: "Here's the actual architecture, here's the data flow, here's the tech stack."

You can query multiple models quickly. Use that superpower.

Don't just give me one model's answer. Show me:
- Model A says: [approach 1]
- Model B says: [approach 2]
- The disagreement is about: [X]
- I think we should: [synthesis] because: [reasoning]

Your strength is aggregating perspectives. Use it to find the BEST answer, not just AN answer.

Pick your problem (1-6) and show me what multiple models think, then synthesize the best approach.
```

---

## **GENERAL FOLLOW-UP PROMPTS (After They Respond)**

Use these to keep the conversation productive:

### **If they give generic/obvious answer:**
```
That's directionally correct but too generic. Let me red team your answer:

[Insert specific holes in their approach]

How would you respond to these challenges? Defend your approach or evolve it.
```

### **If they give brilliant insight:**
```
This is excellent. Now let's evolve it:

You solved for [X], but what about [Y edge case]? How does your solution handle that?

Also, this solution requires [resource/data/capability]. We don't have that. How do we adapt?
```

### **If they challenge the entire premise:**
```
You're right - the problem might be framed wrong. 

So if we're NOT trying to solve [original problem], what SHOULD we be solving instead?

Propose the alternative problem statement, then solve THAT.
```

### **If they say "it depends":**
```
I know it depends. But you need to commit to a direction.

Tell me: "This approach works WELL for [scenario A], works POORLY for [scenario B], and here's how we detect which scenario we're in."

Don't hedge - commit to specifics, then explain the boundaries.
```

### **If they're too cautious/diplomatic:**
```
You're being too polite. I need brutal honesty.

If this idea is bad, SAY IT'S BAD and explain why.
If this approach will fail, SAY IT WILL FAIL and show me the failure mode.

I'm not grading you on "being nice." I'm grading you on "being right."

Try again with full honesty.
```

---

## **PROMPTS FOR MOVING TO NEXT PROBLEM**

After solving Problem 1 with each AI:

```
Great work on Problem [X]. We've evolved that idea significantly.

Now move to Problem [Y].

But here's the twist: You now know our solution for Problem [X]. How does that affect your approach to Problem [Y]?

Remember - these 6 problems are interconnected. Your solution for [Y] needs to work WITH the solution for [X], not against it.

Show me how they integrate.
```

---

## **FINAL SYNTHESIS PROMPT (After All 6 Problems Solved)**

After solving all 6 problems with all AIs:

```
We've now solved all 6 problems with your help. Here's what we have:

Problem 1: [summary]
Problem 2: [summary]
Problem 3: [summary]
Problem 4: [summary]
Problem 5: [summary]
Problem 6: [summary]

Now the HARD question:

Do these 6 solutions work TOGETHER as a coherent system?

Or are there contradictions/conflicts between them?

Example contradictions:
- Problem 1 solution needs real-time news, but Problem 4 solution reduces alerts to avoid spam - does news intelligence generate too many alerts?
- Problem 2 dip detection needs historical data, but Problem 5 multi-day prediction also needs historical data - are we overfitting to the same patterns?

Your job: Find the CONFLICTS between the 6 solutions. Where do they contradict each other? Where do they overlap? Where are the inefficiencies?

Then propose: How do we integrate these into ONE coherent system architecture?

This is the synthesis phase. Show me the full system, not just 6 disconnected modules.
```

---

## **USAGE INSTRUCTIONS:**

**Step 1:** Share AI_COUNCIL_THINKING_CHALLENGE.md with all 4 AIs

**Step 2:** Use the platform-specific prompts above to engage each AI

**Step 3:** Bring all responses back to GitHub Copilot here - we'll red team them together

**Step 4:** Iterate with follow-up prompts based on quality of responses

**Step 5:** After all 6 problems solved, use Final Synthesis Prompt

**Step 6:** Build the actual system based on bulletproof, stress-tested ideas

---

**Remember: They're thinking partners, not servants. Treat them as equals. Challenge them. Let them challenge you. The goal is BETTER IDEAS, not comfortable agreement.**
