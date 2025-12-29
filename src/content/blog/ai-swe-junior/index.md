---
title: "My First SWE Job No Longer Exists. What's the Future for Junior Engineers?"
pubDate: 2025-12-29
description: "AI has made my first SWE job obsolete. What's the future for junior engineers?"
---

## My previous job does not exist anymore

*TLDR; what took ~4 months for a 3-person team to complete would take around three weeks with a proper agent harness.*

My first software experience goes back to ~2022, where I landed a software engineering internship, which turned into a part-time developer role while I was finishing my Master's.

For most of 2023 and 2024, my job mainly consisted of coding as a member of a technical team. The process was as follows: think about an architecture with the team, align on an implementation plan and estimate timelines, scope tasks and dispatch them through the team and go on about the tasks in two-week sprints.

My first full-time mission, as I was writing my Master's thesis, consisted of writing a new EVM client, in a new niche DSL language (Cairo). It was a bit hard to skill up as the resources were scarce and the tooling was non-existent. On the other hand, the core topic of the work, which is writing a compliant EVM implementation, is not a very complex task: you can just pull the specification of the EVM (written in Python), a 20k test vector, and simply *"rewrite"* the EVM in your language. It's basically a loop in which you're:
- Getting the next feature to implement
- Write unit tests for your code
- Test said feature e2e using test vectors
- Fix bugs until tests pass
- Fetch next feature and start again

Most of the complexity of the task actually consisted of sharing context, information and tools around the team: what is the side effect of this function on the global system? What historical event caused this branching condition in this function; and is it still relevant for a proving engine? How do I debug this failing test? I would end up reading specifications for hours, just to make sure I understood the system before coding it.

This took around 4 months, most of which were spent on writing code, tests, upskilling in Rust, navigating big codebases and benchmarking different architectures to find the right one.

One year later, I believe this same task could be done in about three weeks by one person; rather than a 3-person team in 4 months.

## Maybe it's always obvious in hindsight

*TLDR; implementation times fall to 0, feedback loop is 10x faster; saving time on iterations helps converge to the right design faster.*

Some would argue that of course, now that I know the full context of what I needed to do to complete the job; it's easier to lower time estimates. To that, I will counter-argue that:
- Time spent writing tests falls to near 0. The only thing you need is extensive test scenarios.
- Time spent researching and understanding the spec can be divided by 5 through efficient use of agents as Q&A assistants. It's now a lot faster to understand how to do things right.
- Time spent upskilling in Rust can be lowered to the bare minimum required to understand the syntax, tooling and most common programming flows; so a few days at most.
- Side tooling can be parallelised with core effortlessly and no longer stand in the hot path.

What's left is basically just orchestrating all of these tasks and providing clear context for coding agents to be able to code with the DSL that's not in the training set, navigate codebases properly, and define proper success metrics.

Iterations are now a lot faster: we can throw away code easily; we can refactor code easily; and we can test specific architectures with barely any overhead cost.

## Would a junior engineer stand any chance today?

*TLDR: Hiring needs will evolve, as will the nature of work. But there are limits to what we can handle at once.*

Some pessimists will say that juniors are cooked and won't be able to find jobs in this market. I disagree, as I believe that good engineers will be considerably enhanced, and bad ones will be weakened. However, that job will be entirely different. Throw away the old interview processes with their useless Leetcode, and start screening for what will matter next: *Can you learn how to make the right decisions on how to lead development of a specific piece of software ?*

You don't need years of experience to be able to read architecture proposals submitted to you and weigh the pros and cons on how to achieve the desired goal the best way. Neither do you need years of experience to review a final product, that you'll evaluate as if you were a user: *does it achieve the desired goal?* In the future, all engineers will be tech leads.

What *could* happen, though, is that the supply of junior roles will dry up because there will no longer be enough tasks to do anyway.

That, I can't really predict, but I believe that we'll see an explosion of software rather than a shrink; because it will become *so* easy to write software that all the projects that were previously put on the side because of resources constraints will now be considered. In that scenario; why wouldn't you give it to a junior and see what happens, if you have the resources for it?

There's a limit to how many agents a human can manage. Constant context switching is tiring, de-focuses from other priorities, and I'd gladly delegate someone else the task of overseeing something that I think is of lower priority so that I can focus on higher priority tasks.

AI-enhanced engineering has other bottlenecks:
- Products are generated faster than we can keep up, and this will only increase. You either take your hands off the brake; or you'll need help to review that.
- LLMs have small context windows. Humans have good long-term memory. If no one is following what the AI is doing, you'll end up not remembering what has been done and what needs to be done next. How can you orchestrate something that you know nothing about?
- There's a limit to parallelising work, because we're terrible at context switching. I used to be able to lock in a 6-hour coding window without being interrupted. Now, I'm switching between reviewing a new CLI, a test suite, an architecture document and a new frontend for a tool. At the end of the day, I feel completely exhausted.
- Other humans are still extremely valuable to discuss with; because they have different backgrounds, expertise, and interests.

Technical teams will still exist. Humans will still be in the middle. Agents will be delegated all the grunt work. The biggest disruption will be the organisation and nature of the work, and there might be a temporary drought in the SWE labor market, until we can get a clearer view on efficient processes, how to get teams to work in this new paradigm, and what to look for when hiring.
